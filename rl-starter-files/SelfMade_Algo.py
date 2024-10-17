from model import PolicyNetwork, QvalueNetwork, ValueNetwork, Discriminator
from replay_memory import Memory, Transition #.model and .algos replay memory are files from DIAYN code that need to be imported

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_ac.algos.base import BaseAlgo


class DIAYNAlgo(BaseAlgo):
    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None):
        
        num_frames_per_proc = num_frames_per_proc or 128

        # Init base class
        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda,
                         entropy_coef, value_loss_coef, max_grad_norm, recurrence, adam_eps,
                         clip_eps, epochs, batch_size, preprocess_obss, reshape_reward)
        
        # Additional init specific to SACAlgo
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.acmodel = acmodel.to(self.device)
        self.optimizer = Adam(self.acmodel.parameters(), lr=lr, eps=adam_eps)

        # Initialize other components specific to SAC
        self.memory = Memory(self.config["mem_size"], self.config["seed"])
        self.mse_loss = torch.nn.MSELoss()
        self.cross_ent_loss = torch.nn.CrossEntropyLoss()

    def update_parameters(self):
        if len(self.memory) < self.batch_size:
            return None
        else:
            batch = self.memory.sample(self.batch_size)
            states, zs, dones, actions, next_states = self.unpack(batch)
            p_z = torch.from_numpy(self.p_z).to(self.device)

            # Calculating the value target
            reparam_actions, log_probs = self.policy_network.sample_or_likelihood(states)
            q1 = self.q_value_network1(states, reparam_actions)
            q2 = self.q_value_network2(states, reparam_actions)
            q = torch.min(q1, q2)
            target_value = q.detach() - self.config["alpha"] * log_probs.detach()

            value = self.value_network(states)
            value_loss = self.mse_loss(value, target_value)

            logits = self.discriminator(torch.split(next_states, [self.n_states, self.n_skills], dim=-1)[0])
            p_z = p_z.gather(-1, zs)
            logq_z_ns = F.log_softmax(logits, dim=-1)
            rewards = logq_z_ns.gather(-1, zs).detach() - torch.log(p_z + 1e-6)

            # Calculating the Q-Value target
            with torch.no_grad():
                target_q = self.config["reward_scale"] * rewards.float() + \
                           self.config["gamma"] * self.value_target_network(next_states) * (~dones)
            q1 = self.q_value_network1(states, actions)
            q2 = self.q_value_network2(states, actions)
            q1_loss = self.mse_loss(q1, target_q)
            q2_loss = self.mse_loss(q2, target_q)

            policy_loss = (self.config["alpha"] * log_probs - q).mean()
            logits = self.discriminator(torch.split(states, [self.n_states, self.n_skills], dim=-1)[0])
            discriminator_loss = self.cross_ent_loss(logits, zs.squeeze(-1))

            # Zero gradients
            self.optimizer.zero_grad()

            # Backward pass
            policy_loss.backward()
            value_loss.backward()
            q1_loss.backward()
            q2_loss.backward()
            discriminator_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.config["max_grad_norm"])
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.config["max_grad_norm"])
            torch.nn.utils.clip_grad_norm_(self.q_value_network1.parameters(), self.config["max_grad_norm"])
            torch.nn.utils.clip_grad_norm_(self.q_value_network2.parameters(), self.config["max_grad_norm"])
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config["max_grad_norm"])

            # Optimizer step
            self.optimizer.step()

            # Logging
            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            log_entropies.append(log_probs.mean().item())
            log_values.append(value_loss.item())
            log_policy_losses.append(policy_loss.item())
            log_value_losses.append(value_loss.item())
            log_grad_norms.append(sum(p.grad.data.norm(2).item() ** 2 for p in self.policy_network.parameters()) ** 0.5)

            logs = {
                "entropy": np.mean(log_entropies),
                "value": np.mean(log_values),
                "policy_loss": np.mean(log_policy_losses),
                "value_loss": np.mean(log_value_losses),
                "grad_norm": np.mean(log_grad_norms)
            }

            return logs

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)
        zs = torch.cat(batch.z).view(self.batch_size, 1).long().to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)
        actions = torch.cat(batch.action).view(-1, self.config["n_actions"]).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)

        return states, zs, dones, actions, next_states

    def choose_action(self, states):
        states = np.expand_dims(states, axis=0)
        states = torch.from_numpy(states).float().to(self.device)
        action, _ = self.policy_network.sample_or_likelihood(states)
        return action.detach().cpu().numpy()[0]

    def store(self, state, z, done, action, next_state):
        state = torch.from_numpy(state).float().to("cpu")
        z = torch.ByteTensor([z]).to("cpu")
        done = torch.BoolTensor([done]).to("cpu")
        action = torch.Tensor([action]).to("cpu")
        next_state = torch.from_numpy(next_state).float().to("cpu")
        self.memory.add(state, z, done, action, next_state)

    def soft_update_target_network(self, local_network, target_network):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(self.config["tau"] * local_param.data +
                                    (1 - self.config["tau"]) * target_param.data)

    def hard_update_target_network(self):
        self.value_target_network.load_state_dict(self.value_network.state())
        self.value_target_network.eval()

    def get_rng_states(self):
        return torch.get_rng_state(), self.memory.get_rng_state()

    def set_rng_states(self, torch_rng_state, random_rng_state):
        torch.set_rng_state(torch_rng_state.to("cpu"))
        self.memory.set_rng_state(random_rng_state)

    def set_policy_net_to_eval_mode(self):
        self.policy_network.eval()

    def set_policy_net_to_cpu_mode(self):
        self.device = torch.device("cpu")
        self.policy_network.to(self.device)