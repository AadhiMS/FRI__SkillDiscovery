import argparse
import time
import datetime
import torch_ac
import tensorboardX
import sys
from SelfMade_Algo import DIAYNAlgo
import utils
from utils import device
from model import ACModel
from logger import Logger
from agent import SACAgent
import numpy as np
from tqdm import tqdm
from torch_ac.utils import ParallelEnv
import gym as gym 
#from gymnasium.wrappers import FlattenObservation
#from minigrid.wrappers import DictObservationSpaceWrapper
from minigrid.wrappers import FlatObsWrapper, ReseedWrapper

# Notes Log: I added the def get_params(): under the parser arguments but not it seems that the code cannot pick up any of the arguments from the command placed in the terminal 



# Parse arguments



def get_params():
# General parameters
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", required=True,
                        help="algorithm to use: a2c | ppo (REQUIRED)")
    parser.add_argument("--env", required=True,
                        help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--model", default=None,
                        help="name of the model (default: {ENV}_{ALGO}_{TIME})")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--procs", type=int, default=16,
                        help="number of processes (default: 16)")
    parser.add_argument("--frames", type=int, default=10**7,
                        help="number of frames of training (default: 1e7)")

    # Parameters for main algorithm
    parser.add_argument("--epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="batch size for PPO (default: 256)")
    parser.add_argument("--frames-per-proc", type=int, default=None,
                        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--optim-alpha", type=float, default=0.99,
                        help="RMSprop optimizer alpha (default: 0.99)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--text", action="store_true", default=False,
                        help="add a GRU to the model to handle text input")
    parser.add_argument("--mem_size", default=int(1e+6), type = int, help = "The memory size (for DIAYN).")
    parser.add_argument("--n_skills", default=5, type=int, help="The number of skills to learn (for DIAYN).")
# Changed n_skills from 50 to 5
    return parser

if __name__ == "__main__":
    
    args = get_params().parse_args()

    args.mem = args.recurrence > 1

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    txt_logger.info(f"Device: {device}\n")

    # Load environments

    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.seed + 10000 * i))
    txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor

    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model

    acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo

    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
    elif args.algo == "DIAYNAlgo":
        """algo = DIAYNAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)"""
        #env = ParallelEnv(envs)
        params = vars(args)
        default_params = {"lr": 3e-4,
                      "batch_size": 256,
                      "max_n_episodes": 5000,
                      "max_episode_len": 1000,
                      "gamma": 0.99,
                      "alpha": 0.1,
                      "tau": 0.005,
                      "n_hiddens": 300,
                      "n_states": 10,
                      "interval": 3,
                      "reward_scale": 1, #Note: Set this custom, need to double checl with mentor
                      #"mem_size": 10000 #Ive set this myself, 11/13/2024
                      }
        # NEW CODE FROM 10/23/24
        #params = get_params()
        test_env =  envs[0]  #gym.make(args["env_name"])
        #breakpoint()
        test_env = FlatObsWrapper(test_env)
        test_env = ReseedWrapper(test_env, seeds=(params["seed"], )  )
        
        n_states = test_env.observation_space.shape[0]
        #breakpoint()
        n_actions = test_env.action_space.n
        #action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]

        params.update({"n_states": n_states,
                "n_actions": n_actions,
                "env_name": params["env"],
                "do_train": True,
                "train_from_scratch": False})
        #endregion
        #breakpoint()
        params = {**default_params, **params}
        #params = {**vars(args), **default_params} #Old code, replaced with line above
        p_z = np.full(params["n_skills"], 1 / params["n_skills"])
        agent = SACAgent(p_z=p_z, **params)
        logger = Logger(agent, **params)

    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))
    
    if (args.algo != "DIAYNAlgo"): 
        if "optimizer_state" in status:
            algo.optimizer.load_state_dict(status["optimizer_state"])
        txt_logger.info("Optimizer loaded\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    if args.algo == "DIAYNAlgo": #Lets make our own run process
        """
        while num_frames < args.frames:
            # Update model parameters
            update_start_time = time.time()
            exps, logs1 = algo.collect_experiences()

            # Log the size of the memory buffer
            txt_logger.info(f"Memory size after collecting experiences: {len(algo.DIAYNmemory)}")

            logs2 = algo.update_parameters() 
            if logs2 is None:
                txt_logger.info(f"Skipping update {update} due to insufficient memory.")
                continue

            logs = {**logs1, **logs2}
            update_end_time = time.time()

            num_frames += logs["num_frames"]
            update += 1

        # Print logs
        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            txt_logger.info(f"Update {update}, FPS {fps}, Duration {duration}s, Return per episode {return_per_episode}") 
    """
        
        def concat_state_latent(s, z_, n):
            
            z_one_hot = np.zeros(n)
            z_one_hot[z_] = 1
            #breakpoint()
            #print(s.shape, z_one_hot.shape)
            return np.concatenate([s, z_one_hot])

        if False: #not params["train_from_scratch"]:
            episode, last_logq_zs, np_rng_state, *env_rng_states, torch_rng_state, random_rng_state = logger.load_weights()
            agent.hard_update_target_network()
            min_episode = episode
            np.random.set_state(np_rng_state)
            env.np_random.set_state(env_rng_states[0])
            env.observation_space.np_random.set_state(env_rng_states[1])
            env.action_space.np_random.set_state(env_rng_states[2])
            agent.set_rng_states(torch_rng_state, random_rng_state)
            print("Keep training from previous run.")

        else:
            min_episode = 0
            last_logq_zs = 0
            np.random.seed(params["seed"])
            #test_env.unwrapped.seed(params["seed"])
            test_env.observation_space.seed(params["seed"])
            test_env.action_space.seed(params["seed"])
            print("Training from scratch.")

        
        
        logger.on()
        for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
            z = np.random.choice(params["n_skills"], p=p_z)
            state = test_env.reset()[0]
            state = concat_state_latent(state, z, params["n_skills"])
            episode_reward = 0
            logq_zses = []
            #breakpoint()
            #max_n_steps = min(params["max_episode_len"], test_env.spec.max_episode_steps)
            max_n_steps = params["max_episode_len"]
            for step in range(1, 1 + max_n_steps):

                action = agent.choose_action(state)
                #next_state, reward, done, _ = test_env.step(action)
                next_state, reward, done, truncated, _ = test_env.step(action)  # Update to handle 5 values
                next_state = concat_state_latent(next_state, z, params["n_skills"])
                agent.store(state, z, done, action, next_state)
                logq_zs = agent.train()
                if logq_zs is None:
                    logq_zses.append(last_logq_zs)
                else:
                    logq_zses.append(logq_zs)
                episode_reward += reward
                state = next_state
                if done:
                    break

            logger.log(episode,
                       episode_reward,
                       z,
                       sum(logq_zses) / len(logq_zses),
                       step,
                       0,
                       0,
                       0,
                       0,
                       0,
                       )

    else: 
        while num_frames < args.frames:
            # Update model parameters
            update_start_time = time.time()
            exps, logs1 = algo.collect_experiences()
            logs2 = algo.update_parameters(exps) 
            logs = {**logs1, **logs2}
            update_end_time = time.time()

            num_frames += logs["num_frames"]
            update += 1

            # Print logs

            if update % args.log_interval == 0:
                fps = logs["num_frames"] / (update_end_time - update_start_time)
                duration = int(time.time() - start_time)
                return_per_episode = utils.synthesize(logs["return_per_episode"])
                rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
                num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

                header = ["update", "frames", "FPS", "duration"]
                data = [update, num_frames, fps, duration]
                header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
                data += rreturn_per_episode.values()
                header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                data += num_frames_per_episode.values()
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                    .format(*data))

                header += ["return_" + key for key in return_per_episode.keys()]
                data += return_per_episode.values()

                if status["num_frames"] == 0:
                    csv_logger.writerow(header)
                csv_logger.writerow(data)
                csv_file.flush()

                for field, value in zip(header, data):
                    tb_writer.add_scalar(field, value, num_frames)

            # Save status

            if args.save_interval > 0 and update % args.save_interval == 0:
                status = {"num_frames": num_frames, "update": update,
                        "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
                if hasattr(preprocess_obss, "vocab"):
                    status["vocab"] = preprocess_obss.vocab.vocab
                utils.save_status(status, model_dir)
                txt_logger.info("Status saved")
