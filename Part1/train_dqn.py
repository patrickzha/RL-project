import gym
import numpy as np
from matplotlib import pyplot as plt
from dqn_agent import DQNAgent
from rbf_agent import RBFAgent
import torch
import tqdm
import time
import hydra
from pathlib import Path
import wandb
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from common import helper as h
from common import logger as logger
from common.buffer import ReplayBuffer
import yaml

GYM_TASKS = {
    'MountainCarContinuous-v0',
    'LunarLander-v2',
    'BipedalWalker-v3',
    'Hopper-v2',
    'Hopper-v4'
}


def create_env(config_file_name, seed):
    config = yaml.load(open(f'./configs/{config_file_name}.yaml', 'r'),  Loader=yaml.Loader)

    if config['env_name'] in GYM_TASKS:
        env_kwargs = config['env_parameters']
        if env_kwargs is None:
            env_kwargs = dict()
        env = gym.make(config['env_name'], **env_kwargs)
        env.reset(seed=seed)

    else:
        raise NameError("Wrong environment name was provided in the config file! Please report this to @Nikita Kostin.")

    return env

@torch.no_grad()
def test(agent, env, num_episode=50):
    total_test_reward = 0
    for ep in range(num_episode):
        state, done, ep_reward, env_step = env.reset(), False, 0, 0
        rewards = []

        # collecting data and fed into replay buffer
        while not done:
            # Select and perform an action
            action = agent.get_action(state, epsilon=0.0)
            if isinstance(action, np.ndarray): action = action.item()
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            rewards.append(reward)
        total_test_reward += ep_reward 
        info = {'episode': ep, 'ep_reward': ep_reward}
    print(total_test_reward / num_episode)
    return total_test_reward / num_episode

@hydra.main(config_path='cfg', config_name='ex4_cfg')
def main(cfg):
    # set random seed
    h.set_seed(cfg.seed)

    cfg.run_id = int(time.time())
    # create folders if needed
    work_dir = Path().cwd()/'results'/cfg.env_name
    if cfg.save_logging:
        logging_path = work_dir / 'logging'
        h.make_dir(logging_path)
        L = logger.Logger() # create a simple logger to record stats
    if cfg.save_model:
        model_path = work_dir / 'model'/ str(cfg.run_id)
        h.make_dir(model_path)
    
    # use wandb to store stats
    if cfg.use_wandb:
        wandb.init(project="rl_aalto",
                    name=f'{cfg.exp_name}-{cfg.agent_name}-{cfg.env_name}-{str(cfg.seed)}-{cfg.run_id}',
                    group=f'{cfg.exp_name}-{cfg.env_name}',
                    config=cfg)
    
    # create env
    env = create_env(config_file_name="lunarlander_discrete_medium", seed=1)
    if cfg.save_video:
        env = gym.wrappers.RecordVideo(env, work_dir/'video'/'train',
                                        episode_trigger=lambda x: x % 100 == 0,
                                        name_prefix=cfg.exp_name) # save video for every 100 episodes
    # get number of actions and state dimensions
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    # init 
    if cfg.agent_name == "dqn":
        agent = DQNAgent(state_shape, n_actions, batch_size=cfg.batch_size, hidden_dims=cfg.hidden_dims,
                         gamma=cfg.gamma, lr=cfg.lr, grad_clip_norm=1000, tau=cfg.tau)
    elif cfg.agent_name == "rbf":
        agent = RBFAgent(n_actions, gamma=cfg.gamma, batch_size=cfg.batch_size)
    else:
        raise ValueError(f"No {cfg.agent_name} agent implemented")

    #  init buffer
    buffer = ReplayBuffer(state_shape, action_dim=1, max_size=int(cfg.buffer_size))

    for ep in range(cfg.train_episodes):
        state, done, ep_reward, env_step = env.reset(), False, 0, 0
        eps = max(cfg.glie_b/(cfg.glie_b + ep), 0.05)

        # collecting data and fed into replay buffer
        while not done:
            env_step += 1
            if ep < cfg.random_episodes: # in the first #random_episodes, collect random trajectories
                action = env.action_space.sample()
            else:
                # Select and perform an action
                action = agent.get_action(state, eps)
                if isinstance(action, np.ndarray): action = action.item()
                
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward

            # Store the transition in replay buffer
            buffer.add(state, action, next_state, reward, done)

            # Move to the next state
            state = next_state
        
            # Perform one update_per_episode step of the optimization
            if ep >= cfg.random_episodes:
                update_info = agent.update(buffer)
            else: update_info = {}

        info = {'episode': ep, 'epsilon': eps, 'ep_reward': ep_reward}
        info.update(update_info)

        if cfg.use_wandb: wandb.log(info)
        if cfg.save_logging: L.log(**info)
        if (not cfg.silent) and (ep % 100 == 0): 
            print(info)
            avg_test_reward = test(agent, env, num_episode=50)
            if avg_test_reward > 100:
                        print("Reached 100 reward!!")
                        break
    # save model and logging    
    if cfg.save_model:
        agent.save(model_path)
    if cfg.save_logging:
        L.save(logging_path/'logging.pkl')
    
    print('------ Training Finished ------')


if __name__ == '__main__':
    main()