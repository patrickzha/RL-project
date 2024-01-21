import sys, os
sys.path.insert(0, os.path.abspath(".."))
os.environ["MUJOCO_GL"] = "glfw"  # for mujoco rendering
import time
from pathlib import Path

import torch
import gym
import hydra
import wandb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pg_ac import PG
from ddpg import DDPG
from common import helper as h
from common import logger as logger
from make_env import create_env
def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()
# Policy training function
def train(agent, env, max_episode_steps=1000):
    # Run actual training
    reward_sum, timesteps, done, episode_timesteps = 0, 0, False, 0
    # Reset the environment and observe the initial state
    obs = env.reset()
    while not done:
        episode_timesteps += 1
        # Sample action from policy
        action, (act_logprob) = agent.get_action(obs)
        # Perform the action on the environment, get new state and reward
        next_obs, reward, done, _ = env.step(to_numpy(action))
        # Store action's outcome (so that the agent can improve its policy)
        if isinstance(agent, PG):
            done_bool = done
            agent.record(obs, act_logprob, reward, done_bool, next_obs)
        elif isinstance(agent, DDPG):
            # ignore the time truncated terminal signal
            done_bool = float(done) if episode_timesteps < max_episode_steps else 0
            agent.record(obs, action, next_obs, reward, done_bool)
        else:
            raise ValueError
        # Store total episode reward
        reward_sum += reward
        timesteps += 1
        # update observation
        obs = next_obs.copy()
        if isinstance(agent, DDPG) and agent.buffer_ready:
            info = agent.update()
        else:
            info = dict()
    if isinstance(agent, PG):
        info = agent.update()
    # Return stats of training
    info.update({'timesteps': timesteps,
                'ep_reward': reward_sum,})
    return info
# Function to test a trained policy
@torch.no_grad()
def test(agent, env, num_episode=50):
    total_test_reward = 0
    for ep in range(num_episode):
        obs, done = env.reset(), False
        test_reward = 0
        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(obs, evaluation=True)
            obs, reward, done, info = env.step(to_numpy(action))
            test_reward += reward
        total_test_reward += test_reward
        # print("Test ep_reward:", test_reward)
    print("Average test reward:", total_test_reward/num_episode)
    return total_test_reward / num_episode
# The main function
@hydra.main(config_path='cfg', config_name='ex6_cfg')
def main(cfg):
    # sed seed
    h.set_seed(cfg.seed)
    cfg.run_id = int(time.time())
    # create folders if needed
    work_dir = Path().cwd()/'results'/f'{cfg.env_name}'
    if cfg.save_model:
        h.make_dir(work_dir / "model")
        h.make_dir(work_dir / "model" / str(cfg.run_id))
    if cfg.save_logging:
        h.make_dir(work_dir / "logging")
        L = logger.Logger()  # create a simple logger to record stats
    # Model filename
    if cfg.model_path == 'default':
        cfg.model_path = work_dir / 'model' / str(cfg.run_id)
    # use wandb to store stats; we aren't currently logging anything into wandb during testing
    if cfg.use_wandb and not cfg.testing:
        wandb.init(project="rl_aalto",
                   name=f'{cfg.exp_name}-{cfg.env_name}-{str(cfg.seed)}-{str(cfg.run_id)}',
                   group=f'{cfg.exp_name}-{cfg.env_name}',
                   config=cfg)
    # create a env
    # env = gym.make(cfg.env_name)
    env = create_env('bipedalwalker_easy', seed=cfg.seed)
    if cfg.save_video:
        # During testing, save every episode
        if cfg.testing:
            ep_trigger = 1
            video_path = work_dir/'video'/'test'
        # During training, save every 50th episode
        else:
            ep_trigger = 50
            video_path = work_dir/'video'/'train'
        env = gym.wrappers.RecordVideo(env, video_path,
                                        episode_trigger=lambda x: x % ep_trigger == 0,
                                        name_prefix=cfg.exp_name)  # save video every 50 episode
    state_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    # init agent
    if cfg.agent_name == "pg_ac":
        agent = PG(
            state_dim=state_shape[0],
            action_dim=action_dim,
            lr=cfg.lr,
            gamma=cfg.gamma,
            ent_coeff=cfg.ent_coeff,
            normalize=cfg.normalize
        )
    else:  # ddpg
        agent = DDPG(
            state_shape=state_shape,
            action_dim=action_dim,
            max_action=max_action,
            actor_lr=cfg.actor_lr,
            critic_lr=cfg.critic_lr,
            gamma=cfg.gamma,
            tau=cfg.tau,
            batch_size=cfg.batch_size,
            use_ou=cfg.use_ou,
            normalize=cfg.normalize,
            buffer_size=cfg.buffer_size
        )
    if not cfg.testing:  # training
        total_interactions = 0
        for ep in range(cfg.train_episodes + 1):
            # collect data and update the policy
            train_info = train(agent, env)
            total_interactions += train_info['timesteps']
            if cfg.use_wandb:
                wandb.log(train_info)
            if cfg.save_logging:
                L.log(**train_info)
            if ep % 100 == 0:
                if cfg.save_model:
                    agent.save(cfg.model_path)
                print({"ep": ep, "total interactions": total_interactions, **train_info})
                avg_test_reward = test(agent, env, num_episode=50)
                if avg_test_reward > 250:
                    print("Reached 250 reward!!")
                    break
            #if total_interactions > cfg.num_interactions:
            #    break
    else:  # testing
        if cfg.model_path == 'default':
            raise NameError("Specify full model path!")
        model_path = work_dir / 'model' / str(cfg.model_path)
        print("Loading model from", model_path, "...")
        # load model
        agent.load(model_path)
        print('Testing ...')
        test(agent, env, num_episode=50)

# Entry point of the script
if __name__ == "__main__":
    main()
