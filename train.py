import os
import numpy as np
import time
import torch

from torch.utils.tensorboard import SummaryWriter
from collections import deque
from tqdm import tqdm

from arguments import get_args
from environment import make_vec_envs
from model import DDPGActor, DDPGCritic
from d3pg import D3PG
from storage import ReplayBuffer
from utils import update_linear_schedule


def main(args):
    # Create summary writer
    writer_path = os.path.join(args.log_dir, args.task_id, args.run_id)
    writer = SummaryWriter(log_dir=writer_path)

    # Create training envs
    envs = make_vec_envs(args.task_id, args.seed, args.num_processes,
                         args.gamma, args.monitor_dir, args.device)
    obs_size = envs.observation_space.shape[0]
    act_size = envs.action_space.shape[0]

    # Create actor-critic and their target net
    actor_net = DDPGActor(obs_size, act_size).to(args.device)
    critic_net = DDPGCritic(obs_size, act_size).to(args.device)
    target_actor_net = DDPGActor(obs_size, act_size).to(args.device)
    target_critic_net = DDPGCritic(obs_size, act_size).to(args.device)
    target_actor_net.eval()
    target_critic_net.eval()
    # Initialize target net weights
    target_actor_net.sync_param(actor_net)
    target_critic_net.sync_param(critic_net)

    # Create agent
    agent = D3PG(
        actor_net=actor_net,
        critic_net=critic_net,
        target_actor_net=target_actor_net,
        target_critic_net=target_critic_net,
        num_processes=args.num_processes,
        reward_steps=args.reward_steps,
        batch_size=args.batch_size,
        device=args.device,
        gamma=args.gamma,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
    )

    # Create replay buffer
    buffer = ReplayBuffer(
        buffer_size=args.replay_size,
        replay_initial=args.replay_initial,
        num_processes=args.num_processes,
        obs_size=obs_size,
        act_size=act_size,
    )
    buffer.to(args.device)

    obs = envs.reset()
    buffer.obs[0].copy_(obs)

    episode_rewards = deque(maxlen=10)
    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_processes
    for j in tqdm(range(num_updates)):

        if args.use_linear_lr_decay:
            update_linear_schedule(agent.actor_optimizer, j, num_updates, args.actor_lr)
            update_linear_schedule(agent.critic_optimizer, j, num_updates, args.critic_lr)

        with torch.no_grad():
            # Sample actions
            action = actor_net(obs)

            # Get trajectories from envs
            obs, reward, done, infos = envs.step(action)
            mask = torch.tensor(
                [0.0 if done_ else 1.0 for done_ in done], dtype=torch.float)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            buffer.insert(obs, action, reward, mask)

        if j < args.replay_initial:
            continue

        # Train with batch
        if j % args.reward_steps == 0:
            batch = buffer.get_batch(args.batch_size, args.reward_steps)
            agent_output = agent.update(batch)

            # Log training process
            if j % args.log_interval == 0 and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * args.num_processes
                end = time.time()
                speed = int(total_num_steps / (end - start))
                print(
                    "Updates {}, num timesteps {}, FPS {} \n"
                    "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, "
                    "min/max reward {:.1f}/{:.1f}\n"
                        .format(j, total_num_steps,
                                speed,
                                len(episode_rewards), np.mean(episode_rewards),
                                np.median(episode_rewards), np.min(episode_rewards),
                                np.max(episode_rewards))
                )
                writer.add_scalar('reward_mean', np.mean(episode_rewards), total_num_steps)
                writer.add_scalar('speed', speed, total_num_steps)
                for key in agent_output.keys():
                    writer.add_scalar(key, agent_output[key], total_num_steps)

    writer.close()
    envs.close()


if __name__ == '__main__':
    args = get_args()

    main(args)
