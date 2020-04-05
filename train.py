import random
import numpy as np
import time
import torch

from torch.utils.tensorboard import SummaryWriter
from collections import deque
from tqdm import tqdm

from arguments import get_args
from environment import make_vec_envs
from model import DDPGActor, DDPGCritic
from d3pg import D4PG
from storage import ReplayBuffer
from utils import update_linear_schedule


def main(args):
    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.disable_cuda:
        device = torch.device('cuda:0')
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')
    torch.set_num_threads(1)

    # Create summary writer
    writer = SummaryWriter(comment=args.env_id + '-' + args.run_id)
    # Create environment for training
    envs = make_vec_envs(args.env_id, args.seed, args.num_processes, args.gamma, device)
    # Create actor-critic and their target net
    actor_net = DDPGActor(envs.observation_space.shape[0], envs.action_space.shape[0]).to(device)
    critic_net = DDPGCritic(envs.observation_space.shape[0], envs.action_space.shape[0]).to(device)
    target_actor_net = DDPGActor(envs.observation_space.shape[0], envs.action_space.shape[0]).to(device)
    target_critic_net = DDPGCritic(envs.observation_space.shape[0], envs.action_space.shape[0]).to(device)
    target_actor_net.eval()
    target_critic_net.eval()
    # Initialize target net weights
    target_actor_net.sync_param(actor_net)
    target_critic_net.sync_param(critic_net)

    # Setup agent
    agent = D4PG(
        actor_net=actor_net,
        critic_net=critic_net,
        target_actor_net=target_actor_net,
        target_critic_net=target_critic_net,
        num_processes=args.num_processes,
        reward_steps=args.reward_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        max_grad_norm=args.max_grad_norm,
    )

    # Create replay buffer
    replay_buffer = ReplayBuffer(
        buffer_size=args.replay_size,
        replay_initial=args.replay_initial,
        num_processes=args.num_processes,
        obs_size=envs.observation_space.shape[0],
        act_size=envs.action_space.shape[0],
    )
    replay_buffer.to(device)

    obs = envs.reset()
    replay_buffer.obs[0].copy_(obs)

    episode_rewards = deque(maxlen=10)
    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_processes
    for j in tqdm(range(num_updates)):

        if args.use_linear_lr_decay:
            update_linear_schedule(agent.actor_optimizer, j, num_updates, args.actor_lr)
            update_linear_schedule(agent.critic_optimizer, j, num_updates, args.critic_lr)

        with torch.no_grad():
            action = actor_net(obs)

        obs, reward, done, infos = envs.step(action)
        mask = torch.tensor(
            [0.0 if done_ else 1.0 for done_ in done], dtype=torch.float)
        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])

        replay_buffer.insert(obs, action, reward, mask)

        if j < args.replay_initial:
            continue

        # Train with batch
        if j % args.reward_steps == 0:
            batch = replay_buffer.get_batch(args.batch_size, args.reward_steps)
            agent_output = agent.update(batch)

        # Log training process
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes
            end = time.time()
            speed = int(total_num_steps / (end - start))
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            speed,
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards))
            )
            writer.add_scalar('mean_reward', np.mean(episode_rewards), total_num_steps)
            writer.add_scalar('speed', speed, total_num_steps)
            writer.add_scalar('critic_loss', agent_output['critic_loss'], total_num_steps)
            writer.add_scalar('actor_loss', agent_output['actor_loss'], total_num_steps)

    writer.close()
    envs.close()


if __name__ == '__main__':
    args = get_args()

    main(args)
