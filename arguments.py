import os
import random
import argparse
import torch
import numpy as np

from utils import cleanup_log_dir

def get_args():
    parser = argparse.ArgumentParser(description='D3PG_Batch')
    parser.add_argument('--task-id', type=str, default='AntBulletEnv-v0',
                        help='environment name (default: HalfCheetahBulletEnv-v0)')
    parser.add_argument('--run-id', type=str, default='test',
                        help="name of the run")
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=12,
                        help='number of parallel processes (default: 12)')
    parser.add_argument("--disable-cuda", default=False, help='Disable CUDA')

    # Training config
    parser.add_argument('--num-env-steps', type=int, default=4e5,
                        help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--actor-lr', type=float, default=1e-3,
                        help='actor learning rate (default: 1e-3)')
    parser.add_argument('--critic-lr', type=float, default=1e-3,
                        help='critic learning rate (default: 1e-3)')
    parser.add_argument('--use-linear-lr-decay', type=bool, default=True,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--replay-size', type=int, default=int(1e5),
                        help='replay buffer size (default: 1e5)')
    parser.add_argument('--replay-initial', type=int, default=int(1e3),
                        help='initial sample size (default: 1e3)')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='batch size (default: 20)')

    # DDPG config
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='reward discount coefficient (default: 0.99)')
    parser.add_argument('--reward-steps', type=int, default=5,
                        help='number of steps of the Bellman equation (default: 5)')

    # Log config
    parser.add_argument('--log-interval', type=int, default=1000,
                        help='log interval, one log per n updates (default: 1000)')
    parser.add_argument('--log-dir', type=str, default='log/',
                        help='directory to save agent logs (default: log/)')
    parser.add_argument('--monitor-dir', type=str, default='monitor_log/',
                        help='directory to save monitor logs (default: monitor_log/)')
    parser.add_argument('--result-dir', type=str, default='results/',
                        help='directory to save plot results (default: results/)')

    # Evaluate performance
    parser.add_argument('--test-iters', type=int, default=int(1e4),
                        help='test iterations (default: 1000)')
    parser.add_argument('--video-width', type=int, default=720,
                        help='video resolution (default: 720)')
    parser.add_argument('--video-height', type=int, default=720,
                        help='video resolution (default: 720)')

    args = parser.parse_args()

    # Create directories
    args.save_path = os.path.join("saves", args.task_id, args.run_id)
    args.monitor_dir = os.path.join(args.monitor_dir, args.task_id, args.run_id)
    args.result_dir = os.path.join(args.result_dir, args.task_id)

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.monitor_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    cleanup_log_dir(args.monitor_dir)

    # Setup device and random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
    else:
        args.device = torch.device('cpu')
    torch.set_num_threads(1)

    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

    return args


if __name__ == '__main__':
    args = get_args()

    print(args.save_path)
