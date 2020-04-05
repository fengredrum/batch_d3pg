import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description='D3PG_Batch')
    parser.add_argument('--env_id', type=str, default='HalfCheetahBulletEnv-v0',
                        help='environment name (default: HalfCheetahBulletEnv-v0)')
    parser.add_argument('--run_id', type=str, default='test',
                        help="name of the run")
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num_processes', type=int, default=12,
                        help='number of parallel processes (default: 12)')
    parser.add_argument("--disable_cuda", default=False, help='Disable CUDA')

    # Training config
    parser.add_argument('--num-env-steps', type=int, default=10e6,
                        help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--actor-lr', type=float, default=3e-4,
                        help='actor learning rate (default: 3e-4)')
    parser.add_argument('--critic-lr', type=float, default=1e-3,
                        help='critic learning rate (default: 1e-3)')
    parser.add_argument('--use-linear-lr-decay', type=bool, default=True,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--replay_size', type=int, default=int(1e5),
                        help='replay buffer size (default: 1e5)')
    parser.add_argument('--replay_initial', type=int, default=int(1e3),
                        help='initial sample size (default: 1e3)')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='batch size (default: 20)')

    # DDPG config
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='reward discount coefficient (default: 0.99)')
    parser.add_argument('--reward_steps', type=int, default=5,
                        help='number of steps of the Bellman equation (default: 5)')

    # Log config
    parser.add_argument('--log-interval', type=int, default=1000,
                        help='log interval, one log per n updates (default: 1000)')
    parser.add_argument('--log-dir', type=str, default='log/',
                        help='directory to save agent logs (default: log/)')

    # Evaluate performance
    parser.add_argument('--test_iters', type=int, default=int(1e4),
                        help='test iterations (default: 1000)')
    parser.add_argument('--video_width', type=int, default=720,
                        help='video resolution (default: 720)')
    parser.add_argument('--video_height', type=int, default=720,
                        help='video resolution (default: 720)')

    args = parser.parse_args()

    args.save_path = os.path.join("saves", "d4pg-" + args.env_id + "-" + args.run_id)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)


    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

    return args


if __name__ == '__main__':
    args = get_args()

    print(args.save_path)
