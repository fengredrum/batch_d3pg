import os
import glob

import torch
import torch.nn as nn

def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def policy_update(returns, obs_batch, action_batch,
                  critic_net, actor_net,
                  critic_optimizer, actor_optimizer,
                  max_grad_norm=None):

    T, B, _ = obs_batch.size()

    # Train critic
    critic_optimizer.zero_grad()
    value = critic_net(obs_batch.view(T * B, -1), action_batch.view(T * B, -1))
    value = value.view(T, B)
    critic_loss = (returns - value).pow(2).mean()
    critic_loss.backward()
    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(critic_net.parameters(), max_grad_norm)
    critic_optimizer.step()

    # Train actor
    actor_optimizer.zero_grad()
    action = actor_net(obs_batch.view(T * B, -1))
    value = critic_net(obs_batch.view(T * B, -1), action)
    actor_loss = torch.mean(-value)
    actor_loss.backward()
    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(actor_net.parameters(), max_grad_norm)
    actor_optimizer.step()

    return dict(critic_loss=critic_loss.item(), actor_loss=actor_loss.item())
