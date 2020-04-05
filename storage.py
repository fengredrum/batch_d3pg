import torch
import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, replay_initial, num_processes, obs_size, act_size):
        self.obs = torch.zeros(buffer_size, num_processes, obs_size)
        self.actions = torch.zeros(buffer_size, num_processes, act_size)
        self.rewards = torch.zeros(buffer_size, num_processes)
        self.masks = torch.ones(buffer_size, num_processes)

        self.buffer_size = buffer_size
        self.replay_initial = replay_initial
        self.step = 0
        self.is_full = False

    def to(self, device):
        self.obs = self.obs.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.masks = self.masks.to(device)

    def insert(self, observation, action, reward, mask):
        self.obs[(self.step + 1) % self.buffer_size].copy_(observation)
        self.actions[self.step].copy_(action)
        self.rewards[self.step].copy_(reward)
        self.masks[(self.step + 1) % self.buffer_size].copy_(mask)

        self.step = (self.step + 1) % self.buffer_size
        if not self.is_full and self.step % self.buffer_size == 0:
            self.is_full = True

    def get_batch(self, batch_size, num_steps):
        if not self.is_full:
            if self.step < self.replay_initial:
                raise ValueError('Not enough data!')
            else:
                # Warning: replace=False makes random.choice O(n)
                keys = np.random.choice(self.step - num_steps, batch_size, replace=True)
        else:
            keys = np.random.choice(self.buffer_size - num_steps, batch_size, replace=True)
            keys += self.step

        obs_batch, action_batch, reward_batch, mask_batch = [], [], [], []
        for key in keys:
            index = [(i + 1) % self.buffer_size for i in range(key, key + num_steps + 1)]
            obs_batch.append(self.obs[index])
            mask_batch.append(self.masks[index[1:]])
            action_batch.append(self.actions[index[:-1]])
            reward_batch.append(self.rewards[index[:-1]])

        obs_batch = torch.cat(obs_batch, dim=1)
        action_batch = torch.cat(action_batch, dim=1)
        reward_batch = torch.cat(reward_batch, dim=1)
        mask_batch = torch.cat(mask_batch, dim=1)

        return dict(obs_batch=obs_batch,
                    action_batch=action_batch,
                    reward_batch=reward_batch,
                    mask_batch=mask_batch,
                    )
