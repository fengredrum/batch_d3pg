import torch
import torch.nn as nn
import torch.optim as optim


class D3PG(object):
    def __init__(self, actor_net, critic_net,
                 target_actor_net, target_critic_net,
                 num_processes, reward_steps, batch_size, device,
                 gamma=0.99, actor_lr=1e-3, critic_lr=1e-3, max_grad_norm=None):
        self.gamma = gamma
        self.T = reward_steps
        self.B = batch_size * num_processes
        self.max_grad_norm = max_grad_norm
        self.targets = torch.zeros(self.T + 1, self.B, device=device)

        self.actor_net = actor_net
        self.critic_net = critic_net
        self.target_actor_net = target_actor_net
        self.target_critic_net = target_critic_net

        self.actor_optimizer = optim.Adam(actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(critic_net.parameters(), lr=critic_lr)

    def update(self, rollouts):
        obs_batch = rollouts['obs_batch']
        action_batch = rollouts['action_batch']
        reward_batch = rollouts['reward_batch']
        mask_batch = rollouts['mask_batch']

        # Train critic
        self.critic_optimizer.zero_grad()
        with torch.no_grad():
            last_action = self.target_actor_net(obs_batch[-1])
            last_value = self.target_critic_net(obs_batch[-1], last_action)

            reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-6)
            self.targets[-1] = last_value.squeeze(-1)
            for step in reversed(range(self.T)):
                self.targets[step] = reward_batch[step] + \
                                     self.gamma * self.targets[step + 1] * mask_batch[step]

        value = self.critic_net(obs_batch[:-1].view(self.T * self.B, -1), action_batch.view(self.T * self.B, -1))
        value = value.view(self.T, self.B)
        critic_loss = (self.targets[:-1] - value).pow(2).mean()
        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # Train actor
        self.actor_optimizer.zero_grad()
        action = self.actor_net(obs_batch[:-1].view(self.T * self.B, -1))
        value = self.critic_net(obs_batch[:-1].view(self.T * self.B, -1), action)
        actor_loss = torch.mean(-value)
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # Update target nets
        self.target_critic_net.alpha_sync_param(self.critic_net)
        self.target_actor_net.alpha_sync_param(self.actor_net)

        return dict(critic_loss=critic_loss.item(), actor_loss=actor_loss.item())
