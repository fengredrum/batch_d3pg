import torch

from torch import jit


class ComputeReturns(jit.ScriptModule):
    __constants__ = ['gamma', 'T', 'B']

    def __init__(self, target_actor_net, target_critic_net,
                 num_processes, reward_steps, batch_size, device,
                 gamma=0.99):
        super(ComputeReturns, self).__init__()
        self.gamma = gamma
        self.T = reward_steps
        self.B = batch_size * num_processes
        self.register_buffer("targets", torch.zeros(self.T + 1, self.B, device=device))

        self.target_actor_net = target_actor_net
        self.target_critic_net = target_critic_net

    @jit.script_method
    def forward(self, obs, reward_batch, mask_batch):
        last_action = self.target_actor_net(obs)
        last_value = self.target_critic_net(obs, last_action)

        reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-6)
        self.targets[-1] = last_value.squeeze(-1)
        idx = self.T - 1
        for i in range(self.T):
            self.targets[idx - i] = reward_batch[idx - i] + \
                                    self.gamma * self.targets[idx - i + 1] * mask_batch[idx - i]

        return self.targets[:-1].detach()
