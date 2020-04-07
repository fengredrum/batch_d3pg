import torch
import torch.nn as nn
import numpy as np

from torch import jit
from utils import init


class DDPGActor(jit.ScriptModule):
    __constants__ = ['epsilon']

    def __init__(self, obs_size, act_size, hidden_size_1=128, hidden_size_2=128, epsilon=0.3):
        super(DDPGActor, self).__init__()

        self.epsilon = epsilon
        self.act_size = act_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.fc = nn.Sequential(
            init_(nn.Linear(obs_size, hidden_size_1)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size_1, hidden_size_2)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size_2, act_size)),
            nn.Tanh(),
        )

    @jit.script_method
    def forward(self, x):
        action = self.fc(x)
        # if not deterministic:
        action = action + self.epsilon * torch.randn(action.size(), device=action.device)
        # action = action.clamp(-1., 1.)
        return action

    def sync_param(self, model):
        self.load_state_dict(model.state_dict())

    def alpha_sync_param(self, model, alpha=1e-3):
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        model_state = model.state_dict()
        target_net_state = self.state_dict()
        for key, value in model_state.items():
            target_net_state[key] = (1. - alpha) * target_net_state[key] + alpha * value
        self.load_state_dict(target_net_state)


class DDPGCritic(jit.ScriptModule):
    def __init__(self, obs_size, act_size, hidden_size_1=128, hidden_size_2=128):
        super(DDPGCritic, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.obs_fc = nn.Sequential(
            init_(nn.Linear(obs_size, hidden_size_1)),
            nn.ReLU(),
        )

        self.out_fc = nn.Sequential(
            init_(nn.Linear(hidden_size_1 + act_size, hidden_size_2)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size_2, 1)),
        )

    @jit.script_method
    def forward(self, x, a):
        obs_out = self.obs_fc(x)
        return self.out_fc(torch.cat([obs_out, a], dim=1))

    def sync_param(self, model):
        self.load_state_dict(model.state_dict())

    def alpha_sync_param(self, model, alpha=1e-3):
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        model_state = model.state_dict()
        target_net_state = self.state_dict()
        for key, value in model_state.items():
            target_net_state[key] = (1. - alpha) * target_net_state[key] + alpha * value
        self.load_state_dict(target_net_state)
