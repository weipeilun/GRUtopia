from typing import List
import numpy as np
import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import ActorCritic
from grutopia.core.util.rsl_rl import pickle
from huggingface_hub import PyTorchModelHubMixin


class SimplePolicy(nn.Module, PyTorchModelHubMixin):
    """simple policy for h1 locomotion."""

    def __init__(self, path: str = None) -> None:
        super().__init__()

        hidden_dims = [1024, 512, 256]
        activation = torch.nn.ELU()

        num_obs = 350
        self.env_actions = 19

        actor_layers = []
        actor_layers.append(nn.Linear(num_obs, hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(hidden_dims)):
            if layer_index == len(hidden_dims) - 1:
                actor_layers.append(nn.Linear(hidden_dims[layer_index], self.env_actions))
            else:
                actor_layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        if path is not None:
            self.load(path=path)

    def load(self, path: str, load_optimizer=False):
        loaded_dict = torch.load(path, pickle_module=pickle)
        self.actor.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.actor.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.actor.to(device)
        return self.actor

    def train(self, device=None):
        self.actor.train()
        if device is not None:
            self.actor.to(device)
        return self.actor

    def forward(self, batch, target=None):
        action = self.actor(batch)

        if target is not None:
            loss = torch.nn.MSELoss(reduction='mean')(action, target)
            return {'action': action, 'loss': loss}
        else:
            return {'action': action}