
from ray.rllib.utils.annotations import override
from ray.rllib.core.rl_module.rl_module import RLModule
import numpy as np
import pdb
import torch
from torch import nn
from typing import Any, Union
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.models.torch import torch_distributions
from ray.rllib.core.columns import Columns


class MLPAgent(TorchRLModule, ValueFunctionAPI):

    @override(TorchRLModule)
    def setup(self):

        in_size = len(self.observation_space.keys())
        layers = []
        dense_layers = self.model_config['dense_layers']
        self.distribution_actions_lens = []

        for out_size in dense_layers:
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
            in_size = out_size

        self._fc_net = nn.Sequential(*layers)
        
        out_size_logits = 0
        for action in self.action_space:
            out_size_logits += action.n.item()
            self.distribution_actions_lens.append(action.n.item())

        self._pi_head = nn.Linear(in_size, out_size_logits)
        self._vf_head = nn.Linear(in_size, 1)


    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        obs = self.set_batch_obs(batch['obs'])
        embeddings = self._fc_net(obs)
        logits = self._pi_head(embeddings)

        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.EMBEDDINGS: embeddings
            }
    

    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: dict[str, Any],
        embeddings: Union[Any, None] = None
        ):
        
        if embeddings is None:
            obs = self.set_batch_obs(batch['obs'])
            embeddings = self._fc_net(obs)

        values = self._vf_head(embeddings).squeeze(1)
        return values
    
    def set_batch_obs(self, batch):
        tensor_array = torch.hstack([v for v in batch.values()])
        return tensor_array
    
    def get_inference_action_dist_cls(self):
        return torch_distributions.TorchMultiCategorical.get_partial_dist_cls(
            input_lens=self.distribution_actions_lens
        )
