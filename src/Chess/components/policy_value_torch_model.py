import os

import torch
from torch import nn

from Chess.components.torch_base_model import TransformerEncoderBlock
from Chess.entity.config_entity import PrepareTorchBaseModelConfig


class ValuePolicyNet(nn.Module):
    def __init__(self , config : PrepareTorchBaseModelConfig):
        super().__init__()
        self.config = config
        self.input = nn.Linear(config.params_IN_CHANNELS  , config.params_D_MODEL)
        self.pos_embed = nn.Parameter(torch.randn(1,64 , config.params_D_MODEL))
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(config) for _ in range(config.params_NUM_LAYERS)
        ])
        self.policy_fc = nn.Linear(config.params_D_MODEL, 256)
        self.policy_out = nn.Linear(256, config.params_NUM_MOVES)

        self.value_fc = nn.Linear(config.params_D_MODEL , 256)
        self.value_out = nn.Linear(256, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        b = x.shape[0] ##number of plays --> 6000
        x = x.view(b, 64, -1) ### (number of plays , 64 , all numbers in  tensor / number of plays * 64) ---> [number of plays , 64 , 12]
        x = self.input(x) ## increase property 12 --->  128
        x = x + self.pos_embed ## [6000 , 64 , 128] + [1 , 64 , 128] ---> unique point for each sequence


        for block in self.blocks:
            x = block(x)

        pooled = x.mean(dim=1)


        policy_logits = self.policy_out(self.relu(self.policy_fc(pooled)))
        value = torch.tanh(self.value_out(self.relu(self.value_fc(pooled))))

        return policy_logits, value

    def load_backbone_from_policy(value_model , policy_model_path, device , config : PrepareTorchBaseModelConfig):


        policy_model_path = os.path.join("artifacts" , "prepare_torch_callbacks" , "best_model.pth")
        policy_weights = torch.load(policy_model_path, map_location=device)
        value_weights = value_model.state_dict()

        for name in value_weights:
            if name in policy_weights:
                value_weights[name] = policy_weights[name]

        value_model.load_state_dict(value_weights)
        return value_model
