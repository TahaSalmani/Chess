import torch
from Chess.config.configuration import PrepareTorchBaseModelConfig
from torch import nn
from pathlib import Path


class  TransformerEncoderBlock(nn.Module) :
    def __init__(self, config:PrepareTorchBaseModelConfig):
        self.config = config

        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=self.config.params_D_MODEL , num_heads=self.config.params_NUM_HEADS , batch_first=True)
        self.norm1 = nn.LayerNorm(self.config.params_D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(self.config.params_D_MODEL, self.config.params_FF_DIM) ,
            nn.ReLU() ,
            nn.Linear(self.config.params_FF_DIM , self.config.params_D_MODEL)

        )

        self.norm2 = nn.LayerNorm(self.config.params_D_MODEL)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attn_out , _ = self.attn(x , x , x )
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x+ self.dropout(ffn_out))
        return x

class ChessPolicyNet(nn.Module):
    def __init__(self, config:PrepareTorchBaseModelConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(self.config.params_IN_CHANNELS, self.config.params_D_MODEL)
        self.pos_embd = nn.Parameter(torch.randn(1,64 , self.config.params_D_MODEL))
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(self.config) for _ in range(self.config.params_NUM_LAYERS)
        ])
        self.fc = nn.Linear(self.config.params_D_MODEL, 256)
        self.relu = nn.ReLU()
        self.out = nn.Linear(256 , self.config.params_NUM_MOVES)

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, 64, -1)
        x = self.input_proj(x)
        x = x + self.pos_embd
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.relu(self.fc(x))
        return self.out(x)


class PrepareTorchBaseModel(nn.Module):
    def __init__(self, config:PrepareTorchBaseModelConfig):
        super().__init__()
        self.config = config

    def get_base_model (self) :
        model = ChessPolicyNet(self.config)

        self.save_model(path = self.config.model_path , model = model)
        return model

    @staticmethod
    def save_model(path : Path , model : nn.Module) :
        torch.save(model.state_dict(), path)





