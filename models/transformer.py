import os
import numpy as np
import copy
from typing import Optional, Any, Union, Callable

import torch
import torch.nn as nn
from torch import Tensor

import matplotlib.pyplot as plt


class CustomizedTransformerEncoder(nn.TransformerEncoder):
    def __init__(
            self,
            encoder_layer: "CustomizedTransformerEncoderLayer",
            num_layers: int
    ) -> None:
        super().__init__(encoder_layer, num_layers)
        self.layers = nn.ModuleList([
            copy.deepcopy(CustomizedTransformerEncoderLayer(
                d_model=encoder_layer.d_model,
                nhead=encoder_layer.nhead,
                dim_feedforward=encoder_layer.dim_feedforward,
                dropout=encoder_layer.dropout_ratio,
                layer_index=idx
            ))
            for idx in range(num_layers)
        ])

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:
        return super().forward(src, mask, src_key_padding_mask, is_causal)


class CustomizedTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 layer_index: int = 2) -> None:
        super().__init__(d_model, nhead, dropout=dropout)
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_ratio = dropout

        self.layer_index = layer_index
        self.return_attn = False
        self.attn_weights = []

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:
        output = super().forward(src, src_mask, src_key_padding_mask, is_causal)
        return output

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x, attn_weights = self.self_attn(x, x, x,
                                         attn_mask=attn_mask,
                                         key_padding_mask=key_padding_mask,
                                         need_weights=True, is_causal=is_causal)

        if self.return_attn:
            self._save_att(attn_weights)
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        return super()._ff_block(x)

    def _start_att(self):
        self.return_attn = True

    def _save_att(self, attn_weights):
        self.attn_weights.append(np.mean(attn_weights.detach().cpu().numpy(), axis=0))

    def _att2heatmap(self):
        avg_attention = np.mean(self.attn_weights, axis=0)
        x = np.arange(60)
        y = np.arange(60)
        X, Y = np.meshgrid(x, y)

        plt.figure(figsize=(10, 10))
        plt.pcolormesh(X, Y, avg_attention, cmap='viridis', shading='auto')
        plt.colorbar(label='Attention Value')
        plt.title("Attention Heatmap")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.savefig("attention score", bbox_inches='tight', dpi=300)
        plt.close()


class CustomizedGQATransformerEncoder(nn.Module):
    def __init__(
            self,
            encoder_layer: "CustomizedGQATransformerEncoderLayer",
            num_layers: int
    ) -> None:
        super().__init__()
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layers = nn.ModuleList([
            copy.deepcopy(CustomizedGQATransformerEncoderLayer(
                d_model=encoder_layer.d_model,
                nhead=encoder_layer.nhead,
                dim_feedforward=encoder_layer.dim_feedforward,
                dropout=encoder_layer.dropout_ratio,
                layer_index=idx
            ))
            for idx in range(num_layers)
        ])

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        for layer in self.layers:
            x = layer(x, is_causal=is_causal)
        return x


class CustomizedGQATransformerEncoderLayer(GQATransformerEncoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 layer_index: int = 2) -> None:
        super().__init__(d_model, nhead, kv_heads=4, dim_feedforward=dim_feedforward, dropout=dropout)
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_ratio = dropout

        self.layer_index = layer_index
        self.return_attn = False
        self.attn_weights = []

    def forward(self, src: Tensor, is_causal: bool = False) -> Tensor:
        return super().forward(src)




