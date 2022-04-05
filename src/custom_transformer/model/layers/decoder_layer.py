import torch
import torch.nn as nn

from src.custom_transformer.model.layers.feed_forward_layer import FeedForward
from src.custom_transformer.model.layers.multi_head_attention_layer import (
    MultiHeadAttentionLayer,
)
from src.custom_transformer.model.layers.normalisation_layer import Norm


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttentionLayer(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttentionLayer(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        e_outputs: torch.Tensor,
        src_mask: torch.Tensor,
        trg_mask: torch.Tensor,
    ) -> torch.Tensor:
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x
