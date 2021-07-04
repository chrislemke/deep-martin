import torch.nn as nn
import torch

from src.custom_transformer.model.layers.normalisation_layer import Norm
from src.custom_transformer.model.layers.feed_forward_layer import FeedForward
from src.custom_transformer.model.layers.multi_head_attention_layer import MultiHeadAttentionLayer


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x2 = self.norm_1(x)
        attention = self.attn(x2, x2, x2, mask)
        x = x + self.dropout_1(attention[0])
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
