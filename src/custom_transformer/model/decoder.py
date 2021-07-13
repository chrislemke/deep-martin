from typing import Tuple
import torch
import torch.nn as nn

from src.custom_transformer.model.layers.embedding_layer import Embedding
from src.custom_transformer.model.layers.decoder_layer import DecoderLayer
from src.custom_transformer.model.layers.positional_encoder import PositionalEncoder
import src.custom_transformer.model.transformer_model_utils as tu


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n: int, heads: int, dropout: float, max_length: int):
        super(Decoder, self).__init__()
        self.n = n
        self.embed = Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = tu.get_clones(DecoderLayer(d_model, heads, max_length, dropout), n)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, trg: torch.Tensor, e_outputs: torch.Tensor, src_mask: torch.Tensor,
                trg_masks: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.n):
            x = self.layers[i](x, e_outputs, src_mask, trg_masks)
        return self.norm(x)
