from typing import Tuple
import torch
import torch.nn as nn

from src.custom_transformer.model.encoder import Encoder
from src.custom_transformer.model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, src_vocab: int, trg_vocab: int, d_model: int, n: int, heads: int, dropout: float,
                 device: torch.device):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, n, heads, dropout, device)
        self.decoder = Decoder(trg_vocab, d_model, n, heads, dropout, device)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src: torch.Tensor, trg: torch.Tensor, src_mask: torch.Tensor,
                trg_masks: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_masks)
        output = self.out(d_output)
        return output


def get_model(args, src_vocab: int, trg_vocab: int, device: torch.device):
    assert args.d_model % args.heads == 0
    assert args.dropout < 1

    model = Transformer(src_vocab, trg_vocab, args.d_model, args.n_layers, args.heads, args.dropout, device)

    if args.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{args.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    return model.to(device=device)
