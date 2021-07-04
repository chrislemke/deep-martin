import torch
import torch.nn as nn
import math
from torch.autograd import Variable


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, device: torch.device, max_seq_len: int = 200, dropout: float = 0.1):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.device = device
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        x = x + pe.to(self.device)
        return self.dropout(x)
