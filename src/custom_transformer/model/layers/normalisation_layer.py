import torch
import torch.nn as nn


class Norm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super(Norm, self).__init__()
        self.size = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones([self.size]))
        self.bias = nn.Parameter(torch.zeros(self.size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.alpha * (x - x.mean(dim=-1, keepdims=True)) / (x.std(dim=-1, keepdims=True) + self.eps) + self.bias
        return norm
