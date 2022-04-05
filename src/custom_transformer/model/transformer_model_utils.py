import copy

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


def attention_mask(size: int):
    att_mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    att_mask = Variable(torch.from_numpy(att_mask) == 0)
    return att_mask


def create_masks(src: torch.Tensor, trg: torch.Tensor):
    src_mask = (src != 0).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != 0).unsqueeze(-2)
        size = trg.size(1)
        att_mask = attention_mask(size)
        if trg.is_cuda:
            att_mask = att_mask.to("cuda")
        trg_mask = trg_mask & att_mask
    else:
        trg_mask = None
    return src_mask, trg_mask


def get_len(train):
    for i, b in enumerate(train):
        pass
    return i
