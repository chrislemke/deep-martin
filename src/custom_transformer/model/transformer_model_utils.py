import copy
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable


def get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


def attention_mask(size: int):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_masks(src: torch.Tensor, trg: torch.Tensor):
    src_mask = (src != 0)

    trg_mask = (trg != 0)
    size = trg.size(1)
    att_mask = attention_mask(size)
    if trg.is_cuda:
        att_mask = att_mask.to('cuda')
    trg_masks = (trg_mask, att_mask)

    return src_mask, trg_masks


def get_len(train):
    for i, b in enumerate(train):
        pass
    return i
