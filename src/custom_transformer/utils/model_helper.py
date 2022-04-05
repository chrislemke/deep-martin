from typing import Tuple

import torch
from scipy.spatial.distance import cosine
from transformers import BertModel


class ModelHelper:
    @staticmethod
    def print_params(model: BertModel):
        params = list(model.named_parameters())

        print("The model has {:} different named parameters.\n".format(len(params)))
        print("==== Embedding Layer ====\n")
        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print("\n==== Transformer ====\n")
        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print("\n==== Output Layer ====\n")
        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    @staticmethod
    def cosine_similarity(
        vector1: Tuple[torch.Tensor, int], vector2: Tuple[torch.Tensor, int]
    ):
        diff = 1 - cosine(vector1[0][vector1[1]], vector2[0][vector2[1]])
        print(f"Cosine similarity: {diff}")
