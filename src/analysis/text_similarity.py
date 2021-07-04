from typing import Tuple
from sentence_transformers import SentenceTransformer, util
import torch


class SentenceSimilarity:
    def __init__(self):
        self.model = SentenceTransformer('stsb-roberta-large')

    def __encode_sentences(self, sentence1: str, sentence2: str) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings1 = self.model.encode(sentence1, convert_to_tensor=True, device='cuda')
        embeddings2 = self.model.encode(sentence2, convert_to_tensor=True, device='cuda')
        return embeddings1, embeddings2

    def cosine_similarity(self, sentence1: str, sentence2: str) -> float:
        encoded1, encoded2 = self.__encode_sentences(sentence1, sentence2)
        score = util.pytorch_cos_sim(encoded1, encoded2)
        return score.item()
