import spacy
from gensim.models.doc2vec import Doc2Vec
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity


class ToVec:
    def __init__(self, model_path: str):
        self.model = Doc2Vec.load(model_path)
        self.nlp = spacy.load("en_core_web_sm")

    def cosine_distance(self, normal_str: str, simple_str: str) -> float:
        normal_vector = self.model.infer_vector(normal_str.split())
        simple_vector = self.model.infer_vector(simple_str.split())
        return spatial.distance.cosine(normal_vector, simple_vector)

    def cosine_similarity(self, normal_string: str, simple_string: str) -> float:
        normal_vector = self.model.infer_vector(normal_string.split())
        simple_vector = self.model.infer_vector(simple_string.split())
        similarity = cosine_similarity([normal_vector], [simple_vector])
        return similarity[0][0]

    def spacy_cosine_similarity(self, normal_string: str, simple_string: str) -> float:
        normal_vector = self.nlp(normal_string)
        simple_vector = self.nlp(simple_string)
        similarity = cosine_similarity([normal_vector], [simple_vector])
        return similarity[0][0]

    def export_w2vec_data(self, save_file_path: str):
        self.model.wv.save_word2vec_format(save_file_path)
