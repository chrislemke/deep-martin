import re
from typing import Optional

import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm

from src.analysis.doc2vec import ToVec
from src.analysis.text_similarity import SentenceSimilarity


class AnalysisHelper:
    def __init__(
        self,
        doc2vec_model_path: str,
        dataframe: Optional[pd.DataFrame] = None,
        csv_load_path: Optional[str] = None,
    ):

        if csv_load_path is not None:
            self.df = pd.read_csv(csv_load_path)
        elif dataframe is not None:
            self.df = dataframe
        else:
            print(
                "`csv_load_path` and `dataframe` not set! DataFrame must be provided with `add_df`."
            )
        self.to_vec = ToVec(doc2vec_model_path)
        self.sentence_similarity = SentenceSimilarity()
        tqdm.pandas()

    @staticmethod
    def __stop_word_count(sentence):
        count = 0
        for word in sentence.split():
            word = re.sub(r"[^a-zA-Z]+", "", word)
            if word.lower() in STOP_WORDS:
                count += 1
        return count

    def add_df(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def stop_words_count(self):
        print("'stop_words_count':")
        self.df["Normal_stop_word_count"] = self.df["Normal"].apply(
            lambda x: AnalysisHelper.__stop_word_count(x)
        )
        self.df["Simple_stop_word_count"] = self.df["Simple"].apply(
            lambda x: AnalysisHelper.__stop_word_count(x)
        )

    def sentence_length(self):
        print("'sentence_length':")
        self.df["Normal_length"] = self.df["Normal"].apply(lambda x: len(x.split()))
        self.df["Simple_length"] = self.df["Simple"].apply(lambda x: len(x.split()))

    def __cosine_distance(self, row) -> float:
        return self.to_vec.cosine_distance(row["Normal"], row["Simple"])

    def __cosine_similarity(self, row) -> float:
        if type(row["Normal"]) is str and type(row["Simple"] is str):
            return self.to_vec.cosine_similarity(row["Normal"], row["Simple"])
        else:
            return 0.0

    def __sentence_similarity(self, row) -> float:
        return self.sentence_similarity.cosine_similarity(row["Normal"], row["Simple"])

    def cosine_distance(self):
        print("'Cosine_distance':")
        self.df["Cosine_distance"] = self.df.progress_apply(
            lambda row: self.__cosine_distance(row), axis=1
        )

    def cosine_similarity(self):
        print("'Cosine_similarity':")
        self.df["Cosine_similarity"] = self.df.progress_apply(
            lambda row: self.__cosine_similarity(row), axis=1
        )

    def sentence_cosine_similarity(self):
        print("'Sentence_similarity':")
        self.df["Sentence_similarity"] = self.df.progress_apply(
            lambda row: self.__sentence_similarity(row), axis=1
        )

    def store_to_csv(self, csv_save_path: Optional[str]):
        self.df.to_csv(csv_save_path, index_label="index")
        print(f"CSV stored at: {csv_save_path}.")

    def to_df(self) -> pd.DataFrame:
        return self.df
