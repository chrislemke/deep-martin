import codecs
import functools
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, Dataset
from torchtext.legacy import data
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer


class DatasetHelper:
    @staticmethod
    def concat_and_split(
        csv_paths: List[str],
        train_csv_path: str,
        test_csv_path: str,
        shuffle: bool = False,
    ):
        dataframes: List[pd.DataFrame] = []
        for path in tqdm(csv_paths):
            df = pd.read_csv(path)
            dataframes.append(df)
        final_df = pd.concat(dataframes, ignore_index=True)
        if shuffle:
            final_df = final_df.sample(frac=1)

        train_share = int(len(final_df) * 0.8)
        train = final_df[:train_share]
        test = final_df[train_share + 1 :]

        train.to_csv(train_csv_path)
        test.to_csv(test_csv_path)

    @staticmethod
    def train_test_split(
        train_size: float,
        df: Optional[pd.DataFrame] = None,
        csv_path: Optional[str] = None,
        shuffle: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if df is not None:
            print("Using DataFrame provided.")
        elif csv_path is not None:
            df = pd.read_csv(csv_path, index_col=0)
            print("Loaded DataFrame from CSV.")
        else:
            raise ValueError("Neither a DataFrame nor a CSV path is provided.")
        if shuffle:
            df = df.sample(frac=1)
        train_share = int(len(df) * train_size)
        train_df = df[:train_share]
        test_df = df[train_share + 1 :]
        return train_df, test_df

    @staticmethod
    def concat_csvs(csv_paths: List[str], output_csv_path: str, shuffle: bool = False):
        dataframes: List[pd.DataFrame] = []
        for path in csv_paths:
            df = pd.read_csv(path)
            dataframes.append(df)
        final_df = pd.concat(dataframes, ignore_index=True)
        if shuffle:
            final_df = final_df.sample(frac=1)
        final_df.to_csv(output_csv_path, index=False)

    @staticmethod
    def concat_split(
        middle_csv_path: str,
        to_split_csv_path: str,
        output_csv_path: str,
        first_split: float = 0.8,
    ):
        middle_df = pd.read_csv(middle_csv_path)
        to_split_df = pd.read_csv(to_split_csv_path)
        print("middle_df.shape", middle_df.shape)
        print("to_split_df.shape", to_split_df.shape)
        to_split_df = to_split_df.sample(frac=1)
        split_row = int(len(to_split_df) * first_split)
        top_split = to_split_df.iloc[:split_row]
        bottom_split = to_split_df.iloc[split_row:]

        df = pd.concat([top_split, middle_df, bottom_split])
        if "Unnamed: 0" in df.columns:
            df.drop(["Unnamed: 0"], axis=1, inplace=True)

        df.reset_index(drop=True)
        print("df.shape", df.shape)
        print(df.head())
        print(df.tail())
        df.to_csv(output_csv_path, index=False)

    @staticmethod
    def data_loaders_from_datasets(
        train_dataset: Dataset, val_dataset: Dataset, batch_size: int, pin_memory=False
    ) -> Tuple[DataLoader, DataLoader]:
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
        val_data_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
        return train_data_loader, val_data_loader


class HuggingFaceDataset:
    encoder_max_length = 80
    decoder_max_length = 100

    @staticmethod
    def hf_dataset(
        df: pd.DataFrame,
        remove_columns_list: List[str],
        identifier: str,
        batch_size: int = 8,
    ) -> Dataset:
        """
        :param df: Pandas DataFrame which contain a `Normal` and a `Simple` column containing sentences or short paragraphs.
        :remove_columns_list: A list of columns which should be removed. Those columns will not be a part of the dataset
        :identifier: The identifier is also known as the path for the tokenizer (e.g. `bert-base-cased`).
        :batch_size: The default batch size is set to 8. This is also the default value from Hugging Face.
        """

        data = Dataset.from_pandas(df)

        tokenizer = AutoTokenizer.from_pretrained(identifier)
        print(f"Using {identifier} tokenizer.")

        function = functools.partial(HuggingFaceDataset.__process, tokenizer)

        dataset = data.map(
            function=function,
            batched=True,
            batch_size=batch_size,
            remove_columns=remove_columns_list,
        )

        dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "decoder_input_ids",
                "decoder_attention_mask",
                "labels",
            ],
        )
        print(dataset.info)
        return dataset

    @staticmethod
    def __process(auto_tokenizer, batch: Dict):
        tokenizer = auto_tokenizer

        inputs = tokenizer(
            batch["Normal"],
            padding="max_length",
            truncation=True,
            max_length=HuggingFaceDataset.encoder_max_length,
        )
        outputs = tokenizer(
            batch["Simple"],
            padding="max_length",
            truncation=True,
            max_length=HuggingFaceDataset.decoder_max_length,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in batch["labels"]
        ]

        return batch


class DatasetManager:
    @staticmethod
    def create_asset_csv(source_folder_path: str, output_file_path: str):
        """
        :param source_folder_path: Path to the folder which contains the test and validation files. Files need
        to have specific names: `test.orig.txt`, `valid.orig.txt`, `test.simp.0.txt`, `valid.simp.0.txt`, etc.
        :param output_file_path: Path to the output csv file.
        """
        df = pd.DataFrame(columns=["Normal", "Simple"])

        with open(f"{source_folder_path}/train.orig.txt", "r") as train_orig_file:
            for line_index, orig_line in enumerate(train_orig_file):
                for i in range(10):
                    with open(
                        f"{source_folder_path}/train.simp.{i}.txt"
                    ) as train_simp_file:
                        simple_lines = train_simp_file.readlines()
                        df = df.append(
                            {"Normal": orig_line, "Simple": simple_lines[line_index]},
                            ignore_index=True,
                        )

        with open(f"{source_folder_path}/valid.orig.txt", "r") as valid_orig_file:
            for line_index, orig_line in enumerate(valid_orig_file):
                for i in range(10):
                    with open(
                        f"{source_folder_path}/valid.simp.{i}.txt"
                    ) as valid_simp_file:
                        simple_lines = valid_simp_file.readlines()
                        df = df.append(
                            {"Normal": orig_line, "Simple": simple_lines[line_index]},
                            ignore_index=True,
                        )

        df.to_csv(output_file_path)

    @staticmethod
    def create_one_stop_csv_from_txt(txt_path: str, output_file_path: str):
        """Create csv from txt files containing separated sentences."""
        rows: List[str] = []

        with codecs.open(txt_path, "r", encoding="utf-8") as file:
            for line in tqdm(file):
                if line != "*******\n":
                    rows.append(line)

        df = pd.DataFrame(
            list(zip(rows[0::2], rows[1::2])), columns=["Normal", "Simple"]
        )
        df.to_csv(output_file_path)

    @staticmethod
    def create_simpa_csv(source_folder_path: str, output_file_path: str):
        df = pd.DataFrame(columns=["Normal", "Simple"])

        for simpa_type in tqdm(["ls", "ss"]):
            with open(
                f"{source_folder_path}/{simpa_type}_normal.txt", "r"
            ) as normal_file:
                with open(
                    f"{source_folder_path}/{simpa_type}_simplified.txt", "r"
                ) as simple_file:
                    simple_lines = simple_file.readlines()
                    for line_index, line in enumerate(normal_file):
                        df = df.append(
                            {"Normal": line, "Simple": simple_lines[line_index]},
                            ignore_index=True,
                        )

        df.to_csv(output_file_path)

    @staticmethod
    def create_ss_csv(input_file_path: str, output_file_path: str):
        df = pd.DataFrame(columns=["Normal", "Simple"])
        with open(input_file_path, "r") as file:
            for line in tqdm(file):
                split = re.split(r"\t+", line)
                df = df.append(
                    {"Normal": split[0], "Simple": split[1]}, ignore_index=True
                )

        df.to_csv(output_file_path)

    @staticmethod
    def create_wiki_large_csv(
        normal_file_path: str,
        simple_file_path: str,
        output_file_path: str,
        shuffle: bool = False,
    ):
        """
        :param normal_file_path: Path to the text file containing the normal sentences.
        :param simple_file_path: Path to the text file containing the simplified (target) sentences.
        :param output_file_path: Path to the output csv file.
        :param shuffle: Shuffle the dataset after process.
        """
        df = pd.DataFrame(columns=["Normal", "Simple"])
        with open(simple_file_path, "r") as simple_file:
            with open(normal_file_path, "r") as normal_file:
                normal_lines = normal_file.readlines()
                for index, simple_line in tqdm(enumerate(simple_file)):
                    df = df.append(
                        {"Normal": normal_lines[index], "Simple": simple_line},
                        ignore_index=True,
                    )

        if shuffle:
            df = df.sample(frac=1)
        df.to_csv(output_file_path)

    @staticmethod
    def create_cleaned_wiki_split_tsv(tsv_file_path: str, output_file_path: str):
        """
        :param tsv_file_path: Path to tsv file where the source sentence is separated from the target sentence by tab.
        :param output_file_path: Path for the output tsv file.
        """
        delete_list = ["<::::> "]
        with open(tsv_file_path) as in_file, open(output_file_path, "w+") as out_file:
            for line in tqdm(in_file):
                for string in delete_list:
                    line = line.replace(string, "")
                out_file.write(line)

    @staticmethod
    def create_wiki_split_csv(
        tsv_file_path: str,
        output_csv_path: str,
        shuffle: bool = True,
        clean_up: bool = False,
        content_to_drop: Optional[List[str]] = None,
    ):
        tqdm.pandas()

        df = pd.read_csv(tsv_file_path, sep="\t")

        if clean_up and content_to_drop is not None:
            df = df[~df["Normal"].isin(content_to_drop)]

        if shuffle:
            df = df.sample(frac=1)
        df.to_csv(output_csv_path, header=["Normal", "Simple"])


class TextDataset:
    __tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    def __init__(
        self,
        path: str,
        train_file: str,
        test_file: str,
        batch_size: int,
        device: torch.device,
        max_length: int,
    ):
        super(TextDataset, self).__init__()
        __df = pd.read_csv(f"{path}/{train_file}")
        self.batch_size = batch_size
        self.device = device
        self.vocab_size = TextDataset.__tokenizer.vocab_size
        self.rows = __df.shape[0]

        normal = data.Field(
            sequential=True,
            use_vocab=False,
            tokenize=TextDataset.__tokenize,
            pad_token=0,
            fix_length=max_length,
        )
        simple = data.Field(
            sequential=True,
            use_vocab=False,
            tokenize=TextDataset.__tokenize,
            pad_token=0,
            is_target=True,
            fix_length=max_length,
        )

        fields = {"Normal": ("normal", normal), "Simple": ("simple", simple)}

        self.train_data, self.test_data = data.TabularDataset.splits(
            path=path,
            train=train_file,
            test=test_file,
            format="csv",
            fields=fields,
            skip_header=False,
        )

    @staticmethod
    def __tokenize(text: str):
        return TextDataset.__tokenizer(text)["input_ids"]

    def iterators(self):
        train_iterator, test_iterator = data.BucketIterator.splits(
            (self.train_data, self.test_data),
            batch_size=self.batch_size,
            device=self.device,
            shuffle=False,
        )
        return train_iterator, test_iterator
