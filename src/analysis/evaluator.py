from typing import List, Tuple, Dict, Optional
import torch
from tqdm import tqdm
import pandas as pd
from datasets import load_metric
import logging
import sys
import itertools
from transformers import BertTokenizerFast, EncoderDecoderModel, RobertaTokenizerFast

from src.analysis.analysis_helper import AnalysisHelper


class CTransformerEvaluator:

    def __init__(self, eval_dataset_path: str, model_path: str, doc2vec_model_path: str,
                 tokenizer_path: str, log_level='WARNING'):
        self.df = pd.read_csv(eval_dataset_path, index_col=0)
        self.logger = logging.getLogger(__name__)
        self.analysis_helper = AnalysisHelper(doc2vec_model_path)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path, map_location=self.device)

        logging.basicConfig(
            level=log_level,
            handlers=[logging.StreamHandler(sys.stdout)],
            format='%(levelname)s - %(message)s',
        )

    def evaluate(self):
        self.model.eval()


class HFEvaluator:

    def __init__(self, eval_dataset_path: str, model_path: str, doc2vec_model_path: str,
                 tokenizer_path: Optional[str] = None, log_level='WARNING'):
        self.df = pd.read_csv(eval_dataset_path, index_col=0)
        print(len(self.df))
        self.logger = logging.getLogger(__name__)
        self.sari = load_metric('sari')
        self.bert_score = load_metric('bertscore')
        self.rouge = load_metric('rouge')
        self.glue = load_metric('glue', 'stsb')
        self.meteor = load_metric('meteor')
        self.analysis_helper = AnalysisHelper(doc2vec_model_path)
        if tokenizer_path is None:
            tokenizer_path = model_path
            self.logger.info(f'No `tokenizer_path` provided. {model_path} will be used!')

        if 'roberta' in tokenizer_path:
            self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
        elif 'bert' in tokenizer_path:
            self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        else:
            raise ValueError(f'Could not find a suitable tokenizer for: {tokenizer_path}!')
        self.model = EncoderDecoderModel.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logging.basicConfig(
            level=log_level,
            handlers=[logging.StreamHandler(sys.stdout)],
            format='%(levelname)s - %(message)s',
        )

    def __config_model(self, model_config: Dict):
        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.eos_token_id = self.tokenizer.sep_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.encoder.vocab_size
        self.model.config.max_length = model_config['max_length']
        self.model.config.min_length = model_config['min_length']
        self.model.config.no_repeat_ngram_size = model_config['no_repeat_ngram_size']
        self.model.config.early_stopping = model_config['early_stopping']
        self.model.config.length_penalty = model_config['length_penalty']
        self.model.config.num_beams = model_config['num_beams']
        self.model.config.temperature = model_config['temperature']
        self.model.config.top_k = model_config['top_k']
        self.model.config.top_p = model_config['top_p']
        self.model.config.num_beam_groups = model_config['num_beam_groups']
        self.model.config.do_sample = model_config['do_sample']

    def __sources_and_references(self) -> Dict:
        dictionary = {}
        for index, row in self.df.iterrows():
            key = row['Normal'].replace('\n', '')
            value = row['Simple'].replace('\n', '')
            if index % 10 == 0:
                dictionary[key] = [value]
            else:
                dictionary[key].append(value)
        return dictionary

    def generate(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, model_config: Dict) -> List[str]:
        self.__config_model(model_config)
        model = self.model.to(self.device)
        model_output = model.generate(input_ids, attention_mask=attention_mask)
        outputs = self.tokenizer.batch_decode(model_output, skip_special_tokens=True)

        return outputs

    def __tokenize(self, sources: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(sources, padding='max_length', truncation=True, return_tensors='pt')

        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        return input_ids, attention_mask

    def eval_sari_score(self, sources: List[str], predictions: List[str], references: List[List[str]]) -> Dict:
        result = self.sari.compute(sources=sources, predictions=predictions, references=references)
        return {'sari_score': round(result['sari'], 4)}

    def eval_bert_score(self, predictions: List[str], references: List[str]) -> Dict:
        result = self.bert_score.compute(predictions=predictions, references=references, lang='en')
        return {'bert_score_f1': round(result['f1'][0], 4)}

    def eval_meteor_score(self, predictions: List[str], references: List[str]) -> Dict:
        result = self.meteor.compute(predictions=predictions, references=references)
        return {'meteor_score': round(result['meteor'], 4)}

    def eval_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict:
        result = self.rouge.compute(predictions=predictions, references=references, rouge_types=['rouge2'])[
            'rouge2'].mid
        return {
            'rouge2_precision': round(result.precision, 4),
            'rouge2_recall': round(result.recall, 4),
            'rouge2_f_measure': round(result.fmeasure, 4)
        }

    def eval_glue_score(self, predictions: List[int], references: List[int]) -> Dict:
        result = self.glue.compute(predictions=predictions, references=references)
        return {
            'glue_pearson': round(result['pearson'], 4),
            'glue_spearman_r': round(result['spearmanr'], 4)
        }

    def evaluate_for_config_range(self, csv_output_path: str, model_config: Dict):
        dictionary = dict(itertools.islice(self.__sources_and_references().items(), 1))
        result_df = pd.DataFrame(columns=['Index', 'SARI', 'temperature'])

        temperatures = [0.2, 0.75, 0.5, 1.0, 1.5, 2.0, 2.5, 3., 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5,
                        7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
        config = model_config
        index = 0

        for source, references in tqdm(dictionary.items()):
            inputs = self.__tokenize(source)

            for temperature in temperatures:
                config['temperature'] = temperature
                output = self.generate(*inputs, model_config=config)
                sari_result = self.eval_sari_score(sources=[source], predictions=[output[0]],
                                                   references=[references])
                result_df = result_df.append({
                    'Index': index,
                    'SARI': sari_result['sari_score'],
                    'temperature': temperature
                }, ignore_index=True)
            index += 1
        result_df.to_csv(csv_output_path, index=False)


def evaluate_with_dataset(self, model_config: Dict, csv_output_path: Optional[str] = None,
                          extend_dataframe: bool = False):
    result_df = pd.DataFrame(columns=['Normal', 'Simple', 'SARI', 'METEOR', 'ROUGE_F'])

    for source, references in tqdm(self.__sources_and_references().items()):
        inputs = self.__tokenize(source)
        reference_tokens = self.__tokenize(references)
        output = self.generate(*inputs, model_config=model_config)

        glue_result = self.eval_glue_score(predictions=inputs[0][0].tolist(),
                                           references=reference_tokens[0][5].tolist())

        rouge_result = self.eval_rouge_scores(predictions=output, references=[references[0]])
        sari_result = self.eval_sari_score(sources=[source], predictions=[output[0]],
                                           references=[references])
        meteor_result = self.eval_meteor_score(predictions=output, references=[references[0]])

        result_df = result_df.append({
            'Normal': source,
            'Simple': output[0],
            'SARI': sari_result['sari_score'],
            'METEOR': meteor_result['meteor_score'],
            'ROUGE_F': rouge_result['rouge2_f_measure'],
            'SPEARMAN_CORRELATION': glue_result['glue_spearman_r'],
            'PEARSON_CORRELATION': glue_result['glue_pearson']
        }, ignore_index=True)

    if extend_dataframe:
        self.analysis_helper.add_df(result_df)
        self.analysis_helper.sentence_length()
        self.analysis_helper.stop_words_count()
        self.analysis_helper.cosine_similarity()
        result_df = self.analysis_helper.to_df()

    if csv_output_path is not None:
        result_df.to_csv(csv_output_path, index=False)
        print(f'Dataframe saved at: {csv_output_path}.')
