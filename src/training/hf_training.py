from typing import Tuple, Dict, Optional
from datasets import load_metric, Dataset, load_from_disk
import gc
import sys
import os
import wandb
import functools
import torch
import logging
from transformers import logging as hf_logging
from transformers import EvalPrediction, AutoTokenizer, EncoderDecoderModel, Seq2SeqTrainingArguments, Seq2SeqTrainer


class HuggingFaceTrainer:
    __rouge = load_metric('rouge')
    __bert_score = load_metric('bertscore')
    __meteor = load_metric('meteor')

    __logger = logging.getLogger(__name__)

    @staticmethod
    def setup_logger(level: str = 'INFO'):
        logging.basicConfig(
            level=logging.getLevelName(level),
            handlers=[logging.StreamHandler(sys.stdout)],
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    @staticmethod
    def __load_dataset(path: str) -> Tuple[Dataset, Dataset]:
        dataset = load_from_disk(path)

        ds_dict = dataset.train_test_split(shuffle=False, test_size=0.05)
        train_ds = ds_dict['train'].shuffle(seed=42)
        test_ds = ds_dict['test']

        HuggingFaceTrainer.__logger.info(f' Loaded train_dataset length is: {len(train_ds)}.')
        HuggingFaceTrainer.__logger.info(f' Loaded test_dataset length is: {len(test_ds)}.')

        return train_ds, test_ds

    @staticmethod
    def __compute_metrics(auto_tokenizer, prediction: EvalPrediction):
        tokenizer = auto_tokenizer

        labels_ids = prediction.label_ids
        pred_ids = prediction.predictions

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = \
            HuggingFaceTrainer.__rouge.compute(predictions=pred_str, references=label_str, rouge_types=['rouge2'])[
                'rouge2'].mid
        bert_score_output = HuggingFaceTrainer.__bert_score.compute(predictions=pred_str, references=label_str,
                                                                    lang='en')

        meteor_output = HuggingFaceTrainer.__meteor.compute(predictions=pred_str, references=label_str)
        return {
            'bert_score_f1': round(bert_score_output['f1'][0], 4),
            'meteor_score': round(meteor_output['meteor'], 4),
            'rouge2_precision': round(rouge_output.precision, 4),
            'rouge2_recall': round(rouge_output.recall, 4),
            'rouge2_f_measure': round(rouge_output.fmeasure, 4)
        }

    @staticmethod
    def __setup_wandb(resume, training_config, wandb_config):
        if wandb_config is not None:
            HuggingFaceTrainer.__logger.info(
                f"Starting Wandb with:\n"
                f"API-key: {wandb_config['api_key']}\n"
                f"Entity: {wandb_config['entity']}\n"
                f"Project: {wandb_config['project']}\n"
                f"Name: {training_config['run_name']}\n")

            os.environ['WANDB_API_KEY'] = wandb_config['api_key']
            os.environ['WANDB_DISABLED'] = 'false'
            os.environ['WANDB_PROJECT'] = wandb_config['project']
            os.environ['WANDB_ENTITY'] = wandb_config['entity']

            if resume is True and wandb_config['run_id'] is not None:
                os.environ['WANDB_RESUME'] = 'must'
                os.environ['WANDB_RUN_ID'] = wandb_config['run_id']
            else:
                os.environ['WANDB_RESUME'] = 'never'
                os.environ['WANDB_RUN_ID'] = wandb.util.generate_id()

            HuggingFaceTrainer.__logger.info(f"Run id: {os.environ['WANDB_RUN_ID']}\n")
            return ['wandb', 'tensorboard']
        else:
            os.environ['WANDB_DISABLED'] = 'true'
            HuggingFaceTrainer.__logger.info('Wandb is not running!')
            return ['tensorboard']

    @staticmethod
    def __setup_model(model_config, model_path, pretrained_model_path, resume, tie_encoder_decoder, tokenizer):

        if resume:
            model = EncoderDecoderModel.from_pretrained(pretrained_model_path)
            HuggingFaceTrainer.__logger.info(f'Resuming from: {pretrained_model_path}.')

        elif pretrained_model_path is not None and model_path is None:
            model = EncoderDecoderModel.from_pretrained(pretrained_model_path)
            HuggingFaceTrainer.__logger.info(f'Model loaded from: {pretrained_model_path}.')

        elif pretrained_model_path is not None and model_path is not None:
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_path, model_path,
                                                                        tie_encoder_decoder=tie_encoder_decoder)
            model.save_pretrained(pretrained_model_path)
            HuggingFaceTrainer.__logger.info(f'Model stored at:{pretrained_model_path}')

        else:
            raise ValueError(
                'Please provide either `pretrained_model_path` or `model_path` and `pretrained_model_path`.')

        if tokenizer.name_or_path != 'facebook/bart-base':
            model.config.vocab_size = model.config.encoder.vocab_size

        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.eos_token_id = tokenizer.sep_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.max_length = model_config['max_length']
        model.config.min_length = model_config['min_length']
        model.config.no_repeat_ngram_size = model_config['no_repeat_ngram_size']
        model.config.early_stopping = True
        model.config.length_penalty = model_config['length_penalty']
        model.config.num_beams = model_config['num_beams']
        return model

    @staticmethod
    def train(
            ds_path: str,
            training_output_path: str,
            training_config: Dict,
            model_config: Dict,
            save_model_path: str,
            tokenizer_id: str,
            tie_encoder_decoder: bool,
            wandb_config: Optional[Dict] = None,
            hf_logging_enabled: bool = True,
            resume: bool = False,
            pretrained_model_path: Optional[str] = None,
            model_path: Optional[str] = None):
        gc.enable()

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        compute_metrics = functools.partial(HuggingFaceTrainer.__compute_metrics, tokenizer)
        report_to = HuggingFaceTrainer.__setup_wandb(resume, training_config, wandb_config)
        train_ds, eval_ds = HuggingFaceTrainer.__load_dataset(ds_path)

        if hf_logging_enabled:
            HuggingFaceTrainer.__logger.info('HF logging activated.')
            hf_logging.set_verbosity_info()

        model = HuggingFaceTrainer.__setup_model(model_config, model_path, pretrained_model_path, resume,
                                                 tie_encoder_decoder, tokenizer)

        training_arguments = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            evaluation_strategy='steps',
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            fp16=training_config['fp16'] if torch.cuda.is_available() else False,
            output_dir=training_output_path,
            overwrite_output_dir=False,
            logging_steps=training_config['logging_steps'],
            save_steps=training_config['save_steps'],
            eval_steps=training_config['eval_steps'],
            warmup_steps=training_config['warmup_steps'],
            report_to=report_to,
            run_name=training_config['run_name'],
            ignore_data_skip=resume,
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            save_total_limit=training_config['save_total_limit'])

        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_arguments,
            compute_metrics=compute_metrics,
            train_dataset=train_ds,
            eval_dataset=eval_ds
        )

        if resume:
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        trainer.save_model(save_model_path)

        if wandb_config is not None:
            wandb.finish()

        torch.cuda.empty_cache()
