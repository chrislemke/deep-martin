import argparse
from transformers import AutoTokenizer

from training.custom_transformer_training import TransformerTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_path", type=str)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=int, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--print_every", type=int, default=300)
    parser.add_argument("--lr", type=int, default=0.0001)
    parser.add_argument('--load_weights')
    parser.add_argument("--max_length", type=int, default=80)
    parser.add_argument("--checkpoint", type=str, default='false')
    parser.add_argument("--save_model_path", type=str)
    parser.add_argument("--HF_tokenizer", type=str, default='bert-base-cased')
    parser.add_argument("--loss_plot_save_path", type=str)

    args, _ = parser.parse_known_args()

    tokenizer = AutoTokenizer.from_pretrained(args.HF_tokenizer)
    TransformerTrainer.start_training(args, dataset_path=args.ds_path, vocab_size=tokenizer.vocab_size,
                                      train_file=args.train_file, test_file=args.test_file,
                                      save_model_path=args.save_model_path)
