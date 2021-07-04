import logging
import time
import torch
import torch.nn.functional as F
import sys
import pandas as pd

from src.preprocessing.dataset import TextDataset
from src.custom_transformer.model.transformer import get_model
from src.custom_transformer.model import transformer_model_utils as tu
from src.analysis.plotting import Plotter


class TransformerTrainer:
    __logger = logging.getLogger(__name__)

    @staticmethod
    def setup_logger(level: str = 'INFO'):
        logging.basicConfig(
            level=logging.getLevelName(level),
            handlers=[logging.StreamHandler(sys.stdout)],
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    @staticmethod
    def start_training(arguments, vocab_size: int, dataset_path: str, train_file: str, test_file: str,
                       save_model_path: str):
        TransformerTrainer.setup_logger('INFO')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_model(arguments, vocab_size, vocab_size, device)

        model.train()
        start = time.time()

        TransformerTrainer.__logger.info(f'Creating dataset from {dataset_path}.')
        dataset = TextDataset(path=dataset_path, train_file=train_file, test_file=test_file,
                              batch_size=arguments.batch_size, max_length=80, device=device)
        data_iter = dataset.iterators()[0]

        TransformerTrainer.__logger.info('Training starting ...')
        losses = []
        for epoch in range(arguments.epochs):
            running_loss = 0
            epoch_loss = 0

            if arguments.checkpoint == 'true' and save_model_path is not None:
                if epoch > 0:
                    torch.save(model.state_dict(), f'{save_model_path}/model_checkpoint_epoch-{epoch}.pt')
                    TransformerTrainer.__logger.info(
                        f'Checkpoint saved to: {save_model_path}/model_checkpoint_epoch-{epoch}.')

            arguments.train_len = tu.get_len(data_iter)
            optimizer = torch.optim.Adam(model.parameters(), lr=arguments.lr, betas=(0.9, 0.98), eps=1e-9)

            for i, batch in enumerate(data_iter):
                src_input = batch.normal
                trg = batch.simple

                optimizer.zero_grad()
                src_mask, trg_masks = tu.create_masks(src_input, trg)
                outputs = model(src_input, trg, src_mask, trg_masks)

                loss = F.cross_entropy(outputs.reshape(-1, outputs.size(-1)), trg.reshape(-1), ignore_index=-100)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_loss += loss.item() * src_input.size(0)

                if (i + 1) % arguments.print_every == 0:
                    p = int(100 * (i + 1) / arguments.train_len)
                    print("   %dm: epoch %d [%s%s]  %d%%  train loss for mini batch = %.6f" % \
                          ((time.time() - start) // 60, epoch + 1, "".join('#' * (p // 5)),
                           "".join(' ' * (20 - (p // 5))), p, running_loss / arguments.print_every))
                    running_loss = 0
            losses.append(epoch_loss / (dataset.rows / src_input.size(0)))

        if arguments.save_model_path is not None:
            torch.save(model.state_dict(), f'{save_model_path}/model.pt')
            TransformerTrainer.__logger.info(f'Model saved to: {save_model_path}/model.pt.')

        TransformerTrainer.__logger.info('... training finished!')

        if arguments.loss_plot_save_path is not None:
            Plotter.plot_loss(losses, f'{arguments.loss_plot_save_path}/loss_plot.html', x_label='Epoch')
        df = pd.DataFrame(losses, columns=['Loss'])
        df.index += 1
        df.index.names = ['Epoch']
        print(df)
