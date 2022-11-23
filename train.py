#!/usr/bin/python3
import os
import warnings
import argparse
from torch.cuda import device_count

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import wandb
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, Callback
from pytorch_lightning.loggers import WandbLogger

from lit_models.lit_model import LitModel
from lit_models.data_module import DataModule

warnings.filterwarnings(action='ignore')
wandb_logger = WandbLogger(project="CycleGAN")

is_image_uploaded = False

class LogPredictionsCallback(Callback):
    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            n = 12
            real_A, real_B = batch['A'], batch['B']
            fake_A, fake_B = outputs['fake_A'], outputs['fake_B']

            columns = ['real_A', 'real_B', 'fake_A', 'fake_B']

            data = [[wandb.Image(ra), wandb.Image(rb), wandb.Image(fa), wandb.Image(fb)]
                    for ra, rb, fa, fb in zip(real_A, real_B, fake_A, fake_B)]

            wandb_logger.log_table(key='on_valid_end',
                                   columns=columns,
                                   data=data)

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=12, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='data/img2real/', help='root directory of the dataset')
    parser.add_argument('--save_path', type=str, default='ckpts/', help='root directory of the dataset')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

    return parser.parse_args()

def get_callbacks(args):
    # Callbacks
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    chk_callback = ModelCheckpoint(
        dirpath=args.save_path,
        filename='{epoch:02d}-{loss_G:.3f}',
        verbose=True,
        save_top_k=5,
        monitor='loss_G',
        mode='min'
    )

    log_predictions_callback = LogPredictionsCallback()

    return [
            chk_callback, 
            log_predictions_callback, 
            RichProgressBar()
            ]

def main(args):
    dm = DataModule(root=args.dataroot, batch_size=args.batch_size)
    model = LitModel(**vars(args))
    trainer = Trainer(
            # Training Options
            accelerator='auto',
            devices=torch.cuda.device_count(),
            precision=16,
            amp_backend='apex',
            max_epochs=args.n_epochs,
            # Callbacks
            callbacks=get_callbacks(args),
            # Logger
            logger=wandb_logger
    )
    trainer.fit(model, dm)
    return

if __name__ == '__main__':
    main(args_parse())

