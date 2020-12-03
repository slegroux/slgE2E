#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from comet_ml import Experiment

import torch.nn.functional as F
import torch, torchaudio
from torch import Tensor, IntTensor
from typing import List, Set, Dict, Tuple, Optional, Callable
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.utils.data as data
import os
import ast

from model import SpeechRecognitionModel
from decode import GreedyDecoder
from metrics import wer, cer
import torch.optim as optim
from data import CharacterTokenizer, LibriDataset
from argparse import ArgumentParser
from IPython import embed

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

torchaudio.set_audio_backend("sox_io")

ROOT_DIR = "/home/syl20/data/en/librispeech"

# COMET_API_KEY = os.environ['COMET_API_KEY']
COMET_API_KEY = None
# project_name = "speechrecognition"
# experiment_name = "speechrecognition-colab"

if COMET_API_KEY:
  experiment = Experiment(api_key=COMET_API_KEY, project_name=project_name, parse_args=False)
  experiment.set_name(experiment_name)
  experiment.display()
else:
  experiment = Experiment(api_key='dummy_key', disabled=True)


class LitSpeechRec(LightningModule):

    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.criterion = nn.CTCLoss(blank=29, zero_infinity=True)
        self.args = args
        self.train_dl_length = len(self.train_dataloader())

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.model.parameters(), self.args.learning_rate)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.learning_rate, 
                                            steps_per_epoch=self.train_dl_length,
                                            epochs=self.args.epochs,
                                            anneal_strategy='linear')
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #                                 self.optimizer, mode='min',
        #                                 factor=0.50, patience=6)
        return [self.optimizer], [self.scheduler]

    def step(self, batch):
        spectrograms, labels, input_lengths, label_lengths = batch 
        output = self.model(spectrograms)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)
        loss = self.criterion(output, labels, input_lengths, label_lengths)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        logs = {'loss': loss, 'lr': self.optimizer.param_groups[0]['lr'] }
        return {'loss': loss, 'log': logs}

    def train_dataloader(self):
        spec_aug = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        torchaudio.transforms.TimeMasking(time_mask_param=35)
        )
        train_dataset = LibriDataset(ROOT_DIR, CharacterTokenizer, dataset=self.args.train_url, transform=spec_aug)
        dl = DataLoader(dataset=train_dataset,
                            batch_size=self.args.batch_size,
                            num_workers=self.args.data_workers,
                            pin_memory=True,
                            collate_fn=train_dataset.collate)
        return(dl)

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.scheduler.step(avg_loss)
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def val_dataloader(self):
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
        test_dataset = LibriDataset(ROOT_DIR, CharacterTokenizer, dataset=self.args.valid_url, transform=mel_spec)
        return DataLoader(dataset=test_dataset,
                            batch_size=self.args.batch_size,
                            num_workers=self.args.data_workers,
                            collate_fn=test_dataset.collate,
                            pin_memory=True)


def checkpoint_callback(args):
    return ModelCheckpoint(
        filepath=args.save_model_path,
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

def get_args():

    parser = ArgumentParser(description="train end2end speech recognizer")

    # distributed training setup
    parser.add_argument('-n', '--nodes', default=1, type=int, help='number of data loading workers')
    parser.add_argument('-g', '--gpus', default=8, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--data_workers', default=100, type=int,
                        help='n data loading workers, default 0 = main process only')
    parser.add_argument('-db', '--dist_backend', default='ddp', type=str,
                        help='which distributed backend to use. defaul ddp')

    # train and valid
    parser.add_argument('--train_url', default='train-clean-100', required=True, type=str,
                        help='url file to load training data')
    parser.add_argument('--valid_url', default='test-clean', required=True, type=str,
                        help='url file to load testing data')
    parser.add_argument('--valid_every', default=50, required=False, type=int,
                        help='valid after every N iteration')

    # dir and path for models and logs
    parser.add_argument('--save_model_path', default=None, required=True, type=str,
                        help='path to save model')
    parser.add_argument('--load_model_from', default=None, required=False, type=str,
                        help='path to load a pretrain model to continue training')
    parser.add_argument('--resume_from_checkpoint', default=None, required=False, type=str,
                        help='check path to resume from')
    parser.add_argument('--logdir', default='tb_logs', required=False, type=str,
                        help='path to save logs')
    
    # general
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='size of batch')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--pct_start', default=0.3, type=float, help='percentage of growth phase in one cycle')
    parser.add_argument('--div_factor', default=100, type=int, help='div factor for one cycle')
    parser.add_argument("--hparams_override", default="{}", type=str, required=False,
		help='override the hyper parameters, should be in form of dict. ie. {"attention_layers": 16 }')
    parser.add_argument("--dparams_override", default="{}", type=str, required=False,
		help='override the data parameters, should be in form of dict. ie. {"sample_rate": 8000 }')

    args = parser.parse_args()
    args.hparams_override = ast.literal_eval(args.hparams_override)
    args.dparams_override = ast.literal_eval(args.dparams_override)


    if args.save_model_path:
       if not os.path.isdir(os.path.dirname(args.save_model_path)):
           raise Exception("the directory for path {} does not exist".format(args.save_model_path))

    return(args)



if __name__ == "__main__":
    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 30,
        "n_feats": 128,
        "stride":2,
        "dropout": 0.1,
        "learning_rate": 5e-4,
        "batch_size": 64,
        "epochs": 5
    }
    
    args = get_args()

    model = SpeechRecognitionModel(hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
        )

    if args.load_model_from:
        speech_module = LitSpeechRec.load_from_checkpoint(args.load_model_from, model=model, args=args)
    else:
        speech_module = LitSpeechRec(model, args)

    tb_logger = TensorBoardLogger(args.logdir, name='speech_recognition')
    wandb_logger = WandbLogger(name='test_log',project='speech_recognition')

    trainer = Trainer(logger=logger)

    trainer = Trainer(
        max_epochs=args.epochs, gpus=args.gpus,
        num_nodes=args.nodes, distributed_backend=None,
        logger=logger, gradient_clip_val=1.0,
        val_check_interval=args.valid_every,
        checkpoint_callback=checkpoint_callback(args),
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    trainer.fit(speech_module)