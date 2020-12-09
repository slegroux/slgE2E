#!/usr/bin/env python
# (c) 2020 Sylvain Le Groux <slegroux@ccrma.stanford.edu>

# libs
import warnings
from typing import List, Set, Dict, Tuple, Optional, Callable
import os
import ast
import random
import numpy as np

# model
from models import SpeechRecognitionModel
from decode import GreedyDecoder
from metrics import wer, cer
from data import CharacterTokenizer, LibriDataset
from argparse import ArgumentParser

# torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor, IntTensor
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.utils.data as data
import torch

# torchaudio
import torch
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torchaudio
torchaudio.set_audio_backend("sox_io")
from torchaudio.transforms import Spectrogram, MelSpectrogram, FrequencyMasking, TimeMasking

# lightning
from pytorch_lightning import LightningModule, LightningDataModule

# debug
from IPython import embed
from pdb import set_trace

# setup
ROOT_DIR = "/home/syl20/data/en/librispeech"

class LitSpeech(LightningModule):

    def __init__(
                self, 
                n_cnn_layers,
                n_rnn_layers,
                rnn_dim,
                n_class,
                n_feats,
                stride=2,
                dropout=0.1,
                learning_rate=0.0005,
                **kwargs
                ):
        """method used to define model's parameters"""
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = SpeechRecognitionModel(n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1)
        self.criterion = nn.CTCLoss(blank=29, zero_infinity=True)
        # self.train_dl_length = len(self.train_dataloader())        

    def forward(self, x):
        """for inference"""
        return self.model(x)

    def step(self, batch):
        spectrograms, labels, input_lengths, label_lengths = batch 
        output = self.model(spectrograms)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)
        loss = self.criterion(output, labels, input_lengths, label_lengths)
        return loss

    def training_step(self, batch, batch_idx):
        """return loss from single batch"""
        loss = self.step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def v_step(self, batch):
        spectrograms, labels, input_lengths, label_lengths = batch 
        output = self.model(spectrograms)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)
        loss = self.criterion(output, labels, input_lengths, label_lengths)
        decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)

        test_cer, test_wer = [], []
        for j in range(len(decoded_preds)):
            test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
            test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

        avg_cer = torch.FloatTensor([sum(test_cer) / len(test_cer)])
        avg_wer = torch.FloatTensor([sum(test_wer) / len(test_wer)])  # Need workt to make all operations in torch

        return {'val_loss': loss, 'cer': avg_cer, 'wer': avg_wer}

    def validation_step(self, batch, batch_idx):
        loss = self.v_step(batch)['val_loss']
        cer_ = self.v_step(batch)['cer']
        wer_ = self.v_step(batch)['wer']
        self.log("cer", cer_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("wer", wer_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    # def validation_epoch_end(self, outputs):
    #     """
    #     Called at the end of validation to aggregate outputs.
    #     :param outputs: list of individual outputs of each validation step.
    #     """
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     val_acc = sum([x["n_correct_pred"] for x in outputs]) / sum(x["n_pred"] for x in outputs)
    #     avg_wer = torch.stack([x["wer"] for x in outputs]).mean()
    #     avg_cer = torch.stack([x["cer"] for x in outputs]).mean()
    #     tensorboard_logs = {"Loss/val": avg_loss, "val_acc": val_acc, "Metrics/wer": avg_wer, "Metrics/cer": avg_cer}
    #     return {"val_loss": avg_loss, "log": tensorboard_logs, "wer": avg_wer, "cer": avg_cer}

    def validation_epoch_end(self, val_step_outputs):
        # [results epoch 1, results epoch 2, ...]
        avg_val_loss = torch.Tensor([ x['loss'] for x in val_step_outputs ]).mean()
        return {'val_loss': avg_val_loss}
    
    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        # define training optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), self.learning_rate)
        # self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.learning_rate, 
        #                                     steps_per_epoch=self.train_dl_length,
        #                                     epochs=self.args.epochs,
        #                                     anneal_strategy='linear')
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #                                 self.optimizer, mode='min',
        #                                 factor=0.50, patience=6)
        return [self.optimizer]#, [self.scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--n_cnn_layers", default=3, type=int)
        parser.add_argument("--n_rnn_layers", default=5, type=int)
        parser.add_argument("--rnn_dim", default=512, type=int)
        parser.add_argument("--n_class", default=30, type=int)
        parser.add_argument("--n_feats", default=128, type=str)
        parser.add_argument("--stride", default=2, type=int)
        parser.add_argument("--dropout", default=0.1, type=float)

        return parser


class LibriDataModule(LightningDataModule):
    def __init__(self, data_dir='/home/syl20/data/en/librispeech', train_set='train-clean-5', val_set='dev-clean-2', test_set='dev-clean-2', batch_size=64, num_workers=(torch.get_num_threads()-2)):
        super().__init__()
        self.data_dir = data_dir
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_transform = MelSpectrogram(sample_rate=16000, n_mels=128)
        self.train_transform = nn.Sequential(
            MelSpectrogram(sample_rate=16000, n_mels=128),
            FrequencyMasking(freq_mask_param=15),
            TimeMasking(time_mask_param=35)
        )
    
    def prepare_data(self):
        """ called on one GPU only """
        pass
    
    def setup(self, stage=None):
        """ called on every GPU """
        if stage == 'fit' or stage is None:
            self.train = LibriDataset(self.data_dir, CharacterTokenizer, dataset=self.train_set, transform=self.train_transform)
            self.val = LibriDataset(self.data_dir, CharacterTokenizer, dataset=self.val_set, transform=self.val_transform)
            self.dims = self.train[0][0].shape

        if stage == 'test' or stage is None:
            self.test = LibriDataset(self.data_dir, CharacterTokenizer, dataset=self.test_set, transform=self.val_transform)
            self.dims = self.test[0][0].shape

    def train_dataloader(self):
        return DataLoader(dataset=self.train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.train.collate,
                pin_memory=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.val.collate,
                pin_memory=True)    

    def test_dataloader(self):
        return DataLoader(dataset=self.test,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.test.collate,
                pin_memory=True)

    # @staticmethod
    # def add_argparse_args(parent_parser):
    #     """
    #     Define parameters that only apply to this model
    #     """
    #     parser = ArgumentParser(parents=[parent_parser])
    #     parser.add_argument("--data_dir", default='/home/syl20/data/en/librispeech', type=str)
    #     parser.add_argument("--train_set", default='train-clean-5', type=str)
    #     parser.add_argument("--val_set", default='dev-clean-2', type=str)
    #     parser.add_argument("--test_set", default='dev-clean-2', type=str)
    #     parser.add_argument("--batch_size", default=64, type=int)
    #     parser.add_argument("--num_workers", default=100, type=int)

    #     return parser