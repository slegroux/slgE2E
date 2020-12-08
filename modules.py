#!/usr/bin/env python
# (c) 2020 Sylvain Le Groux <slegroux@ccrma.stanford.edu>

# libs
import warnings
from typing import List, Set, Dict, Tuple, Optional, Callable
import os
import ast
import random
import numpy as np

# loggers
from comet_ml import Experiment
import wandb

# model
from model import SpeechRecognitionModel
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
from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
# from pytorch_lightning.metrics import 
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

# debug
from IPython import embed
from pdb import set_trace

class LitDummy(LightningModule):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def loss(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def training_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def training_epoch_end(self, outputs) -> None:
        torch.stack([x["loss"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    def validation_epoch_end(self, outputs) -> None:
        torch.stack([x['x'] for x in outputs]).mean()

    def test_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        self.log('fake_test_acc', loss)
        return {"y": loss}

    def test_epoch_end(self, outputs) -> None:
        torch.stack([x["y"] for x in outputs]).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


class LitDeepSpeech(LightningModule):

    def __init__(self, model, args):
        """method used to define model's parameters"""
        super().__init__()
        self.model = model
        self.criterion = nn.CTCLoss(blank=29, zero_infinity=True)
        self.args = args
        self.train_dl_length = len(self.train_dataloader())
        # save hp to self.hparams
        self.save_hyperparameters()

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

    def train_dataloader(self):
        '''returns training dataloader'''
        spec_aug = nn.Sequential(
        MelSpectrogram(sample_rate=16000, n_mels=128),
        FrequencyMasking(freq_mask_param=15),
        TimeMasking(time_mask_param=35)
        )
        train_dataset = LibriDataset(ROOT_DIR, CharacterTokenizer, dataset=self.args.train_url, transform=spec_aug)
        dl = DataLoader(dataset=train_dataset,
                            batch_size=self.args.batch_size,
                            num_workers=self.args.data_workers,
                            pin_memory=True,
                            collate_fn=train_dataset.collate)
        return(dl)

    def v_step(self, batch):
        spectrograms, labels, input_lengths, label_lengths = batch 
        output = self.model(spectrograms)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)
        loss = self.criterion(output, labels, input_lengths, label_lengths)
        decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
        n_correct_pred = sum([int(a == b) for a, b in zip(decoded_preds, decoded_targets)])

        test_cer, test_wer = [], []
        for j in range(len(decoded_preds)):
            test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
            test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

        avg_cer = torch.FloatTensor([sum(test_cer) / len(test_cer)])
        avg_wer = torch.FloatTensor([sum(test_wer) / len(test_wer)])  # Need workt to make all operations in torch
        logs = {
            "cer": avg_cer,
            "wer": avg_wer,
        }


        # c = cer(decoded, targets)
        # w = wer(decoded, targets)
        # self.log(c, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log(w, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.v_step(batch)
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

    def val_dataloader(self):
        '''returns validation dataloader'''
        mel_spec = MelSpectrogram(sample_rate=16000, n_mels=128)
        test_dataset = LibriDataset(ROOT_DIR, CharacterTokenizer, dataset=self.args.valid_url, transform=mel_spec)
        return DataLoader(dataset=test_dataset,
                            batch_size=self.args.batch_size,
                            num_workers=self.args.data_workers,
                            collate_fn=test_dataset.collate,
                            pin_memory=True)
    
    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_dataloader(self):
        '''returns test dataloader'''
        mel_spec = MelSpectrogram(sample_rate=16000, n_mels=128)
        test_dataset = LibriDataset(ROOT_DIR, CharacterTokenizer, dataset=self.args.valid_url, transform=mel_spec)
        return DataLoader(dataset=test_dataset,
                            batch_size=self.args.batch_size,
                            num_workers=self.args.data_workers,
                            collate_fn=test_dataset.collate,
                            pin_memory=True)

    def configure_optimizers(self):
        # define training optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), self.args.learning_rate)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.learning_rate, 
                                            steps_per_epoch=self.train_dl_length,
                                            epochs=self.args.epochs,
                                            anneal_strategy='linear')
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #                                 self.optimizer, mode='min',
        #                                 factor=0.50, patience=6)
        return [self.optimizer], [self.scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument("--n_cnn_layers", default=3, type=int)
        parser.add_argument("--n_rnn_layers", default=5, type=int)
        parser.add_argument("--rnn_dim", default=512, type=int)
        parser.add_argument("--n_class", default=29, type=int)
        parser.add_argument("--n_feats", default=128, type=str)
        parser.add_argument("--stride", default=2, type=int)
        parser.add_argument("--dropout", default=0.1, type=float)

        return parser