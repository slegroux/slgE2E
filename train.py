#!/usr/bin/env python
# (c) 2020 Sylvain Le Groux <slegroux@ccrma.stanford.edu>

# libs
import warnings
import random
import numpy as np
from argparse import ArgumentParser
from typing import List, Set, Dict, Tuple, Optional, Callable

# pytorch
import torch
from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

#loggers
import wandb

# deepspeech
from modules import LitSpeech, LibriDataModule

# setup
warnings.filterwarnings("ignore", module='torchaudio') #, category=DeprecationWarning) 
warnings.simplefilter("ignore", UserWarning)

# debug
from IPython import embed

# Reproductibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
seed_everything(seed)


def get_args():
    """ get arguments from model, data and training stages """
    parser = ArgumentParser(description="train end2end speech recognizer")
    # Trainer args
    parser = Trainer.add_argparse_args(parser)
    # Model
    parser = LitSpeech.add_model_specific_args(parser)
    # Data
    parser = LibriDataModule.add_argparse_args(parser)
    # loger
    parser.add_argument("--save_dir", default="wandb/", type=str)
    parser.add_argument("--experiment_name", default="speech_recognition", type=str)
    # Training params (opt)
    # parser.add_argument("--epochs", default=100, type=int)
    # # parser.add_argument("--learning_rate", default=0.0005, type=float)
    parser.add_argument("--early_stop_metric", default="val_loss", type=str)
    parser.add_argument("--early_stop_patience", default=3, type=int)
    # get these from trainer args
    # parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    # parser.add_argument("--amp_level", default="02", type=str)
    # parser.add_argument("--precision", default=32, type=int)
    args = parser.parse_args()
    return(args)


if __name__ == "__main__":
    # args
    args = get_args()

    # model
    model = LitSpeech(**vars(args))
    # model = LitSpeech(
    #     args.n_cnn_layers,
    #     args.n_rnn_layers,
    #     args.rnn_dim,
    #     args.n_class,
    #     args.n_feats,
    #     args.stride,
    #     args.dropout
    # )

    # data
    dm = LibriDataModule.from_argparse_args(args)

    # if args.load_model_from:
    #     speech_module = LitDeepSpeech.load_from_checkpoint(args.load_model_from, model=model, args=args)
    # else:
    #     speech_module = LitDeepSpeech(model, args)

    # loggers
    # tb_logger = TensorBoardLogger(args.logdir, name='speech_recognition')
    wandb.login()
    # run_name = args.learning_rate + '_' + ''
    wandb_logger = WandbLogger(name='test', save_dir=args.save_dir, project=args.experiment_name)
    logger = wandb_logger
    # import pdb; pdb.set_trace()
    # trainer
    # trainer = Trainer(logger=logger)
    # trainer = Trainer(
    #     max_epochs=args.epochs, gpus=args.gpus,
    #     num_nodes=args.nodes, distributed_backend=None,
    #     logger=logger, gradient_clip_val=1.0,
    #     val_check_interval=args.valid_every,
    #     checkpoint_callback=checkpoint_callback(args),
    #     resume_from_checkpoint=args.resume_from_checkpoint
    # )
    #         checkpoint_callback=checkpoint_callback(args), 
    
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        save_top_k=5,
        monitor="val_loss",
        mode="min",
        period=1,
    )
    early_stop = EarlyStopping(monitor=args.early_stop_metric, patience=args.early_stop_patience, verbose=True)
    lr_logger = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_logger, early_stop, checkpoint_callback]

    # to debug: use "unit test" fast_dev_run=True
    # for notebook: progress_bar_refresh_rate=50
    #     amp_level=args.amp_level,
    # precision=args.precision,

    trainer = Trainer.from_argparse_args(
        args,
        # fast_dev_run=True,
        logger=logger,
        gpus=-1,
        # max_epochs=10,
        callbacks=callbacks,
        # profiler=True
    )
    trainer.fit(model, datamodule=dm)

    # trainer.test(speech_module)
    # wandb.finish()