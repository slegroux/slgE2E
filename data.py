#!/usr/bin/env python
# (c) 2020 Sylvain Le Groux <slegroux@ccrma.stanford.edu>

import torchaudio
import torch
from torch.utils.data import Dataset
from torchaudio.datasets import LIBRISPEECH
import torch.nn as nn
import string
from typing import List, Set, Dict, Tuple, Optional, Callable
from torch import Tensor
from IPython import embed

LIBRI_DIR = "/home/syl20/data/en/librispeech"


class CharacterTokenizer:
    """
    Convert sentences into indices of characters and back
    """
    def __init__(self):
        m = {}
        m["'"] = 0
        m['<SPACE>'] = 1
        m['<PAD>'] = 28
        counter = 2
        for letter in string.ascii_lowercase:
            m[letter] = counter
            counter += 1
        self._map = m
        self._pam = self.inverse_map(self.map)
    
    @staticmethod
    def inverse_map(map:Dict[str, int])->Dict[int,str]:
        return(dict([ (v,k) for (k,v) in map.items()]))
    
    @property
    def map(self)->Dict[str,int]:
        return(self._map)

    @property
    def pam(self)->Dict[int,str]:
        return(self._pam)
    
    def text2int(self, text:str)->List[int]:
        ints = []
        for c in text.lower():
            if c == " ":
                i = self._map['<SPACE>']
            else:
                i = self._map[c]
            ints.append(i)
        return(ints)

    def int2text(self, ints:List[int])->List[str]:
        chars = []
        for i in ints:
            c = self._pam[i]
            chars.append(c)
        return(''.join(chars).replace("<SPACE>", ' '))
        

class LibriDataset(LIBRISPEECH):
    def __init__(self, root_dir:str, tokenizer:Callable, dataset:str="train-clean-100",  transform:Callable=None):
        super().__init__(root_dir, url=dataset, download=False)
        self.transform = transform
        self.tokenizer = tokenizer()

    def __getitem__(self, n:int)->Tuple[Tensor, int, str, int, int, int]:
        (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id) = \
            super().__getitem__(n)
        if self.transform:
            waveform = self.transform(waveform)

        return(waveform, sample_rate, utterance, torch.Tensor(self.tokenizer.text2int(utterance)), speaker_id, chapter_id, utterance_id)

    def collate(self, batch):
        # (feature, sample_rate, utterance, tokens, speaker_id, chapter_id, utterance_id)
        # features (channel * n_bin * n_frames)
        # padding requires padding dim (n_frames) to be first => transpose(0,2)    
        features = [ item[0].transpose(0,2) for item in batch ]
        labels = [ item[3] for item in batch ]
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.map['<PAD>'])
        # (batch,channel,n_bins,n_frames)
        features = nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0).transpose(1,3)

        return([features,labels])
