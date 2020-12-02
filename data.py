#!/usr/bin/env python
# (c) 2020 Sylvain Le Groux <slegroux@ccrma.stanford.edu>

import torchaudio
import torch
from torch.utils.data import Dataset
from torchaudio.datasets import LIBRISPEECH
import torch.nn as nn
import string
from typing import List, Set, Dict, Tuple, Optional, Callable
from torch import Tensor, IntTensor
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
        # TODO(slg): <EOS>?
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
        spec_len = waveform.shape[-1] #// 2
        label = IntTensor(self.tokenizer.text2int(utterance))
        label_len = len(label)
        return(waveform, label, spec_len, label_len)
        # return(waveform, sample_rate, utterance, IntTensor(self.tokenizer.text2int(utterance)), speaker_id, chapter_id, utterance_id)

    def collate(self, batch):
        # batch: (feature, sample_rate, utterance, tokens, speaker_id, chapter_id, utterance_id)
        # features: (channel * n_bin * n_frames)
        # feature:item[0], utterance: item[2], tokens: item[3]
        # padding requires padding dim (n_frames) to be first => transpose(0,2)
        features = [ item[0].squeeze(0).transpose(0,1) for item in batch ]
        # print("features shape", [x.shape for x in features])
        features_len = [ item[2] for item in batch ] # waveform first channel exract dim(n_frames)
        # print("max", max(features_len))
        labels = [ item[1] for item in batch ]
        labels_len = [item[3] for item in batch]
        # print("max", max(labels_len))
        # print("labels shape", [label.shape for label in labels])
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.map['<PAD>'])
        # print("padded labels ", [x.shape for x in labels])
        # (batch,channel,n_bins,n_frames)
        features = nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0).unsqueeze(1).transpose(2,3)
        # print("padded feats", [x.shape for x in features])

        return(features, labels, features_len, labels_len)

