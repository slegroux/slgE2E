#!/usr/bin/env python

from data import CharacterTokenizer, LibriDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio
from IPython import embed
import pytest

def test_char_map():
    cm = CharacterTokenizer()
    assert cm.map == {"'": 0, '<SPACE>': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'g': 8, 'h': 9, 'i': 10, 'j': 11, 'k': 12, 'l': 13, 'm': 14, 'n': 15, 'o': 16, 'p': 17, 'q': 18, 'r': 19, 's': 20, 't': 21, 'u': 22, 'v': 23, 'w': 24, 'x': 25, 'y': 26, 'z': 27, '<PAD>': 28}
    assert cm.text2int('I am') == [10, 1, 2, 14]

def test_char_map_inverse():
    cm = CharacterTokenizer()
    assert cm.pam == {0: "'", 1: '<SPACE>', 2: 'a', 3: 'b', 4: 'c', 5: 'd', 6: 'e', 7: 'f', 8: 'g', 9: 'h', 10: 'i', 11: 'j', 12: 'k', 13: 'l', 14: 'm', 15: 'n', 16: 'o', 17: 'p', 18: 'q', 19: 'r', 20: 's', 21: 't', 22: 'u', 23: 'v', 24: 'w', 25: 'x', 26: 'y', 27: 'z', 28: '<PAD>'}
    assert cm.int2text([10, 1, 2, 14]) == 'i am'

def test_load_data():
    spec_train, text_train = torch.load('data/train.pt')
    assert len(spec_train) == 28539
    spec_test, text_test = torch.load('data/test.pt')
    assert len(spec_test) == 2620
    
def test_ds():
    ROOT_DIR = "/home/syl20/data/en/librispeech"
    spec_aug = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        torchaudio.transforms.TimeMasking(time_mask_param=35)
    )
    ds = LibriDataset(ROOT_DIR, CharacterTokenizer, dataset="test-clean", transform=spec_aug)
    assert len(ds) == 2620

    dl = DataLoader(ds, batch_size=20, collate_fn=ds.collate)
    for i in range(2):
        sample = next(iter(dl))
        assert (sample[0][0].shape) == (1, 128, 1258)
        assert (sample[1][0].shape) == torch.Size([220])





