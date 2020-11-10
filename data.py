import torchaudio
from torch.utils.data import Dataset
import torch.nn as nn
import string
from typing import List, Set, Dict, Tuple, Optional
from IPython import embed

LIBRI_DIR = "/home/syl20/data/en/librispeech"

class CharMap:
    def __init__(self):
        m = {}
        m["'"] = 0
        m['<SPACE>'] = 1
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
        

class Data(object):
    def __init__(self):
        self.train_ds, self.test_ds = None, None
        self.train_tf, self.test_tf = None, None

    def get_libri_ds(self, data_dir:str=LIBRI_DIR)->(Dataset, Dataset):
        self.train_ds = torchaudio.datasets.LIBRISPEECH(data_dir, url="train-clean-100", download=False)
        self.test_ds = torchaudio.datasets.LIBRISPEECH(data_dir, url="test-clean", download=False)
        return(self.train_ds, self.test_ds)

    def init_tf(self):
        self.train_tf = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35)
        )
        self._test_tf = torchaudio.transforms.MelSpectrogram()
    
    def data_prep(self):
        
