#!/usr/bin/env python
# (c) 2020 Sylvain Le Groux <slegroux@ccrma.stanford.edu>

import torch
from data import CharacterTokenizer
from IPython import embed

def GreedyDecoder(output, labels, label_lengths, tokenizer=CharacterTokenizer, blank_label=29, collapse_repeated=True):
	arg_maxes = torch.argmax(output, dim=2)
	decodes = []
	targets = []
	ct = tokenizer()
	for i, args in enumerate(arg_maxes):
		decode = []
		targets.append(ct.int2text(labels[i][:label_lengths[i]].tolist()))
		for j, index in enumerate(args):
			if index != blank_label:
				if collapse_repeated and j != 0 and index == args[j -1]:
					continue
				decode.append(index.item())
		decodes.append(ct.int2text(decode))
	return decodes, targets