from torch.utils.data import DataLoader
import numpy as np
from torch import tensor
import torch


class Dataset:
    def __init__(self, x):
        self.x = x

    def __getitem__(self, idx):
        sequence = self.x[idx][0]
        tags = self.x[idx][1]
        return sequence, tags, len(sequence)

    def __len__(self):
        return len(self.x)


def collatebatch(batch):
    batch_size = len(batch)
    seqs, tags, lengths = list(zip(*batch))
    max_length = max(lengths)

    padded_seqs = torch.zeros((batch_size, max_length), dtype=torch.long)
    padded_tags = torch.zeros((batch_size, max_length), dtype=torch.long)
    for idx, length in enumerate(lengths):
        padded_seqs[idx, 0:length] = tensor(seqs[idx][0:length])
        padded_tags[idx, 0:length] = tensor(tags[idx][0:length])
    return padded_seqs, padded_tags


def get_dls(dataset, bs=32):
    dataloader = DataLoader(dataset, bs, collate_fn=collatebatch)
    return dataloader
