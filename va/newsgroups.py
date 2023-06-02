import sys
import urllib.request
from pathlib import Path

from smart_open import open
import torch
from torch.utils.data import TensorDataset


def read_word_counts(file) -> torch.Tensor:
    # The .data files are formatted "docIdx wordIdx count" (0-indexed)
    data = torch.LongTensor([[int(i) for i in line.split()] for line in file])
    return torch.sparse_coo_tensor(data[:,:2].T, data[:,2]).to_dense()


def make_datasets(vocab_size: int = 25000, frequency_smoothing: bool = True):
    with open('data/docword.three.txt.gz') as file:
        train_counts = read_word_counts(file)
    
    vocabulary = torch.topk((train_counts>0).sum(dim=0), vocab_size).indices
    train_counts = train_counts[:, vocabulary].float()

    if frequency_smoothing:
        word_counts = train_counts.sum(dim=0).float()
        word_counts += 1./vocab_size
        frequencies = word_counts / word_counts.sum()

        train_counts += frequencies[None,:]

    return TensorDataset(train_counts)


if __name__ == '__main__':
    print(make_datasets())
