from torch import nn
import torch
from read_data import read_data
from preproc import *
from dataloader import get_dls, Dataset
from model import Model
from training_loop import training_loop


def main():
    training_sentence_tags = read_data('data/train.txt')
    valid_sentence_tags = read_data('data/valid.txt')
    vocabtoidx, labeltoidx = vocab_to_idx(training_sentence_tags)
    dataset_train = Dataset(list(prepare_batch(training_sentence_tags, vocabtoidx, labeltoidx)))
    dataset_valid = Dataset(list(prepare_batch(valid_sentence_tags, vocabtoidx, labeltoidx)))
    train_dl = get_dls(dataset_train, 32)
    valid_dl = get_dls(dataset_valid, 32)
    net = Model(len(vocabtoidx), len(labeltoidx), 50, 50)
    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(net.parameters(), 1e1)
    training_loop(net, opt, loss_func, 5, train_dl, valid_dl)


if __name__ == "__main__":
    main()
