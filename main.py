import torch
from torch import nn
from read_data import read_data
from preproc import *
from dataloader import get_dls, Dataset
from model import Model
from training_loop import training_loop


def main(batch_size, emb_size, hidden_size, learning_rate, epochs):
    training_sentence_tags = read_data('data/train.txt')
    valid_sentence_tags = read_data('data/valid.txt')
    vocabtoidx, labeltoidx = vocab_to_idx(training_sentence_tags)
    dataset_train = Dataset(list(prepare_batch(training_sentence_tags, vocabtoidx, labeltoidx)))
    dataset_valid = Dataset(list(prepare_batch(valid_sentence_tags, vocabtoidx, labeltoidx)))
    train_dl = get_dls(dataset_train, batch_size)
    valid_dl = get_dls(dataset_valid, batch_size * 2)
    net = Model(len(vocabtoidx), len(labeltoidx), emb_size, hidden_size)
    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(net.parameters(), learning_rate, momentum=0.9)
    training_loop(net, opt, loss_func, epochs, train_dl, valid_dl)
    

if __name__ == "__main__":
    main()
