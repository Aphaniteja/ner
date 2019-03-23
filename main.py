import torch
import argparse
from torch import nn
from read_data import read_data
from preproc import *
from dataloader import get_dls, Dataset
from model import Model, model_output
from training_loop import training_loop
from utils import classificationreport


def main(batch_size, emb_size, hidden_size, learning_rate, epochs, verbosity, pretrained, freeze):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training_sentence_tags = read_data('data/train.txt')
    valid_sentence_tags = read_data('data/valid.txt')
    vocabtoidx, labeltoidx = vocab_to_idx(training_sentence_tags)
    dataset_train = Dataset(list(prepare_batch(training_sentence_tags, vocabtoidx, labeltoidx)))
    dataset_valid = Dataset(list(prepare_batch(valid_sentence_tags, vocabtoidx, labeltoidx)))
    train_dl = get_dls(dataset_train, batch_size)
    valid_dl = get_dls(dataset_valid, batch_size * 2)
    net = Model(vocabtoidx, len(labeltoidx), emb_size, hidden_size, pretrained, freeze).to(device)
    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(net.parameters(), learning_rate)
    training_loop(net, opt, loss_func, epochs, train_dl, valid_dl, verbosity)
    correct, predicted = model_output(net, valid_dl)
    print(classificationreport(correct.cpu(), predicted.cpu(), target_names=list(zip(*labeltoidx.items()))[0][:-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", type=int, help="batch_size", default=32)
    parser.add_argument("-embs", "--emb_size", type=int, help="emb_size", default=50)
    parser.add_argument("-hs", "--hidden_size", type=int, help="lstm_hidden_size", default=50)
    parser.add_argument("-lr", "--learning_rate", help="learning rate", default=1e1)
    parser.add_argument("-e", "--epochs", type=int, help="no of epochs", default=5)
    parser.add_argument("-v", "--verbosity", help="choose 0 for non verbose", default=1)
    parser.add_argument("-p", "--pretrained", help="choose True for pretrained", default=False)
    parser.add_argument("-f", "--freeze", help="choose False for changing pretrained embeddings", default=True)

    args = parser.parse_args()
    main(**vars(args))
