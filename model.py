from torch import nn
import torch
from preproc import word_embeddings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, vocabtoidx, n_tags, emb_size, lstm_hidden_size,pretrained=False,freeze=True):
        super(Model, self).__init__()
        if not pretrained:
            self.emb = nn.Embedding(len(vocabtoidx) + 1, emb_size)
        else:
            word_vecs=word_embeddings(vocabtoidx)
            self.emb = nn.Embedding(word_vecs.shape[1] + 1, emb_size)
            self.emb=self.emb.from_pretrained(word_vecs,freeze=freeze)
        self.lstm = nn.LSTM(emb_size, lstm_hidden_size, batch_first=True)
        self.linear = nn.Linear(lstm_hidden_size, n_tags)

    def forward(self, x):
        embed = self.emb(x)
        hidden, last = self.lstm(embed)
        hidden = hidden.contiguous().view(-1, hidden.shape[2])
        output = self.linear(hidden)
        return output


def model_output(model, dataloader, set_name="train"):
    correct = torch.tensor([], dtype=torch.long).to(device)
    predicted = torch.tensor([], dtype=torch.long).to(device)
    with torch.no_grad():
        for x, y in dataloader:
            x,y=x.to(device),y.to(device)
            output = model(x)
            preds = torch.max(output, 1)
            correct = torch.cat((correct, y.view(-1)))
            if set_name != "test":
                predicted = torch.cat((predicted, preds[1]))

    return correct, predicted
