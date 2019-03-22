from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, vocab_size, n_tags, emb_size, lstm_hidden_size):
        super(Model, self).__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_size)
        self.lstm = nn.LSTM(emb_size, lstm_hidden_size, batch_first=True)
        self.linear = nn.Linear(lstm_hidden_size, n_tags)

    def forward(self, x):
        embed = self.emb(x)
        hidden, last = self.lstm(embed)
        hidden = hidden.contiguous().view(-1, hidden.shape[2])
        output = self.linear(hidden)
        return output


def model_output(model, dataloader, set_name="train"):
    correct = torch.empty(0, dtype=torch.long)
    predicted = torch.empty(0, dtype=torch.long)
    with torch.no_grad():
        for x, y in dataloader:
            output = model(x)
            preds = torch.max(output, 1)
            correct = torch.cat((correct, y.view(-1)))
            if set_name != "test":
                predicted = torch.cat((predicted, preds[1]))

    return correct, predicted
