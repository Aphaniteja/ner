from torch import nn
import torch
class Model(nn.Module):
    def __init__(self,vocab_size,n_tags,emb_size,lstm_hidden_size):
        super(Model,self).__init__()
        self.emb=nn.Embedding(vocab_size+1,emb_size)
        self.lstm=nn.LSTM(emb_size,lstm_hidden_size,batch_first=True)
        self.linear=nn.Linear(lstm_hidden_size,n_tags)

    def forward(self,x):
        embed=self.emb(x)
        hidden,last=self.lstm(embed)
        hidden=hidden.contiguous().view(-1, hidden.shape[2])
        output=self.linear(hidden)
        return output