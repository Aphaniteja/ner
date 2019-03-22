import torch
def acc(yhat, y):
    mask = y > 0
    return (yhat[mask] == y[mask]).to(torch.float).mean()

def f1score(yhat,y):
    pass

def precision(yhat):
    pass