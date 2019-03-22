import torch
from sklearn.metrics import f1_score,classification_report,precision_score,recall_score
def acc(y,yhat):
    mask = y > 0
    return (yhat[mask] == y[mask]).to(torch.float).mean()

def f1score(y,yhat):
    mask = y > 0
    return f1_score(y[mask],yhat[mask],average='micro')

def classificationreport(y,yhat,target_names):
    mask = y > 0
    return classification_report(y[mask], yhat[mask],target_names=target_names)

def precision(y,yhat):
    mask = y > 0
    return precision_score(y[mask], yhat[mask], average='micro')

def recall(y,yhat):
    mask = y > 0
    return recall_score(y[mask], yhat[mask], average='micro')