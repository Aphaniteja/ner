import torch
from utils import acc


def training_loop(net, opt, loss_func, epochs, train_dl, valid_dl):
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        total_valid_loss = 0
        total_valid_acc = 0
        for x, y in train_dl:
            output = net(x)
            preds = torch.max(output, 1)
            loss = loss_func(output, y.view(-1))
            loss.backward()
            total_loss += loss.item()
            total_acc += acc(y.view(-1), preds[1])
            opt.step()
            opt.zero_grad()
        with torch.no_grad():
            for x, y in valid_dl:
                output = net(x)
                preds = torch.max(output, 1)
                loss = loss_func(output, y.view(-1))
                total_valid_loss += loss.item()
                total_valid_acc += acc(preds[1], y.view(-1))

        print(total_loss / len(train_dl), total_acc / len(train_dl), total_valid_loss / len(valid_dl),
              total_valid_acc / len(valid_dl))

