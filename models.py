import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def toonehot(y):  # y: (N,1) -> (N,10)
    assert(type(y) is torch.Tensor)
    yhot = torch.zeros((y.shape[0], 10))
    idx = 0
    for i in y:
        yhot[idx, i] = 1
        idx += 1
    return yhot.to(y.device)


def plot_loss(losses, title=None):
    train, test = losses[0], losses[1]
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss(Cross-Entropy)')
    plt.legend()
    plt.title(title)
    plt.plot()


def accuracy(y, yp):
    mask = y == yp
    temp = torch.zeros(y.shape[0])
    temp[mask] = 1
    a = torch.sum(temp)/y.shape[0]
    return a


class Logistic_Regression():
    def __init__(self):
        super(Logistic_Regression, self).__init__()
        self.layer = LogisticRegression(max_iter=100)

    def fit(self, x, y):
        self.layer.fit(x, y)

    def predict(self, x):
        out = self.layer.predict(x)
        return out


class SVM():
    def __init__(self):
        super(SVM, self).__init__()
        self.layer = SVC(decision_function_shape='ovo')

    def fit(self, x, y):
        self.layer.fit(x, y)

    def predict(self, x):
        out = self.layer.predict(x)
        return out


class NN1layer(nn.Module):
    def __init__(self, device='cpu', dropout=0.2):
        super(NN1layer, self).__init__()
        self.input_size = 784
        self.classes = 10
        self.dropout = dropout
        self.layer = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(
            self.input_size, self.classes), nn.Softmax(dim=1)).to(device)
        self.loss = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, x):
        out = self.layer(x)
        return out

    def predict(self, x):
        self.eval()
        probs = self.forward(x.to(self.device))
        idx = torch.argmax(probs, dim=1)
        return idx.to(x.device).detach()

    def fit(self, x, y, xe, ye, lr=0.01, eps=20, batch_size=32, regularize=False, print_losses=1):
        if regularize:
            optimizer = torch.optim.Adam(
                params=self.parameters(), lr=lr, weight_decay=lr/100, amsgrad=True)
        else:
            optimizer = torch.optim.Adam(
                params=self.parameters(), lr=lr, amsgrad=True)
        train_loss_list = []
        eval_loss_list = []
        for ep in range(1, eps+1):
            idx = torch.randperm(x.shape[0])
            x = x[idx]
            y = y[idx]
            self.train()
            temploss = 0
            for b in range(0, x.shape[0]-batch_size, batch_size):
                optimizer.zero_grad()
                t = toonehot(y[b:b+batch_size].to(self.device))
                yp = self.forward(x[b:b+batch_size].to(self.device))
                loss = self.loss.forward(yp, t)
                temploss += float(loss)
                loss.backward()
                optimizer.step()
            train_loss_list.append(temploss/int(x.shape[0]/batch_size))
            temploss = 0
            self.eval()
            for b in range(0, xe.shape[0]-batch_size, batch_size):
                t = toonehot(ye[b:b+batch_size].to(self.device))
                yp = self.forward(xe[b:b+batch_size].to(self.device))
                loss = self.loss.forward(yp, t)
                temploss += float(loss)
            eval_loss_list.append(temploss/int(xe.shape[0]/batch_size))
            if ep % print_losses == 0 or ep == 1:
                print('Epoch', ep, ':\n      Training loss:',
                      train_loss_list[-1], ', Evaluation loss:', eval_loss_list[-1])
        return (train_loss_list, eval_loss_list)


class NN3layer(nn.Module):
    def __init__(self, hidden_size=500, device='cpu', dropout=0.2):
        super(NN3layer, self).__init__()
        self.input_size = 784
        self.classes = 10
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.layer = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(in_features=self.input_size, out_features=self.hidden_size), nn.ReLU(), nn.BatchNorm1d(self.hidden_size),
                                   nn.Dropout(self.dropout), nn.Linear(
                                       in_features=self.hidden_size, out_features=self.hidden_size), nn.ReLU(), nn.BatchNorm1d(self.hidden_size),
                                   nn.Dropout(self.dropout), nn.Linear(in_features=self.hidden_size, out_features=self.classes), nn.Softmax(dim=1)).to(device)
        self.loss = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, x):
        out = self.layer(x)
        return out

    def predict(self, x):
        self.eval()
        probs = self.forward(x.to(self.device))
        idx = torch.argmax(probs, dim=1)
        return idx.to(x.device).detach()

    def fit(self, x, y, xe, ye, lr=0.01, eps=20, batch_size=32, regularize=False, print_losses=1):
        if regularize:
            optimizer = torch.optim.Adam(
                params=self.parameters(), lr=lr, weight_decay=lr/100, amsgrad=True)
        else:
            optimizer = torch.optim.Adam(
                params=self.parameters(), lr=lr, amsgrad=True)
        train_loss_list = []
        eval_loss_list = []
        for ep in range(1, eps+1):
            idx = torch.randperm(x.shape[0])
            x = x[idx]
            y = y[idx]
            self.train()
            temploss = 0
            for b in range(0, x.shape[0]-batch_size, batch_size):
                optimizer.zero_grad()
                t = toonehot(y[b:b+batch_size].to(self.device))
                yp = self.forward(x[b:b+batch_size].to(self.device))
                loss = self.loss.forward(yp, t)
                temploss += float(loss)
                loss.backward()
                optimizer.step()
            train_loss_list.append(temploss/int(x.shape[0]/batch_size))
            temploss = 0
            self.eval()
            for b in range(0, xe.shape[0]-batch_size, batch_size):
                t = toonehot(ye[b:b+batch_size].to(self.device))
                yp = self.forward(xe[b:b+batch_size].to(self.device))
                loss = self.loss.forward(yp, t)
                temploss += float(loss)
            eval_loss_list.append(temploss/int(xe.shape[0]/batch_size))
            if ep % print_losses == 0 or ep == 1:
                print('Epoch', ep, ':\n      Training loss:',
                      train_loss_list[-1], ', Evaluation loss:', eval_loss_list[-1])
        return (train_loss_list, eval_loss_list)


class NN10layer(nn.Module):
    def __init__(self, hidden_size=100, device='cpu', dropout=0.2):
        super(NN10layer, self).__init__()
        self.input_size = 784
        self.classes = 10
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.layer = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(in_features=self.input_size, out_features=self.hidden_size), nn.ReLU(), nn.BatchNorm1d(self.hidden_size),
                                   nn.Dropout(self.dropout), nn.Linear(
                                       in_features=self.hidden_size, out_features=self.hidden_size), nn.ReLU(), nn.BatchNorm1d(self.hidden_size),
                                   nn.Dropout(self.dropout), nn.Linear(
                                       in_features=self.hidden_size, out_features=self.hidden_size), nn.ReLU(), nn.BatchNorm1d(self.hidden_size),
                                   nn.Dropout(self.dropout), nn.Linear(
                                       in_features=self.hidden_size, out_features=self.hidden_size), nn.ReLU(), nn.BatchNorm1d(self.hidden_size),
                                   nn.Dropout(self.dropout), nn.Linear(
                                       in_features=self.hidden_size, out_features=self.hidden_size), nn.ReLU(), nn.BatchNorm1d(self.hidden_size),
                                   nn.Dropout(self.dropout), nn.Linear(
                                       in_features=self.hidden_size, out_features=self.hidden_size), nn.ReLU(), nn.BatchNorm1d(self.hidden_size),
                                   nn.Dropout(self.dropout), nn.Linear(
                                       in_features=self.hidden_size, out_features=self.hidden_size), nn.ReLU(), nn.BatchNorm1d(self.hidden_size),
                                   nn.Dropout(self.dropout), nn.Linear(
                                       in_features=self.hidden_size, out_features=self.hidden_size), nn.ReLU(), nn.BatchNorm1d(self.hidden_size),
                                   nn.Dropout(self.dropout), nn.Linear(
                                       in_features=self.hidden_size, out_features=self.hidden_size), nn.ReLU(), nn.BatchNorm1d(self.hidden_size),
                                   nn.Dropout(self.dropout), nn.Linear(in_features=self.hidden_size, out_features=self.classes), nn.Softmax(dim=1)).to(device)
        self.loss = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, x):
        out = self.layer(x)
        return out

    def predict(self, x):
        self.eval()
        probs = self.forward(x.to(self.device))
        idx = torch.argmax(probs, dim=1)
        return idx.to(x.device).detach()

    def fit(self, x, y, xe, ye, lr=0.01, eps=20, batch_size=32, regularize=False, print_losses=1):
        if regularize:
            optimizer = torch.optim.Adam(
                params=self.parameters(), lr=lr, weight_decay=lr/100, amsgrad=True)
        else:
            optimizer = torch.optim.Adam(
                params=self.parameters(), lr=lr, amsgrad=True)
        train_loss_list = []
        eval_loss_list = []
        for ep in range(1, eps+1):
            idx = torch.randperm(x.shape[0])
            x = x[idx]
            y = y[idx]
            self.train()
            temploss = 0
            for b in range(0, x.shape[0]-batch_size, batch_size):
                optimizer.zero_grad()
                t = toonehot(y[b:b+batch_size].to(self.device))
                yp = self.forward(x[b:b+batch_size].to(self.device))
                loss = self.loss.forward(yp, t)
                temploss += float(loss)
                loss.backward()
                optimizer.step()
            train_loss_list.append(temploss/int(x.shape[0]/batch_size))
            temploss = 0
            self.eval()
            for b in range(0, xe.shape[0]-batch_size, batch_size):
                t = toonehot(ye[b:b+batch_size].to(self.device))
                yp = self.forward(xe[b:b+batch_size].to(self.device))
                loss = self.loss.forward(yp, t)
                temploss += float(loss)
            eval_loss_list.append(temploss/int(xe.shape[0]/batch_size))
            if ep % print_losses == 0 or ep == 1:
                print('Epoch', ep, ':\n      Training loss:',
                      train_loss_list[-1], ', Evaluation loss:', eval_loss_list[-1])
        return (train_loss_list, eval_loss_list)


class CNN3NN3layer(nn.Module):
    def __init__(self, device='cpu', dropout=0.2, hidden_size=500):
        super(CNN3NN3layer, self).__init__()
        self.classes = 10
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(nn.Dropout(self.dropout), nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2), nn.ReLU(), nn.BatchNorm2d(8),
                                   nn.Dropout(self.dropout), nn.Conv2d(
                                       in_channels=8, out_channels=16, kernel_size=3, stride=1), nn.ReLU(), nn.BatchNorm2d(16),
                                   nn.Dropout(self.dropout), nn.Conv2d(
                                       in_channels=16, out_channels=32, kernel_size=2, stride=1), nn.ReLU(), nn.BatchNorm2d(32),
                                   nn.Flatten(), nn.Dropout(self.dropout), nn.Linear(in_features=10*10*32,
                                                                                     out_features=self.hidden_size), nn.ReLU(), nn.BatchNorm1d(self.hidden_size),
                                   nn.Dropout(self.dropout), nn.Linear(
                                       in_features=self.hidden_size, out_features=self.hidden_size), nn.ReLU(), nn.BatchNorm1d(self.hidden_size),
                                   nn.Dropout(self.dropout), nn.Linear(in_features=self.hidden_size, out_features=self.classes), nn.Softmax(dim=1)).to(device)
        self.loss = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, x):
        out = self.layer(x)
        return out

    def predict(self, x):
        self.eval()
        probs = self.forward(x.to(self.device).view(x.shape[0], 1, 28, 28))
        idx = torch.argmax(probs, dim=1)
        return idx.to(x.device).detach()

    def fit(self, x, y, xe, ye, lr=0.01, eps=20, batch_size=32, regularize=False, print_losses=1):
        if regularize:
            optimizer = torch.optim.Adam(
                params=self.parameters(), lr=lr, weight_decay=lr/100, amsgrad=True)
        else:
            optimizer = torch.optim.Adam(
                params=self.parameters(), lr=lr, amsgrad=True)
        train_loss_list = []
        eval_loss_list = []
        for ep in range(1, eps+1):
            idx = torch.randperm(x.shape[0])
            x = x[idx]
            y = y[idx]
            self.train()
            temploss = 0
            for b in range(0, x.shape[0]-batch_size, batch_size):
                optimizer.zero_grad()
                t = toonehot(y[b:b+batch_size].to(self.device))
                yp = self.forward(
                    x[b:b+batch_size].to(self.device).view(batch_size, 1, 28, 28))
                loss = self.loss.forward(yp, t)
                temploss += float(loss)
                loss.backward()
                optimizer.step()
            train_loss_list.append(temploss/int(x.shape[0]/batch_size))
            temploss = 0
            self.eval()
            for b in range(0, xe.shape[0]-batch_size, batch_size):
                t = toonehot(ye[b:b+batch_size].to(self.device))
                yp = self.forward(
                    xe[b:b+batch_size].to(self.device).view(batch_size, 1, 28, 28))
                loss = self.loss.forward(yp, t)
                temploss += float(loss)
            eval_loss_list.append(temploss/int(xe.shape[0]/batch_size))
            if ep % print_losses == 0 or ep == 1:
                print('Epoch', ep, ':\n      Training loss:',
                      train_loss_list[-1], ', Evaluation loss:', eval_loss_list[-1])
        return (train_loss_list, eval_loss_list)


class CNN10(nn.Module):
    def __init__(self, device='cpu', dropout=0.2):
        super(CNN10, self).__init__()
        self.classes = 10
        self.dropout = dropout
        self.layer = nn.Sequential(nn.Dropout(self.dropout), nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2), nn.ReLU(), nn.BatchNorm2d(8),  # 13
                                   nn.Dropout(self.dropout), nn.Conv2d(
                                       in_channels=8, out_channels=16, kernel_size=3, stride=1), nn.ReLU(), nn.BatchNorm2d(16),  # 11
                                   nn.Dropout(self.dropout), nn.Conv2d(
                                       in_channels=16, out_channels=32, kernel_size=3, stride=1), nn.ReLU(), nn.BatchNorm2d(32),  # 9
                                   nn.Dropout(self.dropout), nn.Conv2d(
                                       in_channels=32, out_channels=64, kernel_size=3, stride=1), nn.ReLU(), nn.BatchNorm2d(64),  # 7
                                   nn.Dropout(self.dropout), nn.Conv2d(
                                       in_channels=64, out_channels=128, kernel_size=3, stride=1), nn.ReLU(), nn.BatchNorm2d(128),  # 5
                                   nn.Dropout(self.dropout), nn.Conv2d(
                                       in_channels=128, out_channels=256, kernel_size=3, stride=1), nn.ReLU(), nn.BatchNorm2d(256),  # 3
                                   nn.Dropout(self.dropout), nn.Conv2d(
                                       in_channels=256, out_channels=512, kernel_size=2, stride=1), nn.ReLU(), nn.BatchNorm2d(512),  # 2
                                   nn.Dropout(self.dropout), nn.Conv2d(in_channels=512, out_channels=10, kernel_size=2, stride=1), nn.Flatten(), nn.Softmax(dim=1)).to(device)  # 1
        self.loss = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, x):
        out = self.layer(x)
        return out

    def predict(self, x):
        self.eval()
        probs = self.forward(x.to(self.device).view(x.shape[0], 1, 28, 28))
        idx = torch.argmax(probs, dim=1)
        return idx.to(x.device).detach()

    def fit(self, x, y, xe, ye, lr=0.01, eps=20, batch_size=32, regularize=False, print_losses=1):
        if regularize:
            optimizer = torch.optim.Adam(
                params=self.parameters(), lr=lr, weight_decay=lr/100, amsgrad=True)
        else:
            optimizer = torch.optim.Adam(
                params=self.parameters(), lr=lr, amsgrad=True)
        train_loss_list = []
        eval_loss_list = []
        for ep in range(1, eps+1):
            idx = torch.randperm(x.shape[0])
            x = x[idx]
            y = y[idx]
            self.train()
            temploss = 0
            for b in range(0, x.shape[0]-batch_size, batch_size):
                optimizer.zero_grad()
                t = toonehot(y[b:b+batch_size].to(self.device))
                yp = self.forward(
                    x[b:b+batch_size].to(self.device).view(batch_size, 1, 28, 28))
                loss = self.loss.forward(yp, t)
                temploss += float(loss)
                loss.backward()
                optimizer.step()
            train_loss_list.append(temploss/int(x.shape[0]/batch_size))
            temploss = 0
            self.eval()
            for b in range(0, xe.shape[0]-batch_size, batch_size):
                t = toonehot(ye[b:b+batch_size].to(self.device))
                yp = self.forward(
                    xe[b:b+batch_size].to(self.device).view(batch_size, 1, 28, 28))
                loss = self.loss.forward(yp, t)
                temploss += float(loss)
            eval_loss_list.append(temploss/int(xe.shape[0]/batch_size))
            if ep % print_losses == 0 or ep == 1:
                print('Epoch', ep, ':\n      Training loss:',
                      train_loss_list[-1], ', Evaluation loss:', eval_loss_list[-1])
        return (train_loss_list, eval_loss_list)


'''
x = torch.rand((1000,28*28))
y = torch.randint(0, 10, (1000,))
xe = torch.rand((100,28*28))
ye = torch.randint(0, 10, (100,))

clf = NN3layer(dropout=0,device='cuda')
losses = clf.fit(x, y, xe, ye,regularize=True)
print(clf.predict(xe)[1],ye[1],clf.predict(xe)[1]==ye[1])
plot_loss(losses)
print('Accuracy:',accuracy(ye,clf.predict(xe)))
'''
