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

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_c = 3, embed_dim = 512):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size = patch_size, stride = patch_size)
        
    def forward(self, x):# x [batch, in_c, img_size, img_size] -> [batch, embed_size, n_patch/2, n_patch/2] -> [batch, embed_size, n_patch] -> [batch, n_patch, embed_size]
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        return x
        
class Attention(nn.Module):
    def __init__(self, dim, n_heads = 8, qkv_bias = True, attn_p = 0., proj_p = 0.):
        super(Attention, self).__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
        
    def forward(self, x):
        n_samples, n_toks, dim = x.shape 
        if dim != self.dim:
            raise ValueError
        qkv = self.qkv(x)
        qkv = qkv.reshape(n_samples, n_toks, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2,-1)
        dp = (q @ k_t) * self.scale
        attn = dp.softmax(dim = -1)
        attn = self.attn_drop(attn)
        w_av = attn @ v
        w_av = w_av.transpose(1,2)
        w_av = w_av.flatten(2)
        x = self.proj(w_av)
        x = self.proj_drop(x)
        return x
        
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p = 0.):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_features, hidden_features), nn.GELU(), 
                                    nn.Linear(hidden_features, out_features), nn.Dropout(p))
    
    def forward(self, x):
        x = self.layers(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio = 4.0, qkv_bias = True, p = 0., attn_p = 0.):
        super(Transformer, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps = 1E-6)
        self.attn = Attention(dim, n_heads, qkv_bias, attn_p, p)
        self.norm2 = nn.LayerNorm(dim, eps = 1E-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_features, dim, p)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size = 28, patch_size = 7, in_c = 1, n_classes = 10, embed_dim = 512, depth = 10, n_heads = 8, mlp_ratio = 4., qkv_bias = True, dropout=0., device='cpu'):
        super(VisionTransformer, self).__init__()
        self.dropout = dropout
        self.loss = nn.CrossEntropyLoss()
        self.device = device
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_c = in_c, embed_dim = embed_dim).to(device)
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim)).to(device)
        self.pos_embed = nn.Parameter(torch.zeros(1,1+self.patch_embed.n_patches,embed_dim)).to(device)
        self.pos_drop = nn.Dropout(dropout).to(device)
        self.transformers = nn.ModuleList([Transformer(dim = embed_dim, n_heads = n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=dropout, attn_p=dropout) for _ in range(depth)]).to(device)
        self.norm = nn.LayerNorm(embed_dim, eps = 1E-6).to(device)
        self.head = nn.Linear(embed_dim, n_classes).to(device)
        self.prob_dist = nn.Softmax(dim = -1).to(device)

    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for t in self.transformers:
            x = t(x)
        x = self.norm(x)
        cls_token_final = x[:,0]
        x = self.head(cls_token_final)
        out = self.prob_dist(x)
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
#%%
'''
x = torch.rand((1000,1,28,28))
y = torch.randint(0, 10, (1000,))
xe = torch.rand((100,1,28,28))
ye = torch.randint(0, 10, (100,))

clf = VisionTransformer(dropout=0,device='cuda')
losses = clf.fit(x, y, xe, ye,regularize=True)
print(clf.predict(xe)[1],ye[1],clf.predict(xe)[1]==ye[1])
plot_loss(losses)
print('Accuracy:',accuracy(ye,clf.predict(xe)))
'''
