import torch
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt


#mnist = fetch_openml("mnist_784",version = 1, data_home=".")

X = mnist.data / 255
Y = mnist.target
Y = np.array(Y)
Y = Y.astype(np.int32)

plt.imshow(X[0].reshape(28,28), cmap="gray")
print("この画像は{:.0f}".format(Y[0]))

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=1/7, random_state=0)

X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)

ds_train = TensorDataset(X_train, Y_train)
ds_test = TensorDataset(X_test, Y_test)

loader_train = DataLoader(ds_train, batch_size=64, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=64, shuffle=False)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net(28*28, 100, 10)

from torch import optim

loss_fn = nn.CrossEntropyLoss

optimizer = optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    for data, targets in loader_train:
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()
    print(epoch)

def test():
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, targets in loader_test:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data.view_as(predicted)).sum()

    data_num = len(loader_test.dataset)
    print(correct/data_num)

test()