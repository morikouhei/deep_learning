import torch

def activation(x):

    return 1 / (1 + torch.exp(-x))

features = torch.randn((1,5))
weights = torch.randn_like(features)
bias = torch.randn((1,1))
y = activation(torch.mm(features,weights.view(-1,1))+bias)

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import helper
from torch import nn
import torch.nn.functional as F
from torch import optim

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])
trainset = datasets.FashionMNIST("~/.pytorch/F_MNIST_data/", download=True, train=True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = datasets.FashionMNIST("~/.pytorch/F_MNIST_data/", download=True, train=False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

model = Network()
criterion = nn.NLLLoss()
oprimizer = optim.Adam(model.parameters(), lr = 0.0003)

epochs = 30
steps = 0
train_losses, test_losses = [],[]
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        oprimizer.zero_grad()
        loss.backward()
        oprimizer.step()
        running_loss += loss.item()

    else:
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            for images, labels in testloader:
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))
        print(e)
        print(running_loss/len(trainloader))
        print(test_loss/len(testloader))
        print(accuracy/len(testloader),"%")


%matplotlib inline
%config InlineBackend.figure_format = 'retina'

plt.plot(train_losses, label="Training loss")
plt.plot(test_losses, label="Validation loss")
plt.legend(frameon=False)
print(model.state_dict().keys())
torch.save(model.state_dict(), "checkpoint.pth")
state_dict = torch.load("checkpoint.pth")
print(state_dict.keys())

checkpoint = {"input_size":784,
              "output_size":10,
              "hidden_layers":[each.out_features for each in model.hidden_layers],
              "state_dict":model.state_dict()}
torch.save(checkpoint, "checkpoint.pth")

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = fc_model.