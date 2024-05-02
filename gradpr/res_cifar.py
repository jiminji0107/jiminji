import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.optim as optim
from torchvision.models.resnet import resnet18
import numpy as np
import matplotlib.pyplot as plt

class residual(nn.Module):
    def __init__(self, in_chan, out, stride = 1):
        super(residual, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(in_chan),
            nn.ReLU(),
            nn.Conv2d(in_chan, out, 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out)
        )

        if stride != 1 or in_chan != out:
            self.shortcut = nn.Conv2d(in_chan, out, 1, stride = stride)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        out = self.block(x)
        output = F.relu(out + self.shortcut(x))
        return output
    
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            residual(16, 16),
            residual(16, 16),
            residual(16, 16),
            residual(16, 16),
            residual(16, 32, stride = 2),
            residual(32, 32),
            residual(32, 32),
            residual(32, 32),
            residual(32, 32),
            residual(32, 64, stride = 2),
            residual(64, 64),
            residual(64, 64),
            residual(64, 64),
            residual(64, 64),
            residual(64, 64),
            nn.AvgPool2d(kernel_size = 8, stride = 1),
            nn.Flatten(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
model = NeuralNetwork().to("mps")

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform = ToTensor()) 
#indices = torch.randperm(len(train_dataset))
#subset_train_dataset = Subset(train_dataset, indices)
train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True, pin_memory = False)


test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform = ToTensor())
#indice = torch.randperm(len(test_dataset))
#subset_test_dataset = Subset(test_dataset, indice)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, pin_memory = False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9, weight_decay = 0.0001)

def train(model, dataset, criterion, optimizer):
    model.train()
    for epoch in range(5):
        running_loss = 0.0
        count = 0
        for image, label in dataset:
            count += 1
            image = image.to("mps")
            label = label.to("mps")
            optimizer.zero_grad()
            predict = model(image)
            loss = criterion(predict, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch : {epoch+1}, Loss : {running_loss / len(dataset)}")

def eval(model, dataset):
    model.eval()
    total = 0
    correct = 0
    for image, labels in dataset:
        image = image.to("mps")
        labels = labels.to("mps")
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(f"Total : {total}, Correct : {correct}")

print("Train start")
train(model, train_loader, criterion, optimizer)
print("Test start")
eval(model, test_loader)

