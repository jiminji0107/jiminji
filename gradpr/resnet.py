import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = (
    'cuda'
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class Tempmodel(nn.Module):
    def __init__(self):
        super(Tempmodel, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Conv2d(64, 256, 1, 2),
            nn.Flatten(),
            nn.Softmax()
        )

    def forward(self, x):
        out = self.block(x)
        o = self.conv1(out)
        return o
    
model = Tempmodel()

"""class residualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(residualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size = 1, stride = 1),
            nn.BatchNorm2d(out_channels)
        )

        # If stride is not 1 or in/out channels mismatch, use 1x1 convolution to match dimensions
        if stride != 1 or in_channels != out_channels*4:
            self.shortcut = nn.Conv2d(in_channels, out_channels*4, kernel_size=1, stride=stride, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        out += self.shortcut(residual)  # Residual connection
        out = self.relu(out)
        return out


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 0),
            nn.MaxPool2d(3, stride = 2),
            residualBlock(64, 64),
            residualBlock(256, 64)
        )

    def forward(self, x):
        out = self.layer(x)
        return out"""


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform = transform)
indices = torch.randperm(len(train_dataset))[:100]
subset_train_dataset = Subset(train_dataset, indices)
train_loader = DataLoader(subset_train_dataset, batch_size=16, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform = transform)
indic = torch.randperm(len(test_dataset))[:100]
subset_test_dataset = Subset(test_dataset, indic)
test_loader = DataLoader(subset_test_dataset, batch_size=16, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, dataloader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            #print(f"loss : {loss}")
            loss.backward()
            
            #for name, param in model.named_parameters():
            #    plt.plot([param.grad])
            #    plt.show()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"total : {total}, correct : {correct}")

print("Train start")
train(model, train_loader, criterion, optimizer, num_epochs=5)
print("Train end / Eval start")
evaluate(model, test_loader)