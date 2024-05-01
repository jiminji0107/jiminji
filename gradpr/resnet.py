import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch.optim as optim
from torchvision.models.resnet import resnet18
import numpy as np
import matplotlib.pyplot as plt

global outputf

device = (
    'cuda'
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class residualBlock(nn.Module):
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
            nn.BatchNorm2d(out_channels * 4)
        )

        if stride != 1 or in_channels != out_channels*4:
            self.shortcut = nn.Conv2d(in_channels, out_channels*4, kernel_size=1, stride=stride, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        x1 = self.block(x)
        x = F.relu(x1 + self.shortcut(x))
        return x
    

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride = 2),
            residualBlock(64, 64),
            residualBlock(256, 64),
            residualBlock(256, 64, stride = 2),
            residualBlock(256, 128),
            residualBlock(512, 128),
            residualBlock(512, 128),
            residualBlock(512, 128, stride = 2),
            residualBlock(512, 256),
            residualBlock(1024, 256),
            residualBlock(1024, 256),
            residualBlock(1024, 256),
            residualBlock(1024, 256),
            residualBlock(1024, 256, stride = 2),
            residualBlock(1024, 512),
            residualBlock(2048, 512),
            residualBlock(2048, 512),
            nn.AvgPool2d(kernel_size = 2, stride = 1),
            nn.Linear(6, 1000),
            nn.Flatten()
        )

    def forward(self, x):
        out = self.layer(x)
        return out
    
model = NeuralNetwork().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform = transform) 
indices = torch.randperm(len(train_dataset))[:100]
subset_train_dataset = Subset(train_dataset, indices)
train_loader = DataLoader(subset_train_dataset, batch_size=16, shuffle=True, pin_memory = False)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform = transform)
indice = torch.randperm(len(test_dataset))[:100]
subset_test_dataset = Subset(test_dataset, indice)
test_loader = DataLoader(subset_test_dataset, batch_size=16, shuffle=False, pin_memory = False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
outputf = 'param.txt'
#with open(outputf, 'w') as f:    
#    for name, param in model.named_parameters():
#        f.write(f"name : {name}, param : {param}")
#    f.write("Writing gradient\n\n\n")

def train(model, dataloader, criterion, optimizer, num_epochs=5):
    global outputf
    model.train()
    for epoch in range(num_epochs):
        with open(outputf, 'w') as f:
            running_loss = 0.0
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                #print(f"loss : {loss}")
                loss.backward()
                
                for name, param in model.named_parameters():
                    f.write(f"name : {name} grad : {param.grad.norm(2)}\n")
                optimizer.step()
                running_loss += loss.item()
        print(loss.grad_fn)    
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"total : {total}, correct : {correct}")

print("Train start")
train(model, train_loader, criterion, optimizer, num_epochs=5)
print("Train end / Eval start")
evaluate(model, test_loader)