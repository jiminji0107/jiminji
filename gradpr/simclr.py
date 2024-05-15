import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
import torch.optim as optim
from torchvision.models.resnet import resnet18
import matplotlib.pyplot as plt
#from pytorch_metric_learning.losses import NTXentLoss
from PIL import Image
#from torchlars import LARS
import matplotlib.pyplot as plt

class residual(nn.Module):
    def __init__(self, in_chan, out, stride = 1):
        super(residual, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, 3, stride = stride, padding = 1),
            nn.BatchNorm2d(in_chan),
            nn.ReLU(),
            nn.Conv2d(in_chan, out, 3, stride = 1, padding = 1),
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
    def __init__(self, mode):
        super(NeuralNetwork, self).__init__()
        self.neural = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            residual(16, 16),
            residual(16, 16),
            residual(16, 16),
            residual(16, 16),
            residual(16, 32),
            residual(32, 32, stride = 2),
            residual(32, 32),
            residual(32, 32),
            residual(32, 32),
            residual(32, 64),
            residual(64, 64, stride = 2),
            residual(64, 64),
            residual(64, 64),
            residual(64, 64),
            residual(64, 64),
            nn.AvgPool2d(kernel_size = 8, stride = 1),
            nn.Flatten()
        )
        self.classification = nn.Linear(64,10)

        self.projection = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

        self.mode = mode

    def forward(self, x):
        x = self.neural(x)
        if self.mode == "train":
            x = self.projection(x)
            return x
        elif self.mode == "train_classification" or self.mode == "test":
            x = self.classification(x)
            return x
    
"""class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 128)
        )

    def forward(self, x):
        x = self.projection(x)
        return x"""


class SimClrModel(nn.Module):
    def __init__(self, mode):
        super(SimClrModel, self).__init__()
        self.encoder = NeuralNetwork(mode)
        #self.projection = Mlp()
        self.mode = mode

    def forward(self, x1, x2 = None):
        if self.mode == "train":
            x1 = self.encoder(x1)
            #x1 = self.projection(x1)
            x2 = self.encoder(x2)
            #x2 = self.projection(x2)
            return x1, x2
        
        else:
            x1 = self.encoder(x1)
            #x2 = self.encoder(x2)
            return x1 

#바꿔야할듯    
model = SimClrModel("train").to("mps")

def color_distortion(s = 0.5):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p = 0.8)
    rnd_gray = transforms.RandomGrayscale(p = 0.2)
    color_distort = nn.Sequential(
        rnd_color_jitter,
        rnd_gray
    )
    return color_distort

transform = transforms.Compose([
    transforms.RandomResizedCrop((32, 32), scale = (0.08, 1.0), ratio = (0.75, 1.3333)),
    color_distortion(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

class CustomDataset(Dataset):
    def __init__(self, root, train = True, transform=None, download = False):
        self.image = CIFAR10(root = root, train = train, download = download)  # 이미지 파일 경로 리스트를 초기화
        self.transform = transform      # 변환을 적용할 transform을 초기화

    def __getitem__(self, index):
        image, _ = self.image[index]

        if self.transform:
            image1 = self.transform(image)    # 첫 번째 변환된 이미지를 생성
            image2 = self.transform(image)    # 두 번째 변환된 이미지를 생성
        
        return image1, image2  # 두 개의 변환된 이미지를 반환

    def __len__(self):
        return len(self.image)  # 데이터셋의 전체 길이를 반환
    
class lossfunc(nn.Module):
    def __init__(self, batch_size, temperature=0.1):
        super(lossfunc, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

    def forward(self, out1, out2):
        total = torch.cat((out1, out2), dim=0)
        total = F.normalize(total, p=2, dim=1)

        similarity_matrix = torch.mm(total, total.T) / self.temperature
        mask = torch.eye(2 * self.batch_size, dtype=torch.bool).to(similarity_matrix.device)
        similarity_matrix.masked_fill_(mask, float('-1e9'))

        log_prob = -F.log_softmax(similarity_matrix, dim=1)

        print(log_prob.shape)

        loss = 0
        #바꿔라 이부분
        for k in range(self.batch_size):
            loss += log_prob[k, self.batch_size + k] + log_prob[self.batch_size + k, k]
        return loss / 2*self.batch_size

train_dataset = CustomDataset(root = './data', train = True, transform = transform, download = True)
num_train = len(train_dataset)
num_val = int(num_train * 0.1)
num_train -= num_val

train_dataset, val_dataset = random_split(train_dataset, [num_train, num_val])
#indices = torch.randperm(len(train_dataset))[:1000]
#subset_train_dataset = Subset(train_dataset, indices)

train_loader = DataLoader(train_dataset, batch_size=3000, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = 1000)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform = test_transform)
test_loader = DataLoader(test_dataset, batch_size=3000, shuffle=False)

criterion = lossfunc(3000)
#optimizer = LARS(optim.SGD(model.parameters(), lr = 4.8))
optimizer = optim.Adam(model.parameters(), lr = 0.001)

loss_val = []

def train(model, dataset, criterion, optimizer, num_epoch):
    model.train()
    for epoch in range(num_epoch):
        running_loss = 0.0
        for batch in dataset:
            #image = image.to("mps")
            image1, image2 = batch
            image1, image2 = image1.to("mps"), image2.to("mps")
            
            optimizer.zero_grad()
            out1, out2 = model(image1, image2)
            loss = criterion(out1, out2)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss_val.append(loss.item())
        print(f"Epoch : {epoch+1}, Loss : {running_loss / len(dataset)}")

print("Train start")
train(model, train_loader, criterion, optimizer, 10)
#plt.plot(loss_val)
#plt.xlabel('batch num')
#plt.ylabel('loss')
#plt.show()
print("Train end")
torch.save(model.encoder.state_dict(), "pretrained_simclr_model.pth")
model = SimClrModel("train_classification")
model.encoder.load_state_dict(torch.load("pretrained_simclr_model.pth"))

for param in model.encoder.neural.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(model.encoder.classification.parameters(), lr=0.001)

train_cla_dataset = CIFAR10(root = "./data", download = True, transform = test_transform, train = True)
indic = torch.randperm(len(train_cla_dataset))
sub = Subset(train_cla_dataset, indic)
train_cla_loader = DataLoader(train_cla_dataset, batch_size = 2000)

criterion = nn.CrossEntropyLoss()

def train_clas(model, dataset, criterion, optimizer, num_epoch):
    model.train()
    for epoch in range(num_epoch):
        running_loss = 0
        total = 0
        correct = 0
        for image, label in dataset:
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predict = output.max(1)
            total += label.size(0)
            correct += (predict == label).sum().item()
        print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {running_loss/len(train_loader)}, Accuracy: {100.*correct/total}")


def eval(model, dataset):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for image, labels in dataset:
            #image = image.to("mps")
            #labels = labels.to("mps")
            output1 = model(image)
            _, predicted = torch.max(output1.data, 1)
            _, predicted 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(f"Total : {total}, Correct : {correct}")


print("Class start")
train_clas(model, train_cla_loader, criterion, optimizer, 10)
print("Test start")
eval(model, test_loader)

