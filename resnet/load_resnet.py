import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5  # 不做这个会出错。我也不知道为什么
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 这个转置可以去除batch的影响（个人理解）
    plt.show()


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'transforms.Compose就是把各种transform组合起来'
# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize(40),  # 最小边的长变为40，尺度缩放
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 Dataset
train_dataset = dsets.CIFAR10(root='./data/',
                              train=True,
                              transform=transform,
                              download=True)

test_dataset = dsets.CIFAR10(root='./data/',
                             train=False,
                             transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)


# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层是不变卷积
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        # print("conv1前：",residual.size())
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 可以决定是否下采样
        if self.downsample:
            residual = self.downsample(x)
        # f(x)+x
        # print(out.size())
        # print(residual.size())
        out += residual
        out = self.relu(out)
        return out


# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            # 当步长不为1或者不是从图像到第一个层的时候，进行下采样。
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels  # 一层的输入变成了下一层的输出
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        # print("forward:out2", out.size())
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        # print("全连接前", out.size())
        out = self.fc(out)
        return out


# 每层2个Block。总共三层。这里分层主要是为了区分是否下采样，当然还可以分得更细
resnet = ResNet(ResidualBlock, [2, 2]).cuda()
resnet.load_state_dict(torch.load('resnet.pkl'))

# Test
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.cuda())
    outputs = resnet(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    # print(labels.cuda())
    # print(predicted)
    correct += (predicted == labels.cuda()).sum()

print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))

# 随机来几张图片
dataiter = iter(test_loader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(100)))

# 用模型预测以下结果
outputs = resnet(Variable(images.cuda()))
_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(100)))
