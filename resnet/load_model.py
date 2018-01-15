import torch
import torch.nn as nn
from torch.autograd import Variable
from math import sin
import numpy as np

# test_data = torch.FloatTensor([2, 3])
# # 保存数据
# torch.save(test_data, "test_data.pkl")
#
# print(test_data)
# # 提取数据
# print(torch.load("test_data.pkl"))

# print(torch.load('resnet2.pkl'))

net = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)
torch.save(net.state_dict(), "net_params.pkl")  # 保存参数
torch.save(net, "net.pkl")  # 保存网络

N = 1000


def generate_data(a1, b1, c1, d1):
    a, b, c, d = [], [], [], []
    a.append(a1)
    b.append(b1)
    c.append(c1)
    d.append(d1)
    for i in range(1, N):
        a.append(a[i - 1] + 0.1)
        b.append(b[i - 1] + 0.1)
        c.append(c[i - 1] + 0.1)
        d.append(d[i - 1] + 0.1)

    y = []
    for i in range(N):
        y.append(sin(a[i]) * sin(b[i]) * sin(c[i]) * sin(d[i]))
        # 使用round寻找最近的数
        # 当距离相等时，返回整数偶数，如果是小数点后的数字，处理很复杂不深究
        y[i] = round(y[i], 1) * 10

    # 重构abcdy
    a = np.array(a).reshape([-1, 1])
    b = np.array(b).reshape([-1, 1])
    c = np.array(c).reshape([-1, 1])
    d = np.array(d).reshape([-1, 1])
    y = np.array(y).reshape([-1])

    a = Variable(torch.FloatTensor(a))
    b = Variable(torch.FloatTensor(b))
    c = Variable(torch.FloatTensor(c))
    d = Variable(torch.FloatTensor(d))
    x = torch.cat((a, b, c, d), dim=1)
    # 把x变成四维张量，batch大小为10000
    x = x.view(N, 2, 2)
    x = x.unsqueeze(1)

    y2 = torch.LongTensor(y)

    y = Variable(torch.LongTensor(y))
    return x, y, y2


# print(torch.load('net_params.pkl'))

# 2*2 Convolution采用小卷积，步长为2时卷积大小可以一直不变
def conv2x2(in_channels, out_channels, stride=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=2,
                     stride=stride, padding=1, bias=False)


# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv2x2(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层通道数不变
        self.conv2 = conv2x2(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        # print("conv1前：",residual.size())
        out = self.conv1(x)
        # print("conv1后：",out.size())
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # print("conv2后：",out.size())
        out = self.bn2(out)
        # 可以决定是否下采样，这里统一不进行下采样。
        if self.downsample:
            residual = self.downsample(x)
        # f(x)+x
        # print(out.size())
        # print(residual.size())
        out += residual
        out = self.relu(out)
        return out


# ResNet Module定义残差网络，精度不高而且时间所限，我们深度搞得不要太深
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv2x2(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0])
        self.layer3 = self.make_layer(block, 64, layers[1])
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(256, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=2):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            # 当步长不为1或者不是从图像到第一个层的时候，进行下采样。
            downsample = nn.Sequential(
                conv2x2(self.in_channels, out_channels, stride=stride),
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
        # print("forward:out", out.size())
        # print("forward:x",x.size())
        out = self.layer1(out)
        # print("forward:out2", out.size())
        out = self.layer2(out)
        out = self.layer3(out)
        # 这里不要做平均池化
        # out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        # print("全连接前", out.size())
        out = self.fc(out)
        return out


resnet = ResNet(ResidualBlock, [2, 2]).cuda()
resnet.load_state_dict(torch.load('resnet2.pkl'))

# Test
correct = 0
total = 0
for i in range(50):
    x2, y, y2 = generate_data(np.random.random(),
                              np.random.random(),
                              np.random.random(),
                              np.random.random())
    x2 = x2.cuda()

    outputs = resnet(x2)
    _, predicted = torch.max(outputs.data, 1)
    # print(predicted)
    total += y2.size(0)
    # print(y2.cuda().type)
    # print(predicted.type)
    correct += (predicted == y2.cuda()).sum()

print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))
