import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义make_layer
def make_layer18(in_channel, out_channel, block_num, stride=1):
    shortcut = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1, stride),  # 尺寸不变
        nn.BatchNorm2d(out_channel)
    )
    layers = list()
    layers.append(ResBlock18(in_channel, out_channel, stride, shortcut))

    for i in range(1, block_num):
        layers.append(ResBlock18(out_channel, out_channel))
    return nn.Sequential(*layers)


def make_layer50(in_channel, out_channel, block_num, stride=1):
    shortcut = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1, stride),  # 尺寸不变
        nn.BatchNorm2d(out_channel)
    )
    layers = list()
    layers.append(ResBlock50(in_channel, out_channel, stride, shortcut))

    for i in range(1, block_num):
        layers.append(ResBlock50(out_channel, out_channel))
    return nn.Sequential(*layers)


# ResBlock 2 conv with the same size output
class ResBlock18(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResBlock18, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class ResBlock50(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResBlock50, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride, 1, bias=False),
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

# 堆叠Resnet，见上表所示结构
class ResNet18(nn.Module):
    """
    implement of ResNet thanks to https://www.ji
    anshu.com/p/972e2c5e6871  \n
    https://zhuanlan.zhihu.com/p/42706477
    """

    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        self.pre = nn.Sequential(  # 第一层
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # nn.MaxPool2d(3, 2, 1)
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer1 = make_layer18(64, 64, 2, 2)  # 7*7
        self.layer2 = make_layer18(64, 128, 2, 2)  #
        self.layer3 = make_layer18(128, 256, 2, 2)  # stride = 2
        self.layer4 = make_layer18(256, 512, 2)  # stride = 2
        # self.avg = nn.AvgPool2d(4)  # 平均化
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(512, num_classes)  # 全连接

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out
class ResNet18(nn.Module):
    """
    implement of ResNet thanks to https://www.ji
    anshu.com/p/972e2c5e6871  \n
    https://zhuanlan.zhihu.com/p/42706477
    """

    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        self.pre = nn.Sequential(  # 第一层
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # nn.MaxPool2d(3, 2, 1)
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer1 = make_layer18(64, 64, 2, 2)  # 7*7
        self.layer2 = make_layer18(64, 128, 2, 2)  #
        self.layer3 = make_layer18(128, 256, 2, 2)  # stride = 2
        self.layer4 = make_layer18(256, 512, 2)  # stride = 2
        # self.avg = nn.AvgPool2d(4)  # 平均化
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(512, num_classes)  # 全连接

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

class ResNet50(nn.Module):
    def __init__(self, num_classes=2, input_channels=1):
        super(ResNet18, self).__init__()
        self.pre = nn.Sequential(  # 第一层
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # nn.MaxPool2d(3, 2, 1)
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer1 = make_layer18(64, 64, 2, 2)  # 7*7
        self.layer2 = make_layer18(64, 128, 2, 2)  #
        self.layer3 = make_layer18(128, 256, 2, 2)  # stride = 2
        self.layer4 = make_layer18(256, 512, 2)  # stride = 2
        # self.avg = nn.AvgPool2d(4)  # 平均化
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(512, num_classes)  # 全连接

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

def trainResNet(net, dataloader, epoch=5):
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=0.05, momentum=0.8)  # momentum小一点

    for round in range(epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (round, i + 1, running_loss / 2000))
                running_loss = 0.0
