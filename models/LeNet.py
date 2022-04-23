'''
File: LeNet.py
Project: Models
File Created: Thursday, 11th November 2021 5:00:44 pm
Author: ysy (1424535213@qq.com)
-----
Last Modified: Thursday, 11th November 2021 5:00:58 pm
Modified By: ysy (1424535213@qq.com>)
-----
May force be with you !
'''

from re import L
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn.modules.loss import MSELoss
from torch.utils.data import DataLoader
import torch.optim as optim
import sys



# lenet using max_pool2d
class LeNet(nn.Module) :
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        # self.fc3 = nn.Linear(84, 10)


    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size = 2, stride = 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size = 2, stride = 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

# class LeNet_lstm(nn.Module) :
#     def __init__(self):
#         super(LeNet_lstm, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.conv3 = nn.Conv2d(16, 120, 5)
#         self.fc1 = nn.Linear(8 * 59536, 3000)
#         self.fc2 = nn.Linear(3000, 120)
#         self.fc3 = nn.Linear(120, 84)
#         self.fc4 = nn.Linear(84, 2)
#         # self.fc3 = nn.Linear(84, 10)


#     def forward(self, x):

#         x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size = 2, stride = 2)
#         x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size = 2, stride = 2)
#         x = x.view(x.size()[0], -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x
# train_LeNet, optimizer MSE

def trainLeNet(lenet, dataloader, epoch = 5) :
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(lenet.parameters(), lr=0.01, momentum=0.4)

    for round in range(epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = torch.unsqueeze(inputs, 1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = lenet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print(f'[{round + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

def testLeNet(lenet, dataloader) :
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            matrix, labels = data
            matrix = torch.unsqueeze(matrix, 1)
            outputs = lenet(matrix)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    
if __name__ == '__main__':
    cnn = LeNet_lstm()
    a = torch.rand(8, 1, 256, 256)
    b = cnn(a)
    print(b.shape)