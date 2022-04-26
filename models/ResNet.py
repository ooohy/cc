from toml import TomlArraySeparatorEncoder
import torch
import torch.nn as nn
import torch
import bitarray
from bitarray.util import urandom
import numpy as np
import sys
sys.path.append("/root/cc/")
# sys.path.append("/Users/daisy/CipherClassification/cc")
from cipher import aes
from cipher import des
from scipy.fftpack import fft
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from AlexNet import AlexNet_lstm
# from LeNet import lenet_lstm
from torch.utils.data import ConcatDataset
import time
import gc
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import utils


# 定义make_layer
def make_layer18(in_channel, out_channel, block_num, stride=1):
    shortcut = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1, stride),  # 尺寸不变
        nn.BatchNorm2d(out_channel)
    )
   
   
   
    for i in range(1, block_num):
        layers.append(ResBlock18(out_channel, out_channel))
    return nn.Sequential(*layers)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def make_layer101(in_channel, mid, out_channel, block_num, stride=1):
    shortcut = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1, stride),  # 尺寸不变
        nn.BatchNorm2d(out_channel)
    )
    layers = list()
    layers.append(ResBlock101(in_channel, out_channel, stride, shortcut))

    for i in range(1, block_num):
        layers.append(ResBlock101(out_channel, out_channel))
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

class ResBlock101(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResBlock101, self).__init__()
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

    def __init__(self, num_classes=2, in_channel=1):
        super(ResNet18, self).__init__()
        self.pre = nn.Sequential(  # 第一层
            nn.Conv2d(in_channel, 64, kernel_size=7, stride=1, padding=3, bias=False),
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


class ResNet101(nn.Module):
    def __init__(self, num_classes=2, input_channels=8):
        super(ResNet18, self).__init__()
        self.pre = nn.Sequential(  # 第一层
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # nn.MaxPool2d(3, 2, 1)
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer1 = make_layer101(64, 256, 3, 2)  # 7*7
        self.layer2 = make_layer101(256, 512, 4, 2)  #
        self.layer3 = make_layer101(512, 1024, 23, 2)  # stride = 2
        self.layer4 = make_layer101(1024, 2048, 3)  # stride = 2
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(2048, num_classes)  # 全连接

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

class DataSet(Dataset):
    def __init__(self, label, size=24, num_workers=24):
        super(DataSet, self).__init__()
        self.num_workers = num_workers
        self.label = label

        if label == 1:
            self.encryptor = aes.AES_ECB()
        elif label == 0:
            self.encryptor = des.TDES_ECB()

        self.mat = self.get_mat_batch(size)

    def __getitem__(self, index):
        if torch.cuda.is_available():
            return self.mat[index].cuda(), torch.tensor(self.label).cuda()
        else:
            return self.mat[index], torch.tensor(self.label)
    
    def __len__(self):
        return self.mat.shape[0]


    def process(self, batch_size, in_channel, bitwise, matrix_size):
        ba = bitarray.bitarray()
        total_len = batch_size * in_channel * bitwise * matrix_size**2
        ba.frombytes(self.encryptor.encrypt(urandom(total_len).tobytes()))
        ba = ba[:total_len]
        ba_count = [ba[p: p+bitwise].count(1) for p in range(0, total_len, bitwise)]
        cipher_np = np.array(ba_count, dtype=np.float32).reshape(batch_size, in_channel, matrix_size, matrix_size)

        return torch.from_numpy(cipher_np)
    
    def get_mat_batch(self, size):
        cipher_pt_arr = Parallel(n_jobs=self.num_workers)(delayed(self.process)(384, 8, 8, 256) for i in range(size))
        # cipher_pt_arr = []
        # for i in range(size):
        #     cipher_pt_arr.append(self.process(256, 8, 8, 256))
        return torch.cat(cipher_pt_arr, 0)

class DataSet_complex(Dataset):
    def __init__(self, label, size=24, num_workers=24):
        super(DataSet, self).__init__()
        self.num_workers = num_workers
        self.label = label

        if label == 1:
            self.encryptor = aes.AES_ECB()
        elif label == 0:
            self.encryptor = des.TDES_ECB()

        self.mat = self.get_mat_batch(size)

    def __getitem__(self, index):
        if torch.cuda.is_available():
            return self.mat[index].cuda(), torch.tensor(self.label).cuda()
        else:
            return self.mat[index], torch.tensor(self.label)
    
    def __len__(self):
        return self.mat.shape[0]


    def process(self, batch_size, bitwise, matrix_size):
        ba = bitarray.bitarray()
        total_len = batch_size * bitwise * matrix_size**2
        ba.frombytes(self.encryptor.encrypt(urandom(total_len).tobytes()))
        cipher_complex_np_arr = [fft(np.array(ba[p: p+bitwise])) for p in range(0, total_len, bitwise)]
        cipher_complex_np = np.stack(cipher_complex_np_arr, axis=0) 
        cipher_pt = torch.from_numpy(cipher_complex_np).transpose(2, 1).reshape() # (batch_size, bitwise, 2)
        # cipher_complex_np = fft(np.array(ba[:total_len].tolist()))
        # ba_count = [ba[p: p+bitwise].count(1) for p in range(0, total_len, bitwise)]
        cipher_np = np.array(ba_count, dtype=np.float32).reshape(batch_size, bitwise, matrix_size, matrix_size)

        return torch.from_numpy(cipher_np)
    
    def get_mat_batch(self, size):
        cipher_pt_arr = Parallel(n_jobs=self.num_workers)(delayed(self.process)(384, 8, 8, 256) for i in range(size))
        # cipher_pt_arr = []
        # for i in range(size):
        #     cipher_pt_arr.append(self.process(256, 8, 8, 256))
        return torch.cat(cipher_pt_arr, 0)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()
    
    resnet = ResNet18(in_channel=8).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adadelta(resnet.parameters())

    counter = 0
    # dataloader = DataLoader(dataset, 16, shuffle=True, num_workers=8)
    for epoch in range(40):
        dataset_des = DataSet(0, size=20, num_workers=20)
        dataset_aes = DataSet(1, size=20, num_workers=20)
        dataset = ConcatDataset([dataset_aes, dataset_des])
        dataloader = DataLoader(dataset, 32, True, num_workers=8)

        running_loss = 0
        for i, data in enumerate(dataloader, 0):
            resnet.zero_grad()
            inputs, labels = data
            # print("embedding.shape", embedding.shape)
            out = resnet(inputs)
            predict = torch.softmax(out, dim=1)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()

            # print("out.shape", out.shape)
            running_loss += loss.item()
            if i % 50 == 49:
                print('[%d, %5d] loss: %.3f' %
                    (epoch, i + 1, running_loss / 50))
                print("  >>  ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
                writer.add_scalar('Loss/train', running_loss / 50, counter)
                counter+=1
                running_loss = 0
            
    torch.save(resnet, "/root/trainedmodel/resnet18_rand_all.model")