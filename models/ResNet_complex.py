import tensorboard
import torch
import torch.nn as nn
import torch
import bitarray
from bitarray.util import urandom
import numpy as np
import sys
sys.path.append("/root/cc/")
sys.path.append("/Users/daisy/CipherClassification/cc")
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
from torchsummary import summary

from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexMaxPool2d, ComplexReLU
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d
from torch.nn import CrossEntropyLoss


class DataSet_complex(Dataset):
    def __init__(self, label, size=24, num_workers=24):
        super(DataSet_complex, self).__init__()
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


    def process(self, batch_size, bitwise, mat_size):
        ba = bitarray.bitarray()
        total_len = batch_size * bitwise * mat_size**2
        ba.frombytes(self.encryptor.encrypt(urandom(total_len).tobytes()))
        cipher_complex_np_arr = [fft(np.array(ba[p: p+bitwise].tolist(), dtype=np.float32)) for p in range(0, total_len, bitwise)]
        cipher_complex_np = np.stack(cipher_complex_np_arr, axis=0) 
        cipher_pt = torch.from_numpy(cipher_complex_np).reshape(-1, bitwise, mat_size, mat_size) # (batch_size, bitwise_in_channel)
        # cipher_complex_np = fft(np.array(ba[:total_len].tolist()))
        # ba_count = [ba[p: p+bitwise].count(1) for p in range(0, total_len, bitwise)]

        return cipher_pt
    
    def get_mat_batch(self, size):
        # cipher_pt_arr = Parallel(n_jobs=self.num_workers)(delayed(self.process)(384, 8, 32) for i in range(size))
        cipher_pt_arr = []
        for i in range(size):
            cipher_pt_arr.append(self.process(128, 8, 256))
        return torch.cat(cipher_pt_arr, 0)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = ComplexConv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                ComplexConv2d(in_channels, out_channels, kernel_size=1, stride=2),
                ComplexBatchNorm2d(out_channels)
            )
        else:
            self.conv1 = ComplexConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = ComplexConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = ComplexBatchNorm2d(out_channels)
        self.bn2 = ComplexBatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = complex_relu(self.bn1(self.conv1(input)))
        input = complex_relu(self.bn2(self.conv2(input)))
        input = input + shortcut
        return complex_relu(input)
    
class ResNet18(nn.Module):
    def __init__(self, in_channels, resblock, num_class=2):
        super().__init__()
        self.layer0 = nn.Sequential(
            ComplexConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            ComplexMaxPool2d(kernel_size=3, stride=2, padding=1),
            ComplexBatchNorm2d(64),
            ComplexReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = ComplexLinear(512, num_class)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = x.view(-1, 512)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    dataset = DataSet_complex(1, 1, 1)
    dataset_des = des
    resnet = ResNet18(8, ResBlock, num_class=2)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)
    writer = SummaryWriter()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adadelta(resnet.parameters())


    # summary(resnet, (8, 256, 256))
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = resnet(inputs)

            # transform to real
            outputs = torch.view_as_real(outputs)
            outputs = outputs.norm(p=2, dim=2)
            
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

           # print statistics
            last_loss = 0
            running_loss += loss.item()
            if i % 50 == 49:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                (epoch, i + 1, running_loss / 50))
                print("  >>  ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
                    # last_loss = running_loss
                last_loss = running_loss / 50 
                running_loss = 0.0
        print("epoch", epoch)