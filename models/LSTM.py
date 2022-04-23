from torch.autograd import Variable
import torch.nn as nn
import torch
import bitarray
from bitarray.util import urandom
import numpy as np
import sys
# sys.path.append("/root/cc/")
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
import torch.nn.functional as f


class LSTM_CNN(nn.Module):
    def __init__(self, h0, c0, seq_len=128):
        super(LSTM_CNN, self).__init__()
        self.h0 = h0
        self.c0 = c0
        self.rnn = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, bidirectional=True) #(input_size,hidden_size,num_layers)
        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.AdaptiveAvgPool1d(output_size = (1)),
            nn.Linear(256 * 256 * 8, 2),
        )
    
    def forward(self, x):
        x = f.normalize(x, dim=1)
        embedding, (hn, cn)= self.rnn(x, (self.h0, self.c0)) # (seq, batch, input)
        embedding = torch.transpose(embedding, 1, 0) # (batch, seq, input)
        # embedding = torch.unsqueeze(embedding, 1)#(batch, seq, output_size) -> (batch, seq, output_size)
        out = self.classifier(embedding)
        return out

class DataSet(Dataset):
    def __init__(self, label, size=18, num_workers=18, cuda=True):
        super(DataSet, self).__init__()
        self.num_workers = num_workers
        self.label = label
        self.cuda = cuda

        if label == 1:
            self.encryptor = aes.AES_ECB()
        elif label == 0:
            self.encryptor = des.TDES_ECB()

        self.seq = self.get_seq_batch(size)

    def __getitem__(self, index):
        if torch.cuda.is_available():
            return self.seq[index].cuda(), torch.tensor(self.label).cuda()
        else:
            return self.seq[index], torch.tensor(self.label)
    
    def __len__(self):
        return self.seq.shape[0]


    def process(self, batch_size, seq_len, input_size):
        ba = bitarray.bitarray()
        ba.frombytes(self.encryptor.encrypt(urandom(batch_size * input_size * seq_len).tobytes()))
        ba = ba[:batch_size * input_size * seq_len]
        cipher_np = np.array(ba.tolist(),dtype=np.float32)
        cipher_fft = fft(cipher_np)
        cipher_complex = torch.from_numpy(cipher_fft)
        cipher_pt = torch.view_as_real(cipher_complex).reshape(batch_size, seq_len, input_size, 2).transpose(2, 3).reshape(batch_size, seq_len, 2*input_size)
        return cipher_pt
    
    def get_seq_batch(self, size):
        cipher_pt_arr = Parallel(n_jobs=self.num_workers)(delayed(self.process)(3027, 256, 64) for i in range(size))

        # cipher_pt_arr = []
        # for i in range(size):
            # cipher_pt_arr.append(self.process(16, 64, 256))
        return torch.cat(cipher_pt_arr, 0)

if __name__ == '__main__':
    """
    rnn = nn.LSTM(input_size=10, hidden_size=20, num_layers=2,bidirectional=True) #(input_size,hidden_size,num_layers)
    input = torch.randn(5, 3, 10) #(seq_len, batch, input_size)
    h0 = torch.randn(4, 3, 20) #(num_layers,batch,output_size)
    c0 = torch.randn(4, 3, 20) #(num_layers,batch,output_size)
    output, (hn, cn) = rnn(input, (h0, c0))
    """
    torch.multiprocessing.set_start_method('spawn')
    torch.manual_seed(0)
    h0 = torch.randn(4, 384, 128) #(num_layers,batch,output_size)
    c0 = torch.randn(4, 384, 128) 

    print("flag 1")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("flag 1")
    net = LSTM_CNN(h0, c0)
    print("flag 1")
    net.to(device)
    print("flag 1")
    writer = SummaryWriter()
    print("flag 1")
    criterion = nn.CrossEntropyLoss()
    print("flag 1")
    opt = torch.optim.Adadelta(net.parameters())
    print("flag 1")

    # train LSTM
    counter = 0
    dataset_aes = DataSet(label=1, size=18, num_workers=18)
    dataset_des = DataSet(label=0, size=18, num_workers=18)
    # dataset_aes = DataSet(label=1, size=2, num_workers=6, cuda=False)
    # dataset_des = DataSet(label=0, size=2, num_workers=6, cuda=False)
    # dataset = ConcatDataset([dataset_aes, dataset_des])
    dataset = ConcatDataset([dataset_aes, dataset_des])
    # dataloader = DataLoader(dataset, 384, shuffle=True, num_workers=8)
    dataloader = DataLoader(dataset, 16, shuffle=True, num_workers=8)
    for epoch in range(40):
        running_loss = 0
        for i, data in enumerate(dataloader, 0):
            net.zero_grad()
            inputs, labels = data
            # print("embedding.shape", embedding.shape)
            inputs = torch.transpose(inputs, 1, 0)
            inputs = f.normalize(inputs, dim=1)
            out = net(inputs)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()

            # print("out.shape", out.shape)
            running_loss += loss.item()
            if i % 50 == 49:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch, i + 1, running_loss / 50))
                print("  >>  ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
                writer.add_scalar('Loss/train', running_loss / 50, counter)
                counter+=1
                running_loss = 0
            
    torch.save(net, "/root/trainedmodel/LSTM_CNN/rnn.model")