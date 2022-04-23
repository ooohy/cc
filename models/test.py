import torch
import LSTM

c0 = torch.randn(4, 384, 128) 
h0 = torch.randn(4, 384, 128) #(num_layers,batch,output_size)

net = LSTM.LSTM_CNN(c0, h0)
a = torch.rand(256, 8, 128)
# dataset = LSTM.DataSet(label=1, size=1, num_workers=1, cuda=False)
# print("hello world")
net(a)