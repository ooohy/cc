from ctypes import util
from models.ResNet import ResNet18
from models.utils import train
import preprocess
import torch
import torch.utils.data.dataset as dataset 
from torch.utils.data import DataLoader
import time


cipherDir_aes = "/Users/daisy/Downloads/cipher_aes"
cipherDir_des = "/Users/daisy/Downloads/cipher_des"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = ResNet18().to(device)

t1 = time.time()
aes_data = preprocess.DataSet(0, preprocess.getFeature(cipherDir_aes, preprocess.bitcount, 224))
des3_data = preprocess.DataSet(1, preprocess.getFeature(cipherDir_des, preprocess.bitcount, 224))

print(time.time() - t1)
trainDataset = dataset.ConcatDataset([aes_data, des3_data])
# dataloader = DataLoader(trainDataset, batch_size=512, shuffle=True)
dataloader = DataLoader(trainDataset, batch_size=2, shuffle=True)

train(resnet, dataloader, epoch=2, modelPath="/Users/daisy/Downloads/resnet18.model")