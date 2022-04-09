from ctypes import util
from models.ResNet import ResNet18
from models.utils import train
import utils
import torch
import torch.utils.data.dataset as dataset 
from torch.utils.data import DataLoader


device = torch.device('cuda')
resnet = ResNet18().to(device)
aes_data = utils.DataSet(0, utils.getFeature("/root/autodl-tmp/cipher/aes", utils.bitcount, 224))
des3_data = utils.DataSet(1, utils.getFeature("/root/autodl-tmp/cipher/des3", utils.bitcount, 224))
trainDataset = dataset.ConcatDataset([aes_data, des3_data])
dataloader = DataLoader(trainDataset, batch_size=512, shuffle=True)

train(resnet, dataloader, epoch=10, modelPath="/root/trainedModel/resnet18.model")