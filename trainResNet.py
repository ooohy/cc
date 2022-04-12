from models.ResNet import ResNet18
from models.utils import train
import preprocess
import torch
import torch.utils.data.dataset as dataset 
from torch.utils.data import DataLoader
import time
import ray
import json
import joblib
from preprocess import DataSet


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    # print train start time
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    resnet = ResNet18().to(device) 

    
    featureDir_dataloader_mini_pytorch = "/root/autodl-tmp/feature/feature_dataloader_mini"
    # dataloader = DataLoader(trainDataset, batch_size=2, shuffle=True, num_workers=5)
    dataloader = torch.load(featureDir_dataloader_mini_pytorch)

    train(resnet, dataloader, epoch=10, modelPath="/root/trainedmodel/resnet18_AES_TDES_ECB.model")
    print("finish train at:", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))