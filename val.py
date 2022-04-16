from numpy import concatenate
import models.ResNet
import torch
from preprocess import DataSet_csv
import os 
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import random
from models.utils import batchVal


# torch.multiprocessing.set_start_method('spawn')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = torch.load('/root/trainedmodel/resnet18_AES_DES3_ECB_full_32.model').to(device)
feature_dir_aes = "/root/autodl-tmp/feature/feature_pieces/aes_full"
feature_dir_des3 = "/root/autodl-tmp/feature/feature_pieces/des_full"
feature_dir_aes_arr = [feature_dir_aes + '/' + path for path in os.listdir(feature_dir_aes)]
feature_dir_des3_arr = [feature_dir_des3 + '/' + path for path in os.listdir(feature_dir_des3)]

dataset_aes = DataSet_csv(0, feature_dir_aes_arr[23])
dataset_des = DataSet_csv(1, feature_dir_des3_arr[23])
iterDataloader_aes = iter(DataLoader(dataset_aes, 32, False))
iterDataloader_des = iter(DataLoader(dataset_des, 32, False))

sum = 0
corrcet_sum = 0
for i in range(2 * dataset_aes.__len__()):
    try:
        if random.randint(0, 1):
            correct = batchVal(resnet, next(iterDataloader_aes))
            print(correct)
            sum += 1
            corrcet_sum += correct
        else:
            correct = batchVal(resnet, next(iterDataloader_des))
            print(correct)
            sum += 1
            corrcet_sum += correct
    except: pass

print("sum : ", sum)
print("score: ", corrcet_sum / sum)
