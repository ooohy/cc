from numpy import concatenate
import models.ResNet
import torch
from preprocess import DataSet_np
import os 
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import random
from models.utils import batchVal


# torch.multiprocessing.set_start_method('spawn')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = torch.load('/root/trainedmodel/resnet18_AES_DES3_ECB_mini_256_in8.model').to(device)
# feature_dir_aes = "/root/autodl-tmp/feature/feature_pieces/aes_full"
# feature_dir_des3 = "/root/autodl-tmp/feature/feature_pieces/des_full"
# feature_dir_aes = "/root/autodl-tmp/feature/aes_rand_256_in8"
# feature_dir_des3 = "/root/autodl-tmp/feature/des3_rand_256_in8"
feature_dir_aes = "/root/autodl-tmp/feature/aes_rand_256_in8"
feature_dir_des3 = "/root/autodl-tmp/feature/des3_rand_256_in8"
feature_dir_aes_arr = [feature_dir_aes + '/' + path for path in os.listdir(feature_dir_aes)][0:10000]
feature_dir_des3_arr = [feature_dir_des3 + '/' + path for path in os.listdir(feature_dir_des3)][0:10000]

dataset_aes = DataSet_np(0, feature_dir_aes_arr, in_channel=8)
dataset_des = DataSet_np(1, feature_dir_des3_arr, in_channel=8)
iterDataloader_aes = iter(DataLoader(dataset_aes, 128, False))
iterDataloader_des = iter(DataLoader(dataset_des, 128, False))

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
