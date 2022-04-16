from models.ResNet import ResNet18
from models.utils import train
from models.utils import lazyLoadAndTrain
from preprocess import bitcount
import preprocess
from preprocess import getFeature_joblib
import torch
from preprocess import DataSet_joblib
from torch.utils.data import DataLoader
import torch.utils.data as data
import time
import ray
import json
from functools import partial
import os
# import pandas as pd
import modin.pandas as pd
from preprocess import DataSet_csv
from preprocess import DataSet
from torch.utils.tensorboard import SummaryWriter

# ray.init(
#     _system_config={
#         "object_spilling_config": json.dumps(
#             {"type": "filesystem", "params": {"directory_path": "/Users/daisy/ray"}},
#         )
#     },
# )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    # print train start time
    # print("start preprocess at:", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    # cipherDir_aes = "/Users/daisy/Downloads/cipher_aes"
    # cipherDir_des = "/Users/daisy/Downloads/cipher_des"
    # cipherDir_aes = "/root/autodl-tmp/cipher/aes"
    # cipherDir_des = "/root/autodl-tmp/cipher/des3"
    # cipherDir_aes = "/root/testdata/aes"
    # cipherDir_des = "/root/testdata/des"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train_partial = partial(train, epoch=10)
    resnet = ResNet18().to(device)
    # feature_dir_dict = dict()
    # feature_dir_aes = "/Users/daisy/Downloads/feature/aes_feature.csv"
    # feature_dir_des3 = "/Users/daisy/Downloads/feature/des3_feature"
    # feature_dir_aes = "/root/autodl-tmp/feature/feature_pieces/aes_full"
    # feature_dir_des3 = "/root/autodl-tmp/feature/feature_pieces/des_full"
    # feature_dir_dict[0] = feature_dir_aes
    # feature_dir_dict[1] = feature_dir_des3
    # dataset = torch.load("/root/autodl-tmp/feature/feature_dataset_mini")
    # dataloader = DataLoader(dataset, 128, True)
    # train(resnet, dataloader, 10, "/root/trainedmodel/resnet18.model")
    feature_dir_aes = "/root/autodl-tmp/feature/feature_pieces/aes_full"
    feature_dir_des3 = "/root/autodl-tmp/feature/feature_pieces/des_full"
    # dataset_aes = DataSet_joblib(0, feature_dir_aes)
    # dataset_des3 = DataSet_joblib(1, feature_dir_des3)
    # dataset = data.ConcatDataset([dataset_aes, dataset_des3

    # dataloader = DataLoader(dataset, 128, True)
    # train(resnet, dataloader, 8, modelPath="/root/trainedmodel/resnet18_AESTDES_ECB.py")

    # getFeature_joblib(cipherDir_aes, bitcount, 224, feature_dir_aes, num_workers=12)
    # df = pd.read_csv(feature_dir_aes)
    # print(df.shape)

    feature_dir_aes_arr = [feature_dir_aes + '/' + path for path in os.listdir(feature_dir_aes)]
    feature_dir_des3_arr = [feature_dir_des3 + '/' + path for path in os.listdir(feature_dir_des3)]
    # feature_dir_aes_arr = feature_dir_aes_arr[0:1]
    # feature_dir_des3_arr = feature_dir_des3_arr[0:1]
    # dataset_path = "/root/autodl-tmp/dataset"
    # counter = 0
    # for feature in feature_dir_aes_arr:
    #     torch.save(DataSet_csv(0, feature), dataset_path + '/' + 'aes' + "/" + "AES_" + str(counter) + ".dataset")
    #     counter += 1
    #     print(counter)
    # counter = 0
    # for feature in feature_dir_des3_arr:
    #     torch.save(DataSet_csv(0, feature), dataset_path + '/' + 'des3' + '/' + "tdes_" + str(counter) + ".dataset")
    #     counter += 1
    #     print(counter)

    # dataset_arr = []
    # for feature in feature_dir_aes_arr:
    #     dataset_arr.append(DataSet_csv(0, feature))
    # for feature in feature_dir_des3_arr:
    #     dataset_arr.append(DataSet_csv(1, feature))
    # dataset = data.ConcatDataset(dataset_arr)
    # torch.save(dataset, "AES")

    
    counter = 0
    writer = SummaryWriter()
    for round in range(5):
        for i in range(23):
            dataset_arr = []
            dataset_arr.append(DataSet_csv(0, feature_dir_aes_arr[i]))
            dataset_arr.append(DataSet_csv(1, feature_dir_des3_arr[i]))
            dataset = data.ConcatDataset(dataset_arr)
            dataloader = DataLoader(dataset, 128, True, num_workers=8)
            loss = train(resnet, dataloader, epoch=1)
            print("round : " , round)
            print(" data_batch : ", i)
            print(" loss: ", loss)
            writer.add_scalar('Loss/train', loss, counter)
            counter += 1
    torch.save(resnet, "/root/trainedmodel/resnet18_AES_DES3_ECB_full_32.model")

    # lazyLoadAndTrain(resnet, train_partial, 256, feature_dir_dict, 6, 24, 36, 8, "/root/trainedmodel/resnet18.model")
    # des3_data = preprocess.DataSet(1, preprocess.getFeature_ray(cipherDir_des, preprocess.bitcount, 224))
    # aes_data = preprocess.DataSet(0, preprocess.getFeature_ray(cipherDir_aes, preprocess.bitcount, 224))
    # aes_data = preprocess.DataSet(0, preprocess.getFeature_joblib(cipherDir_aes, preprocess.bitcount, 224))
    # des3_data = preprocess.DataSet(1, preprocess.getFeature_joblib(cipherDir_des, preprocess.bitcount, 224))

    # print("finish preprocess at:", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    # trainDataset = dataset.ConcatDataset([aes_data, des3_data])
    # dataloader = DataLoader(trainDataset, batch_size=1024, shuffle=True, num_workers=8)
    # dataloader = DataLoader(trainDataset, batch_size=512, shuffle=True, num_workers=5)

    # train(resnet, dataloader, epoch=10, modelPath="/root/trainedmodel/resnet18_AES_TDES_ECB.model")
    # train(resnet, dataloader, epoch=10, modelPath="/Users/daisy/Downloads/resnet18_AES_TDES_ECB.model")
    # print("finish train at:", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))