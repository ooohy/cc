from models.ResNet import ResNet18
from models.utils import train
from models.utils import lazyLoadAndTrain
import preprocess
import torch
from preprocess import DataSet, DataSet_joblib
from torch.utils.data import DataLoader
import torch.utils.data as data 
import time
import ray
import json
from functools import partial

# ray.init(
#     _system_config={
#         "object_spilling_config": json.dumps(
#             {"type": "filesystem", "params": {"directory_path": "/Users/daisy/ray"}},
#         )
#     },
# )



if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    # print train start time
    # print("start preprocess at:", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    # cipherDir_aes = "/Users/daisy/Downloads/cipher_aes"
    # cipherDir_des = "/Users/daisy/Downloads/cipher_des"
    cipherDir_aes = "/root/autodl-tmp/cipher/aes"
    cipherDir_des = "/root/autodl-tmp/cipher/des3"
    # cipherDir_aes = "/root/testdata/aes"
    # cipherDir_des = "/root/testdata/des"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_partial = partial(train, epoch=10)
    resnet = ResNet18().to(device)

    # feature_dir_dict = dict()
    # feature_dir_aes = "/Users/daisy/Downloads/feature/aes_feature"
    # feature_dir_des3 = "/Users/daisy/Downloads/feature/des3_feature"
    # feature_dir_aes = "/root/autodl-tmp/feature/feature_pieces/aes"
    # feature_dir_des3 = "/root/autodl-tmp/feature/feature_pieces/des3"
    # feature_dir_dict[0] = feature_dir_aes
    # feature_dir_dict[1] = feature_dir_des3
    # dataset = torch.load("/root/autodl-tmp/feature/feature_dataset_mini")
    # dataloader = DataLoader(dataset, 128, True)
    # train(resnet, dataloader, 10, "/root/trainedmodel/resnet18.model")
    feature_dir_aes = "/root/autodl-tmp/feature/feature_pieces/aes_full"
    feature_dir_des3 = "/root/autodl-tmp/feature/feature_pieces/des_full"
    # dataset_aes = DataSet_joblib(0, feature_dir_aes)
    # dataset_des3 = DataSet_joblib(1, feature_dir_des3)
    # dataset = data.ConcatDataset([dataset_aes, dataset_des3])
    lazyLoadAndTrain(resnet, train_partial)

    dataloader = DataLoader(dataset, 128, True)
    train(resnet, dataloader, 8, modelPath="/root/trainedmodel/resnet18_AESTDES_ECB.py")



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