from pydoc import describe
from cipher import aes
from cipher import des
import torch
import time
import os
from bitarray import bitarray
# from multiprocessing.pool import ThreadPool as Pool
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from functools import partial
import numpy as np
import torch.utils.data as data
import ray
from joblib import Parallel, delayed
import random
import string
from models import ResNet
from torch.utils.data import dataset
from torch.utils.data import DataLoader



def file2cipher(file_dir, targetDir, encrypt, label, file_percent=1):
    """
    fileDir: the directory of the file to be encrypted
    targetDir: the directory of the encrypted file
    encryptAlgo: the algorithm to encrypt the file
    """
    # count filenum in file_dir
    file_num = 0
    count = 0
    file_array = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            file_array.append(os.path.join(root, file))
    file_array = file_array[:int(len(file_array) * file_percent)]

    def process(file):
        with open(file, "rb") as f:
            line = f.readlines()
            # join line
            lines = b''.join(line)
            cipher = encrypt(lines)
            with open(targetDir + '/' + label + "__" + generate_random_num(10) + ".cipher", 'wb') as f:
                f.write(cipher)
    Parallel(n_jobs=24)(delayed(process)(file) for file in file_array)


def bitcount(file_path, bitCountWise=8):
    with open(file_path, 'rb') as f:
        count_arr = []
        line = f.readlines()
        # join line
        lines = b''.join(line)
        # get feature
        ba = bitarray()
        ba.frombytes(lines)
        p = 0
        while(p < len(ba)):
            count_arr.append(ba[p:p+bitCountWise].count(1))
            p += bitCountWise
    return count_arr


def getFeature_mp(file_dir, function, inputSize):
    """
    shape : (num * inputSize * inputSize)
    function : bitcount etc.
    """
    feature_nparr = []

    def process(file):
        # apply the function
        feature = function(file)
        feature = feature[0:(len(feature) // inputSize ** 2) * inputSize ** 2]
        feature_np = np.array(feature, detype=np.int8).reshape(-1, inputSize, inputSize)
        feature_nparr.append(feature_np)

    file_array = os.listdir(file_dir)
    args = [file_dir + "/" + file for file in file_array]
    pool = ThreadPool(32)
    # pool.map(process, args)
    pool.map(process, args)
    pool.close()
    pool.join()
    return np.concatenate(feature_nparr, axis=0)


def getFeature_ray(file_dir, function, inputSize):
    """
    shape : (num * inputSize * inputSize)
    function : bitcount etc.
    """

    @ray.remote
    def process(file):
        # apply the function
        feature = function(file)
        feature = feature[0:(len(feature) // inputSize ** 2) * inputSize ** 2]
        feature_np = np.array(feature, dtype=np.int8).reshape(-1, inputSize, inputSize)
        return feature_np

    file_array = [file_dir + "/" + file for file in os.listdir(file_dir)]

    feature = [process.remote(file) for file in file_array]
    return np.concatenate(ray.get(feature), axis=0)


def getFeature(file_dir, function, inputSize):
    """
    shape : (num * inputSize * inputSize)
    function : bitcount etc.
    """
    feature_nparr = []

    def process(file):
        # apply the function
        feature = function(file)
        feature = feature[0:(len(feature) // inputSize ** 2) * inputSize ** 2]
        feature_np = np.array(feature, dtype=np.int8).reshape(-1, inputSize, inputSize)
        feature_nparr.append(feature_np)

    file_array = os.listdir(file_dir)
    args = [file_dir + "/" + file for file in file_array]
    # pool.map(process, args)
    for f in args:
        process(f)
    return np.concatenate(feature_nparr, axis=0)

def getFeature_joblib(file_dir, function, inputSize):
    """
    shape : (num * inputSize * inputSize)
    function : bitcount etc.
    """
    def process(file):
        # apply the function
        feature = function(file)
        feature = feature[0:(len(feature) // inputSize ** 2) * inputSize ** 2]
        feature_np = np.array(feature, dtype=np.int8).reshape(-1, inputSize, inputSize)
        return feature_np

    file_array = os.listdir(file_dir)
    args = [file_dir + "/" + file for file in file_array]

    
    feature_nparr = Parallel(n_jobs=23)(delayed(process)(file) for file in args)
    return np.concatenate(feature_nparr, axis=0)


class DataSet(data.Dataset):
    """
    shape: (num * 1 * inputSize * inputSize)
    """

    def __init__(self, label, feature_np):
        super(DataSet, self).__init__()
        self.feature_np = feature_np
        self.label = label

    def __len__(self):
        return self.feature_np.shape[0]

    def __getitem__(self, index):
        if torch.cuda.is_available():
            return torch.from_numpy(self.feature_np[index]).unsqueeze(0).cuda(), torch.tensor(self.label, dtype=torch.long).cuda()
        else:
            return torch.from_numpy(self.feature_np[index]).unsqueeze(0), torch.tensor(self.label, dtype=torch.long)


def generate_random_num(length):
    str_list=[random.choice(string.digits+string.ascii_letters) for i in range(length)]
    random_str="".join(str_list)
    return random_str

if __name__ == '__main__':
    wiki_zh = "/root/autodl-tmp/data/wiki_zh"
    imagenet = "/root/autodl-tmp/data/imagenet100"
    voice = "/root/autodl-tmp/data/voice"

    cipherDir_aes = "/root/autodl-tmp/cipher/aes_mini"
    cipherDir_des = "/root/autodl-tmp/cipher/des3_mini"
    # voice = "/Users/daisy/Downloads/voice"
    # cipherDir_aes = "/Users/daisy/Downloads/cipher_aes"
    # cipherDir_des = "/Users/daisy/Downloads/cipher_des"

    # aes_ecb = aes.AES_ECB()
    # des3_ecb = des.TDES_ECB()

    # count = 0
    # for dir in os.listdir(wiki_zh):
    #     count = file2cipher(wiki_zh + "/" + dir, cipherDir_des,
    #                         des3_ecb.encrypt, "wiki_DES3_ECB", count)
    # for dir in os.listdir(imagenet):
    #     count = file2cipher(imagenet + "/" + dir, cipherDir_des,
    #                         des3_ecb.encrypt, "imagenet_DES3_ECB", count)
    # file2cipher(wiki_zh, cipherDir_des, des3_ecb.encrypt, "wiki_DES3_ECB", 0.8)
    # file2cipher(imagenet, cipherDir_des, des3_ecb.encrypt, "imagenet_DES3_ECB", 0.08)
    # file2cipher(voice, cipherDir_des, des3_ecb.encrypt, "voice_DES3_ECB", 0.8)

    # count = 0
    # for dir in os.listdir(wiki_zh):
    #     count = file2cipher(wiki_zh + "/" + dir, cipherDir_aes,
    #                         aes_ecb.encrypt, "wiki_AES_ECB", count)
    # for dir in os.listdir(imagenet):
    #     count = file2cipher(imagenet + "/" + dir, cipherDir_aes,
    #                         aes_ecb.encrypt, "imagenet_AES_ECB", count)
    # file2cipher(wiki_zh, cipherDir_aes, aes_ecb.encrypt, "wiki_AES_ECB", 0.8)
    # file2cipher(imagenet, cipherDir_aes, aes_ecb.encrypt, "imagenet_AES_ECB", 0.08)
    # file2cipher(voice, cipherDir_aes, aes_ecb.encrypt, "voice_AES_ECB", 0.8)

    # np_file = "/root/auto-tmp/feature"
    # t1 = time.time()
    # getFeature(cipherDir, bitcount, 224)
    # print(time.time() - t1)
    # data = DataSet(0, np_file)

    # feature prepare
    print("start preprocess at:", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    # cipherDir_aes = "/Users/daisy/Downloads/cipher_aes"
    # cipherDir_des = "/Users/daisy/Downloads/cipher_des"
    cipherDir_aes = "/root/autodl-tmp/cipher/aes_mini"
    cipherDir_des = "/root/autodl-tmp/cipher/des3_mini"
    # cipherDir_aes = "/root/testdata/aes"
    # cipherDir_des = "/root/testdata/des"
    # featureDir_des = "/root/autodl-tmp/feature/des3_mini_feature"
    # featureDir_aes = "/root/autodl-tmp/feature/aes_mini_feature"
    featureDir_dataloader_mini_pytorch = "/root/autodl-tmp/feature/feature_dataloader_mini"


    aes_dataset = DataSet(0, getFeature_joblib(cipherDir_aes, bitcount, 224))    
    des3_dataset = DataSet(1, getFeature_joblib(cipherDir_des, bitcount, 224))
    trainDataset = dataset.ConcatDataset([aes_dataset, des3_dataset])
    dataloader = DataLoader(trainDataset, batch_size=256, shuffle=True, num_workers=12)
    torch.save(dataloader, featureDir_dataloader_mini_pytorch)
    print("finish preprocess at:", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

