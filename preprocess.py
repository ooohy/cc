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

import sys
sys.path.append(
    "/Users/daisy/CipherClassification/CryptoClassificationNewBranch")


def file2cipher(file_dir, targetDir, encrypt, label, file_percent=1):
    """
    fileDir: the directory of the file to be encrypted
    targetDir: the directory of the encrypted file
    encryptAlgo: the algorithm to encrypt the file
    """
    # count filenum in file_dir
    file_num = 0
    count = 0
    for root, dirs, files in os.walk(file_dir):
        file_num += len(files)
    file_num = int(file_num * file_percent)

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            with open(os.path.join(root,file), "rb") as f:
                line = f.readlines()
                # join line
                lines = b''.join(line)
                cipher = encrypt(lines)
                with open(targetDir + '/' + label + "__" + str(count) + ".cipher", 'wb') as f:
                    f.write(cipher)
                count += 1
                if count == file_num:
                    return count
    return count



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
        feature_np = np.array(feature).reshape(-1, inputSize, inputSize)
        feature_nparr.append(feature_np)

    file_array = os.listdir(file_dir)
    args = [file_dir + "/" + file for file in file_array]
    pool = ThreadPool(32)
    # pool.map(process, args)
    pool.map(process, args)
    pool.close()
    pool.join()
    return np.concatenate(feature_nparr, axis=0)


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
        feature_np = np.array(feature).reshape(-1, inputSize, inputSize)
        feature_nparr.append(feature_np)

    file_array = os.listdir(file_dir)
    args = [file_dir + "/" + file for file in file_array]
    # pool.map(process, args)
    map(process, args)
    return np.concatenate(feature_nparr, axis=0)


class DataSet(data.Dataset):
    """
    shape: (num * 1 * inputSize * inputSize)
    """

    def __init__(self, label, feature):
        super(DataSet, self).__init__()
        self.feature_tensor = torch.from_numpy(feature).unsqueeze(1)
        self.label = label

    def __len__(self):
        return self.feature_tensor.shape[0]

    def __getitem__(self, index):
        if torch.cuda.is_available():
            return self.feature_tensor[index].cuda(), self.label[index].cuda()
        else:
            return self.feature_tensor[index], torch.Tensor(self.label)


if __name__ == '__main__':
    wiki_zh = "/root/autodl-tmp/data/wiki_zh"
    imagenet = "/root/autodl-tmp/data/imagenet100"
    voice = "/root/autodl-tmp/data/voice"

    cipherDir_aes = "/root/autodl-tmp/cipher/aes"
    cipherDir_des = "/root/autodl-tmp/cipher/des3"

    aes_ecb = aes.AES_ECB()
    des3_ecb = des.TDES_ECB()

    count = 0
    for dir in os.listdir(wiki_zh):
        count = file2cipher(wiki_zh + "/" + dir, cipherDir_des,
                            des3_ecb.encrypt, "wiki_DES3_ECB", count)
    for dir in os.listdir(imagenet):
        count = file2cipher(imagenet + "/" + dir, cipherDir_des,
                            des3_ecb.encrypt, "imagenet_DES3_ECB", count)
    file2cipher(voice, cipherDir_des, des3_ecb.encrypt,
                "voice_DES3_ECB", count)

    count = 0
    for dir in os.listdir(wiki_zh):
        count = file2cipher(wiki_zh + "/" + dir, cipherDir_aes,
                            aes_ecb.encrypt, "wiki_AES_ECB", count)
    for dir in os.listdir(imagenet):
        count = file2cipher(imagenet + "/" + dir, cipherDir_aes,
                            aes_ecb.encrypt, "imagenet_AES_ECB", count)
    file2cipher(voice, cipherDir_aes, aes_ecb.encrypt, "voice_AES_ECB", count)

    # np_file = "/root/auto-tmp/feature"
    # t1 = time.time()
    # getFeature(cipherDir, bitcount, 224)
    # print(time.time() - t1)
    # data = DataSet(0, np_file)
