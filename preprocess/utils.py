from cipher import aes
import torch
import time
from multiprocessing import cpu_count
import os
from bitarray import bitarray
# from multiprocessing.pool import ThreadPool as Pool
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import multiprocessing
from functools import partial
import numpy as np
import torch.utils.data as data

import sys
sys.path.append(
    "/Users/daisy/CipherClassification/CryptoClassificationNewBranch")


def file2cipher_mp(fileDir, targetDir, encrypt):
    """
    fileDir: the directory of the file to be encrypted
    targetDir: the directory of the encrypted file
    encryptAlgo: the algorithm to encrypt the file
    """
    def process(file):
        with open(fileDir + '/' + file, 'rb') as f:
            line = f.readlines()
            # join line
            lines = b''.join(line)
            cipher = encrypt(lines)
            with open(targetDir + '/' + file + ".cipher", 'wb') as f:
                f.write(cipher)
    file_array = os.listdir(fileDir)
    args = [file for file in file_array]
    pool = Pool(cpu_count())
    pool.map(process, args)
    pool.close()
    pool.join()


def file2cipher(fileDir, targetDir, encrypt):
    """
    fileDir: the directory of the file to be encrypted
    targetDir: the directory of the encrypted file
    encryptAlgo: the algorithm to encrypt the file
    """
    def process(file):
        with open(fileDir + '/' + file, 'rb') as f:
            line = f.readlines()
            # join line
            lines = b''.join(line)
            cipher = encrypt(lines)
            with open(targetDir + '/' + file + ".cipher", 'wb') as f:
                f.write(cipher)
    file_array = os.listdir(fileDir)
    args = [file for file in file_array]
    map(process, args)


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


def getFeature(file_dir, target_dir, function, inputSize):
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
    pool = ThreadPool(multiprocessing.cpu_count())
    # pool.map(process, args)
    pool.map(process, args)
    pool.close()
    pool.join()
    np.save(target_dir, np.concatenate(feature_nparr, axis=0))


class DataSet(data.Dataset):
    """
    shape: (num * 1 * inputSize * inputSize)
    """

    def __init__(self, label, feature_dir):
        super(DataSet, self).__init__()
        feature_np = np.load(feature_dir)
        self.feature_tensor = torch.from_numpy(feature_np).unsqueeze(1)
        self.label = label

    def __len__(self):
        return self.feature_tensor.shape[0]

    def __getitem__(self, index):
        if torch.cuda.is_available():
            return self.feature_tensor[index].cuda(), self.label[index].cuda()
        else:
            return self.feature_tensor[index], torch.Tensor(self.label)


if __name__ == '__main__':
    # file = "/Users/daisy/Downloads/plaintData/"
    cipherDir = "/Users/daisy/Downloads/cipherData/"
    # encrypt_obj = aes.AES_ECB()
    # file2cipher(file, cipherDir, encrypt_obj.encrypt)
    np_file = "/Users/daisy/Downloads/cipherFeature.feature.npy"
    # t1 = time.time()
    # getFeature(cipherDir, np_file, bitcount, 224)
    # print(time.time() - t1)
    data = DataSet(0, np_file)
    print(data.__len__())
    print(data.__getitem__(0)[0])
    print(data.__getitem__(0)[1].shape)
