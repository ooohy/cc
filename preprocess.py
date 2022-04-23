import torch
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
# import pandas as pd
import math
import random
import string
from torch.utils.data import Dataset
import modin.pandas as pd
from bitarray.util import urandom
from cipher import des, aes


def file2cipher(file_dir, targetDir, encrypt, label, file_percent=1):
    """
    fileDir: the directory of the file to be encrypted
    targetDir: the directory of the encrypted file
    encryptAlgo: the algorithm to encrypt the file
    """
    # count filenum in file_dir
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

def randCipher(file_dir, num_file, num_workers):
    def process(i):
        file_name = file_dir+ '/' + \
            generate_random_num(10) + '_feature' + ".npy"
        with open(file_name, "wb") as f:
          f.write(urandom(2**22))
    Parallel(num_workers)(delayed(process)(i) for i in range(num_file))

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
        feature_np = np.array(
            feature, detype=np.int8).reshape(-1, inputSize, inputSize)
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
        feature_np = np.array(
            feature, dtype=np.int8).reshape(-1, inputSize, inputSize)
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
        feature_np = np.array(
            feature, dtype=np.int8).reshape(-1, inputSize, inputSize)
        feature_nparr.append(feature_np)

    file_array = os.listdir(file_dir)
    args = [file_dir + "/" + file for file in file_array]
    # pool.map(process, args)
    for f in args:
        process(f)
    return np.concatenate(feature_nparr, axis=0)


def getFeature_joblib(file_dir, function, inputSize, feature_dir, num_workers=24):
    """
    shape : (num * inputSize * inputSize)
    function : bitcount etc.
    get feature in csv
    """
    def process(file):
        # apply the function
        feature = function(file)
        feature = feature[0:(len(feature) // inputSize_2) * inputSize_2]
        feature_df = pd.DataFrame(
            np.array(feature, dtype=np.int8).reshape(-1, inputSize_2))
        return feature_df

    # def pd2csv(index):
    #     feature.iloc[index * each_pieces_len:(index + 1) * each_pieces_len].to_csv(feature_dir + "/" + "feature_slice_" + str(index) + ".csv", index=False)

    inputSize_2 = inputSize ** 2
    args = [file_dir + "/" + file for file in os.listdir(file_dir)]
    feature_pdarr = Parallel(n_jobs=num_workers)(
        delayed(process)(file) for file in args)

    feature = pd.concat(feature_pdarr, axis=0)
    feature.to_csv(feature_dir, index=False)


def getFeature_np(file_dir, feature_dir, function, inputSize, num_workers=24, in_channel=1):
    """
    shape : (num * inputSize * inputSize)
    function : bitcount etc.
    get feature in csv
    """
    def process(file):
        # apply the function
        feature = function(file)
        feature = feature[0:(len(feature) // block_size) * block_size]
        feature_np = np.array(
            feature, dtype=np.float32).reshape(-1, inputSize, inputSize)
        file_name = feature_dir + '/' + \
            generate_random_num(10) + '_feature' + ".npy"
        np.save(file_name, feature_np)

    # def pd2csv(index):
    #     feature.iloc[index * each_pieces_len:(index + 1) * each_pieces_len].to_csv(feature_dir + "/" + "feature_slice_" + str(index) + ".csv", index=False)

    block_size = in_channel * inputSize ** 2
    args = [file_dir + "/" + file for file in os.listdir(file_dir)]
    Parallel(n_jobs=num_workers)(delayed(process)(file) for file in args)


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


class DataSet_csv(data.Dataset):
    def __init__(self, label, feature_dir):
        super(DataSet_csv, self).__init__()
        self.label = torch.tensor(label, dtype=torch.long)
        feature_df = pd.read_csv(feature_dir)
        feature_pt = torch.tensor(feature_df.values, dtype=torch.float32)
        inputSize = int(feature_pt.shape[1] ** 0.5)
        self.feature_pt = feature_pt.reshape(-1, 1, inputSize, inputSize)

    def __getitem__(self, index):
        if torch.cuda.is_available():
            return self.feature_pt[index].cuda(), self.label.cuda()
        else:
            return self.feature_pt[index], self.label

    def __len__(self):
        return self.feature_pt.shape[0]


class DataSet_lazyloading(data.Dataset):
    def __init__(self, label, feature_piece_dir):
        super(DataSet_lazyloading, self).__init__()
        self.feature_piece_dir = feature_piece_dir
        self.label = label
        feature_df = pd.read_csv(feature_piece_dir)
        inputSize = int(feature_df.shape[1] ** 0.5)
        self.feature_pt = torch.tensor(
            feature_df.values, dtype=torch.int8).reshape(-1, 1, inputSize, inputSize)

    def __len__(self):
        return self.feature_pt.shape[0]

    def __getitem__(self, index):
        if torch.cuda.is_available():
            return self.feature_pt[index].cuda(), torch.tensor(self.label, dtype=torch.long).cuda()
        else:
            return self.feature_pt[index], torch.tensor(self.label, dtype=torch.long)


class DataSet_joblib(data.Dataset):
    def __init__(self, label, feature_dir):
        super(DataSet_joblib, self).__init__()
        self.label = label
        feature_dir_array = [feature_dir + "/" +
                             file for file in os.listdir(feature_dir)]
        num_worker = len(feature_dir_array)
        feature_df_array = Parallel(num_worker)(
            delayed(pd.read_csv(feature) for feature in feature_dir_array))
        feature_df = pd.concat(feature_df_array, axis=0)
        inputSize = int(feature_df.shape[1] ** 0.5)
        self.feature_pt = torch.tensor(
            feature_df.values, dtype=torch.int8).reshape(-1, 1, inputSize, inputSize)

    def __len__(self):
        return self.feature_pt.shape[0]

    def __getitem__(self, index):
        if torch.cuda.is_available():
            return self.feature_pt[index].cuda(), torch.tensor(self.label, dtype=torch.long).cuda()
        else:
            return self.feature_pt[index], torch.tensor(self.label, dtype=torch.long)


class DataSet_np(data.Dataset):
    def __init__(self, label, file_path_arr, num_workers=24, in_channel=1):
        super(DataSet_np, self).__init__()
        self.label = torch.tensor(label, dtype=torch.long)
        feature_np_array = Parallel(num_workers)(delayed(np.load)(feature) for feature in file_path_arr)
        feature_np = np.concatenate(feature_np_array, axis=0)
        size = feature_np.shape[1]
        feature_np = feature_np.reshape(-1, in_channel, size, size)
        self.feature_pt = torch.from_numpy(feature_np)

    def __len__(self):
        return self.feature_pt.shape[0]

    def __getitem__(self, index):
        if torch.cuda.is_available():
            return self.feature_pt[index].cuda(), self.label.cuda()
        else:
            return self.feature_pt[index], self.label


def generate_random_num(length):
    str_list = [random.choice(string.digits+string.ascii_letters)
                for i in range(length)]
    random_str = "".join(str_list)
    return random_str

class DataSet_rand_rnn(data.Dataset):
    def __init__(self, label, encoder, num_workers=24):
        urandom()


        urandom



if __name__ == '__main__':
    # wiki_zh = "/root/autodl-tmp/data/wiki_zh"
    # imagenet = "/root/autodl-tmp/data/imagenet100"
    # voice = "/root/autodl-tmp/data/voice"
    rand = "/root/autodl-tmp/data/rand"

    cipherDir_aes = "/root/autodl-tmp/cipher/aes"
    cipherDir_des = "/root/autodl-tmp/cipher/des3"
    cipherDir_rand_aes = "/root/autodl-tmp/cipher/rand/aes"
    cipherDir_rand_des3 = "/root/autodl-tmp/cipher/rand/des3"
    # voice = "/Users/daisy/Downloads/voice"
    # cipherDir_aes = "/Users/daisy/Downloads/cipher_aes"
    # cipherDir_des = "/Users/daisy/Downloads/cipher_des"

    aes_ecb = aes.AES_ECB()
    des3_ecb = des.TDES_ECB()

    # count = 0
    # for dir in os.listdir(wiki_zh):
    #     count = file2cipher(wiki_zh + "/" + dir, cipherDir_des,
    #                         des3_ecb.encrypt, "wiki_DES3_ECB", count)
    # for dir in os.listdir(imagenet):
    #     count = file2cipher(imagenet + "/" + dir, cipherDir_des,
    #                         des3_ecb.encrypt, "imagenet_DES3_ECB", count)
    # file2cipher(voice, cipherDir_des, des3_ecb.encrypt, "voice_DES3_ECB", 0.1)
    file2cipher(rand, cipherDir_rand_des3, des3_ecb.encrypt, "rand_DES3_ECB")
    file2cipher(rand, cipherDir_rand_aes, aes_ecb.encrypt, "rand_AES_ECB")

    # count = 0
    # for dir in os.listdir(wiki_zh):
    #     count = file2cipher(wiki_zh + "/" + dir, cipherDir_aes,
    #                         aes_ecb.encrypt, "wiki_AES_ECB", count)
    # for dir in os.listdir(imagenet):
    #     count = file2cipher(imagenet + "/" + dir, cipherDir_aes,
    #                         aes_ecb.encrypt, "imagenet_AES_ECB", count)
    # file2cipher(voice, cipherDir_aes, aes_ecb.encrypt, "voice_AES_ECB", 0.1)

    # np_file = "/root/auto-tmp/feature"
    # t1 = time.time()
    # getFeature(cipherDir, bitcount, 224)
    # print(time.time() - t1)
    # data = DataSet(0, np_file)

    # feature_dir_aes = "/Users/daisy/Downloads/feature/aes_feature"
    # feature_dir_des3 = "/Users/daisy/Downloads/feature/des3_feature"
    # feature_dir_aes = "/root/autodl-tmp/feature/feature_pieces/aes_full"
    # feature_dir_des3 = "/root/autodl-tmp/feature/feature_pieces/des_full"
    # feature_aes_dir = "/root/autodl-tmp/feature/aes_256_in8"
    # feature_des3_dir = "/root/autodl-tmp/feature/des_256_in8"
    # feature_des3_dir = "/root/autodl-tmp/feature/rand_256_in8"
    # getFeature_joblib(cipherDir_aes, bitcount, 224, feature_dir_aes, num_workers=24)
    # getFeature_joblib(cipherDir_des, bitcount, 224, feature_dir_des3, num_workers=24)
    feature_aes_dir = '/root/autodl-tmp/feature/aes_rand_256_in8'
    feature_des3_dir = '/root/autodl-tmp/feature/des3_rand_256_in8'

    # feature_dir_aes = "/Users/daisy/Downloads/feature/aes_feature"
    # feature_dir_des3 = "/Users/daisy/Downloads/feature/des3_feature"
    # getFeature_joblib(cipherDir_aes, bitcount, 224, feature_dir_aes, pieces=24, num_workers=12)
    # getFeature_joblib(cipherDir_des, bitcount, 224, feature_dir_des3, pieces=24, num_workers=12)
    # aes_feature = "/root/autodl-tmp/feature/AES_ECB_feature.csv"

    # des_feature = "/root/autodl-tmp/feature/TDES_ECB_feature.csv"

    # aes_full_dir = "/root/autodl-tmp/feature/feature_pieces/aes_full"

    # des_full_dir = "/root/autodl-tmp/feature/feature_pieces/des_full"

    # aes_arr = [aes_full_dir + '/' + path for path in os.listdir(aes_full_dir)]
    # des_arr = [des_full_dir + '/' + path for path in os.listdir(des_full_dir)]

    # aes_df_arr = []
    # des_df_arr = []
    # getFeature_np(cipherDir_rand_aes, feature_aes_dir,
                #   bitcount, 256, num_workers=24, in_channel=8)
    getFeature_np(cipherDir_rand_des3, feature_des3_dir,
                  bitcount, 256, num_workers=24, in_channel=8)

    # for file in aes_arr:
    #     aes_df_arr.append(pd.read_csv(aes_arr))
    # aes_df = pd.concat(aes_df_arr)
    # aes_df.to_csv(aes_feature)

    # for file in des_arr:
    #     des_df_arr.append(pd.read_csv(aes_arr))
    # des_df = pd.concat(des_df_arr)
    # des_df.to_csv(des_feature)
