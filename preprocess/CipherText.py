import torch.utils.data as data
import torch
import numpy as np
import os
from numba import jit


class DataSet(data.Dataset):
    def __init__(self, label: int, dirPath, source='Hex',
                 cipherMatrixSize=64, bitCountWise=16, dataSize=1024, intNumMatrixFromOneFile=True,
                 fixedNumMatrixFromOneFile=0, use_cuda=False, in_channel=1):
        super(DataSet, self).__init__()
        self.label = label
        self.cipherMatrixSize = cipherMatrixSize
        self.bitCountWise = bitCountWise
        self.dataSize = dataSize
        """ warning
        DO NOT USE np.empty to init a np array, just use array !!!
        """
        # self.matrix_np = np.empty((1, in_channel, cipherMatrixSize, cipherMatrixSize), dtype=np.float32)
        self.matrix_arr = []
        self.bitLenWise = bitCountWise * cipherMatrixSize * cipherMatrixSize
        self.fixedNumMatrixFromOneFile = fixedNumMatrixFromOneFile
        self.intNumMatrixFromOneFile = intNumMatrixFromOneFile
        self.use_cuda = use_cuda
        self.in_channel = in_channel

        # oneMatrixFromOneFile and intNumMatrixFromOneFile can not be used at the same time
        if self.fixedNumMatrixFromOneFile and self.intNumMatrixFromOneFile:
            raise ValueError(
                "oneMatrixFromOneFile and intNumMatrixFromOneFile can not be used at the same time")

        # load data from file in byte or Hex
        if(source == 'Hex'):
            self.loadFromHex(dirPath)
        elif(source == 'Byte'):
            self.loadFromByte(dirPath)
        else:
            raise ValueError("source must be 'Hex' or 'Byte'")

    def __getitem__(self, index):
        if self.use_cuda:
            label = torch.tensor(self.label).cuda()
            matrix = torch.from_numpy(self.matrix_np[index]).cuda()
        else:
            label = torch.tensor(self.label)
            matrix = torch.from_numpy(self.matrix_np[index])

        return matrix, label

    def __len__(self):
        return self.dataSize

    def bit2Matrix(self, bitStream):
        # updata dataSize
        bitWiseInMatrix = self.bitCountWise * \
            self.cipherMatrixSize ** 2 * self.in_channel
        loadedDataSize = len(bitStream) // bitWiseInMatrix
        if loadedDataSize < self.dataSize:
            self.dataSize = loadedDataSize

        bitCountTemp_arr = []
        for count in range(self.dataSize * self.cipherMatrixSize**2 * self.in_channel):
            bitCountTemp_arr.append(
                bitStream[count*self.bitCountWise: (count+1)*self.bitCountWise].count('1'))
        self.matrix_np = np.array(bitCountTemp_arr, dtype=np.float32).reshape(
            self.dataSize, self.in_channel, self.cipherMatrixSize, self.cipherMatrixSize)

    def loadFromByte(self, dirPath):
        bitStream = ''

        # only read fixed len of data from the file
        if self.fixedNumMatrixFromOneFile:
            for file in os.listdir(dirPath):
                with open(dirPath + '/' + file, 'rb') as f:
                    line = f.read()
                bitTemp = ''
                for byte in line:
                    bitTemp += bin(byte)[2:].zfill(8)
                    if len(bitTemp) >= self.bitLenWise*self.fixedNumMatrixFromOneFile:
                        break
                try:
                    bitStream += bitTemp[:self.bitLenWise *
                                         self.fixedNumMatrixFromOneFile]
                except:
                    break
                if len(bitStream) == self.bitLenWise * self.dataSize * self.in_channel * self.cipherMatrixSize**2:
                    break

        # read intNumMatrixFromOneFile number of data from every file
        elif self.intNumMatrixFromOneFile:
            for file in os.listdir(dirPath):
                with open(dirPath + '/' + file, 'rb') as f:
                    line = f.read()
                bitTemp = ''
                for byte in line:
                    bitTemp += bin(byte)[2:].zfill(8)
                n = len(bitTemp) // self.bitLenWise
                bitStream += bitTemp[:self.bitLenWise * n]
                if len(bitStream) >= self.bitLenWise * self.dataSize * self.in_channel * self.cipherMatrixSize ** 2:
                    break

        self.bit2Matrix(bitStream)

    def loadFromHex(self, dirPath):
        bitStream = ''

        # read fixed len data forom a file
        if self.fixedNumMatrixFromOneFile:
            for file in os.listdir(dirPath):
                with open(dirPath + '/' + file, 'r') as f:
                    line = f.read()
                bitTemp = ''
                for hex in line:
                    bitTemp += bin(int(hex, 16))[2:].zfill(8)
                    if len(bitTemp) >= self.bitLenWise * self.fixedNumMatrixFromOneFile:
                        break
                try:
                    bitStream += bitTemp[:self.bitLenWise *
                                         self.fixedNumMatrixFromOneFile]
                except:
                    break
                if len(bitStream) == self.bitLenWise * self.dataSize * self.in_channel * self.cipherMatrixSize**2:
                    break

        # read intNumMatrixFromOneFile number of data from every file
        elif self.intNumMatrixFromOneFile:
            for file in os.listdir(dirPath):
                with open(dirPath + '/' + file, 'r') as f:
                    line = f.read()
                bitTemp = ''
                for hex in line:
                    bitTemp += bin(int(hex, 16))[2:].zfill(8)
                n = len(bitTemp) // self.bitLenWise
                try:
                    bitStream += bitTemp[:self.bitLenWise * n]
                except:
                    break
                if len(bitStream) >= self.bitLenWise * self.dataSize * self.in_channel * self.cipherMatrixSize**2:
                    break

        self.bit2Matrix(bitStream)
