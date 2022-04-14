from asyncore import ExitNow
import torch
import torch.nn as nn
import os
from preprocess import DataSet_lazyloading
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.dataset as dataset
import gc
import time
from joblib import Parallel, delayed
import math


def train(model, dataloader, epoch=10, criterion_name="CrossEntropyLoss", optimizer_name="SGD", lr=0.05, momentum=0.8, modelPath=None, lazyloading=False):
    if criterion_name == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif criterion_name == "L1Loss":
        criterion = nn.L1Loss()

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    last_loss = 0
    for round in range(epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.long()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

           # print statistics
            last_loss = 0
            if lazyloading is False:
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (round, i + 1, running_loss / 2000))
                    print("  >>  ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
                    # last_loss = running_loss
                    last_loss = running_loss / 2000
                    running_loss = 0.0
    if modelPath is not None:
        torch.save(model, modelPath)
    if lazyloading is True:
        return last_loss


def lazyLoadAndTrain(model, train_partial, batchsize, feature_dir_dict, step=8, num_workers=24, pieces_num=24, epoch=10, modelPath=None):
    """
    dataloader_partial : shuffle has been fixed
    """
    feature_pieces_dict = dict()
    for key, value in feature_dir_dict.items():
        feature_pieces_dict[key] = [value + "/" + file for file in os.listdir(feature_dir_dict[key])]
        

    for round in range(epoch):
        for p in range(0, pieces_num, step):
            dataset_array = []
            # for key, value in feature_pieces_dict.items():
            #     dataset_array.append(DataSet_lazyloading(int(key), next(value)))
            #     dataset_piece = dataset.ConcatDataset(dataset_array)
            label_arr = []
            dir_arr = []
            for key, value in feature_pieces_dict.items():
                label_arr.extend([key]*step)
                dir_arr.extend(value[p:p+step])
            dataset_array = Parallel(num_workers)(delayed(DataSet_lazyloading)(label, dir) for label, dir in zip(label_arr, dir_arr))
            dataset_pieces = dataset.ConcatDataset(dataset_array)
            dataloader = DataLoader(dataset_pieces, shuffle=True, batch_size=batchsize)
            loss = train_partial(model, dataloader, epoch=1, modelPath=None, lazyloading=True)

            # print Loss
            try: 
                print('[%d, %2d] loss: %.3f' %(round, math.ceil(p/step), loss/ 2000))
            except:
                pass
            # print time
            print("  >>  ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            # clear memory
            # del dataset_array, dataset_piece, dataloader
            # gc.collect()

    torch.save(model, modelPath)


def val(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            matrix, labels = data
            outputs = model(matrix)
            _, predict = torch.max(outputs.data, 1)

            correct += (predict == labels).sum().item()
    return correct / total


def batchVal(model, dataloader):
    correct = 0
    total = dataloader.__len__()
    pre = torch.empty()
    with torch.no_grad():
        for data in dataloader:
            matrix, labels = data
            outputs = model(matrix)
            _, predict_tensor = torch.max(outputs.data, 1)
            _, predict = torch.mode(predict_tensor)
            correct += (predict == labels)


def classifier(model, dataloader):
    pre = torch.empty()
    with torch.no_grad():
        for data in dataloader:
            matrix, _ = data
            outputs = model(matrix)
            _, predict = torch.max(outputs.data, 1)
            pre = torch.cat((pre, predict), 0)
    _, re = torch.mode(pre)
    re = re.tolist()[0]
    return(re)
