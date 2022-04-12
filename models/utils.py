import torch
import torch.nn as nn
import os
from preprocess import DataSet
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.dataset as dataset


def train(model, dataloader, epoch=10, criterion_name="CrossEntropyLoss", optimizer_name="SGD", lr=0.05, momentum=0.8, modelPath=None):
    if criterion_name == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif criterion_name == "L1Loss":
        criterion = nn.L1Loss()

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for round in range(epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_loss = 0.0
            # print statistics
            print_static(round, i, running_loss)
    if modelPath:
        torch.save(model, modelPath)


def print_static(round, i, running_loss):
    if i % 2000 == 1999:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
            (round, i + 1, running_loss / 2000))


def lazyLoadAndTrain(train, feature_dir_dict, piece_num, dataloader_partial, epoch=10, criterion_name="CrossEntropyLoss", optimizer_name="SGD", lr=0.05, momentum=0.8, modelPath=None):
    feature_iter_dict = dict()
    for key, value in feature_dir_dict.items():
        feature_iter_dict[key] = iter([value + "/" + file for file in os.listdir(feature_dir_dict[key])])
        
    for round in range(epoch):
        for piece in piece_num:
            dataset_array = []
            for key, value in feature_iter_dict.items():
                dataset_array.append(DataSet(int(key), next(value)))
                dataset_piece = dataset.ConcatDataset(dataset_array)
            dataloader_partial(dataset_piece)
            train(model, dataloader_partial, epoch=1, criterion_name=criterion_name, optimizer_name=optimizer_name, lr=lr, momentum=momentum, modelPath=modelPath)

            


    torch.save(model, modelPath)


def val(model, dataloader) :
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            matrix, labels = data
            outputs = model(matrix)
            _, predict = torch.max(outputs.data, 1)

            correct += (predict == labels).sum().item()
    return correct / total

def batchVal(model, dataloader) :
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