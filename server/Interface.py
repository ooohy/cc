import json
import sys
sys.path.append("../")
import preprocess.CipherText as ct
import torch.utils.data as data
from models import utils
import InitDataLoader
import InitModel
import torch


def train(req_json):
    req_dict = json.load(req_json)
    modelType = req_dict['modelType']
    modelPath = req_dict['modelPath']
    dirJson = req_dict['dirDict']
    ratio = req_dict['ratio']
    epoch = req_dict['epoch'] # test size
    batchSize = req_dict['batchSize']
    cipherMatrixSize = req_dict['cipherMatrixSize']
    bitCountWise = req_dict['bitCountWise']

    # init model
    model = InitModel.init(modelType)

    # init data loader
    dir_dict = json.loads(dirJson) # key is the label of dataset by default
    dataset_arr = []
    for key, value in dir_dict.items():
        dataset_arr.append(ct.DataSet(label=int(key), dirPath=value, cipherMatrixSize=cipherMatrixSize, 
                                            bitCountWise=bitCountWise, source='Hex'))
    dataset = data.ConcatDataset(dataset_arr)
    dataset_dict = InitDataLoader.spilt_train_val_dataset(dataset, ratio)
    trainLoader = data.DataLoader(dataset_dict['train'], batch_size=batchSize, shuffle=True)
    testLoader = data.DataLoader(dataset_dict['val'], batch_size=batchSize, shuffle=True)

    # train model
    utils.train(model, trainLoader, epoch, modelPath=modelPath)

    # val model
    score = utils.val(model, testLoader)
    re_dict = dict()
    re_dict['score'] = score
    return json.dump(re_dict)

def classify(req_dict):
    modelPath = req_dict['modelPath']
    cipherMatrixSize = req_dict['cipherMatrixSize']
    bitCountWise = req_dict['bitCountWise']
    fileDir = req_dict['fileDir']

    # init model
    model = torch.load(modelPath)

    # init data to be detected 
    dataset = ct.DataSet(label=0, dataSize=400000, cipherMatrixSize=224, dirPath=fileDir)
    dataloader = data.DataLoader(dataset, batch_size=64, shuffle=False)

    # classify
    label = utils.classifier(model, dataloader)
    return label