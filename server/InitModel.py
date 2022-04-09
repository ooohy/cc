import models

def init(modelType: str):
    if modelType == "lenet":
        return models.lenet()
    if modelType == "alexnet":
        return models.alexnet()
    if modelType == "resnet":
        return models.resnet()
