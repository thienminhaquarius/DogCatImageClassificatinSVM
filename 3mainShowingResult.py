import os
import generate_db as db
import numpy as np
import pickle

def saveDataOnDisk(modelData, modelSavePath):
    pickle.dump(modelData, open(modelSavePath, "wb"))

def loadDataFromDisk(modelSavePath):
    data = pickle.load(open(modelSavePath, 'rb'))
    return data

svmResultSavePath = './exp/svmlinear/db1/kqua.dat'

# SVMLinear-------------------------------------------------------------------------------------

print('\nLoad KNN result from: ',svmResultSavePath)
# load Result from Disk
svmResultlFromDisk = loadDataFromDisk(svmResultSavePath)

print('Please wait...')

print('SVMLinear results')
print(svmResultlFromDisk)
