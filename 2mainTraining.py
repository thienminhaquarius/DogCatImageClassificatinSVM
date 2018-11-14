
import os
import generate_db as db
import numpy as np
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm


def load_features(src):
    data = []
    labels = []
    with open(src, "r") as file:
        for i, line in enumerate(file):
            feat_path = line[:-1]
            print("[+] Load features : ", feat_path, " Index : ", i)
            label = feat_path.split(os.path.sep)[-2]
            data.append(np.load(feat_path)[0])
            labels.append(label)

        return data, labels


def saveDataOnDisk(modelData, modelSavePath):
    pickle.dump(modelData, open(modelSavePath, "wb"))


def loadDataFromDisk(modelSavePath):
    data = pickle.load(open(modelSavePath, 'rb'))
    return data

svmModelSavePath = './exp/svmlinear/db1/model.dat'
svmResultSavePath = './exp/svmlinear/db1/kqua.dat'

# Train data.------------------------------------------------------
print('\nTraining data and make result...')
dbPath = './db/db1'
trainFileName = 'train.txt'
testFileName = 'test.txt'

print("\n[+] Load features data and labels for training....")
trainData, trainLabels = load_features(os.path.join(dbPath, trainFileName))
print("\n[+] Load features data and labels for testing....")
testData, testLabels = load_features(os.path.join(dbPath, testFileName))
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.fit_transform(testLabels)

# SVMLinear-------------------------------------------------------------------------------------
print('\nTraining SVMLinear...')
modelSvmLinear = svm.LinearSVC()
modelSvmLinear.fit(trainData, trainLabels)
print('Done')

print('Program is still running...')

print('\nSave Model on Disk...')
saveDataOnDisk(modelSvmLinear, svmModelSavePath)
print('Done, Folder changed:"./exp/svmlinear/model.dat"')

print('Program is still running...')

predictLabelsSvmLinear = modelSvmLinear.predict(testData)
svmResult = classification_report(testLabels, predictLabelsSvmLinear, target_names=le.classes_)

del modelSvmLinear

print('\nSave Result on Disk...')
saveDataOnDisk(svmResult, svmResultSavePath)
print('Done, Folder changed:"./exp/svmlinear/kqua.dat"')

print('Program is still running...')

print('\nSVMLinear results')
print(svmResult)
