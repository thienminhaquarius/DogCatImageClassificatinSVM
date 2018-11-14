
import os
import generate_db as db
import numpy as np

#Prepare data for training

def write_file(path, files):
    print("[+]Write ", path)
    with open(path, "w") as f:
        for file in files:
            f.write(file)
            f.write("\n")


# All Images Path in one file...--------------------------------
print('\nAll databases path in one file...')
imagesSrc='./images'
all_files=[]
for folder in os.listdir(imagesSrc):
	print("[+]Access folder ",folder)
	folder_path = os.path.join(imagesSrc, folder)
	files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]    
	all_files.extend(list(set(files)))
write_file("allFilePath.txt", all_files)
print("Done!, File change: ./allFilePath.txt")
#/All Images Path in one file...---------------------------------

# Features extraction...-----------------------------------------
print('\nFeatures extraction...')
import extract_features as fe
allFilePathSrc='./allFilePath.txt'
fe.extract_features(allFilePathSrc)
print("Done!, File change: ./features/vgg16_fc2")
#/Features extraction...-----------------------------------------

# Generate Database for training and testing...------------------
print('\nGenerate Database for training and testing...')
featuresSrc='./features/vgg16_fc2'
dbPath='./db/db1'
db.generate_data(featuresSrc,dbPath)
print("\nDone, File Change: ./db/db1")
#/End Generate Database for training and testing...------------------



