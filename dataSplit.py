from os import listdir, mkdir, remove
from os.path import isfile, join
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

dataPath = "C:\\Users\\slawo\\PycharmProjects\\AgeFace\\.venv\\20-50\\20-50\\train"
Y = np.array([])
allFiles = np.array([])

for i in range(20,51):
    files = [f for f in listdir(join(dataPath, str(i))) if isfile(join(dataPath, str(i), f))]
    y = [i for y in files]
    files = np.array(files)
    y = np.array(y)
    Y = np.concatenate((Y, y))
    allFiles = np.concatenate((allFiles, files))



X_train, X_val, y_train, y_val = train_test_split(allFiles, Y, test_size=0.2, random_state=42)


mkdir("/20-50\\20-50\\val")
for i in range(20,51):
    mkdir(join("/20-50\\20-50\\val", str(i)))


for i in range(len(y_val)):
    savePath = join("/20-50\\20-50\\val", str(int(y_val[i])))
    filePath = join("/20-50\\20-50\\train", str(int(y_val[i])), str(X_val[i]))
    image = cv2.imread(filePath)
    filename = str(X_val[i])
    cv2.imwrite(join(savePath,filename), image)
    remove(join("/20-50\\20-50\\train", str(int(y_val[i])), str(X_val[i])))
