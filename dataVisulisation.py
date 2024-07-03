from os import listdir, mkdir, remove
from os.path import isfile, join
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

dataPath = "C:\\Users\\slawo\\PycharmProjects\\AgeFace\\20-50\\20-50\\val"
Y = np.array([])
allFiles = np.array([])

for i in range(20, 51):
    files = [join(dataPath, str(int(i)), f) for f in listdir(join(dataPath, str(i))) if
             isfile(join(dataPath, str(i), f))]
    y = [i for y in files]
    files = np.array(files)
    y = np.array(y)
    Y = np.concatenate((Y, y))
    allFiles = np.concatenate((allFiles, files))
IMG_RESOLUTION = (224, 224)

a  = (Y[200]-20)/30
print(a.shape,a)


# y = 50
# x = 20
#
# y = (y-20)/30
# x = (x-20)/30
#
# print(x,y)
#
# x = (x*30)+20

