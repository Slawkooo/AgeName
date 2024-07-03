import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from os.path import isfile, join
from os import listdir
import numpy as np


model = tf.keras.models.load_model(r"C:\Users\slawo\PycharmProjects\AgeFace\model.h5")

def my_data_generator_test():
    for sample, label in zip(X_test, y_test):
        x = cv2.imread(sample)
        x = cv2.resize(x, IMG_RESOLUTION)
        x = x.flatten()
        x = x / 255
        label = (label - 20) / 30
        yield x, (label)
def my_data_generator_train():
    for sample, label in zip(X_train, y_train):
        x = cv2.imread(sample)
        x = cv2.resize(x, IMG_RESOLUTION)
        x = x.flatten()
        x = x / 255
        label = (label - 20)/30

        yield x, (label)

def my_data_generator_val():
    for sample, label in zip(X_val, y_val):
        x = cv2.imread(sample)
        x = cv2.resize(x, IMG_RESOLUTION)
        x = x.flatten()
        x = x / 255
        label = (label - 20) / 30
        yield x, (label)

IMG_RESOLUTION = (224, 224)
batch_size = 32

testPath = "C:\\Users\\slawo\\PycharmProjects\\AgeFace\\20-50\\20-50\\test"
y_test = []
X_test = []

for i in range(20, 51):
    files = [join(testPath, str(int(i)), f) for f in listdir(join(testPath, str(i))) if
                 isfile(join(testPath, str(i), f))]
    y = [i for y in files]
    files = np.array(files)
    y = np.array(y)
    y_test = np.concatenate((y_test, y))
    X_test = np.concatenate((X_test, files))

trainPath = "C:\\Users\\slawo\\PycharmProjects\\AgeFace\\20-50\\20-50\\train"
y_train = np.array([])
X_train = np.array([])

for i in range(20, 51):
    files = [join(trainPath, str(int(i)), f) for f in listdir(join(trainPath, str(i))) if
             isfile(join(trainPath, str(i), f))]
    y = [i for y in files]
    files = np.array(files)
    y = np.array(y)
    y_train = np.concatenate((y_train, y))
    X_train = np.concatenate((X_train, files))

valPath = "C:\\Users\\slawo\\PycharmProjects\\AgeFace\\20-50\\20-50\\val"
y_val = np.array([])
X_val = np.array([])

for i in range(20, 51):
    files = [join(valPath, str(int(i)), f) for f in listdir(join(valPath, str(i))) if
                 isfile(join(valPath, str(i), f))]
    y = [i for y in files]
    files = np.array(files)
    y = np.array(y)
    y_val = np.concatenate((y_val, y))
    X_val = np.concatenate((X_val, files))


x_shape = (224*224*3,)
y_shape = ()

x_type = tf.float32
y_type = tf.int8

x_type = tf.float32
y_type = tf.int8

train_ds = tf.data.Dataset.from_generator(my_data_generator_train, output_signature=(
         tf.TensorSpec(shape=x_shape, dtype=x_type),
         tf.TensorSpec(shape=y_shape, dtype=y_type)))


train_ds = train_ds.batch(batch_size, drop_remainder=True)

val_ds = tf.data.Dataset.from_generator(my_data_generator_val, output_signature=(
         tf.TensorSpec(shape=x_shape, dtype=x_type),
         tf.TensorSpec(shape=y_shape, dtype=y_type)))

val_ds = val_ds.batch(batch_size, drop_remainder=True)

test_ds = tf.data.Dataset.from_generator(my_data_generator_test, output_signature=(
         tf.TensorSpec(shape=x_shape, dtype=x_type),
         tf.TensorSpec(shape=y_shape, dtype=y_type)))

test_ds = test_ds.batch(batch_size, drop_remainder=True)

model.evaluate(train_ds)
model.evaluate(val_ds)
model.evaluate(test_ds)
