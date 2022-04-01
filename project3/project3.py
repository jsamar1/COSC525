# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:14:26 2022

@author: Riley
"""
import numpy as np
import os
import PIL
#from PIL import Image
import tensorflow as tf
import pathlib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt


def load_image(filename, data_dir):
    images = tf.zeros((1,32,32,1), dtype='uint8')
    for name in filename:
        img = tf.io.read_file(data_dir + name)
        image = tf.image.decode_png(img, channels=1)
        image = tf.expand_dims(image,axis=0) # makes dims (1,32,32,1)
        images = tf.keras.layers.Concatenate(axis=0)([images,image])
    images = images[1:] # gets rid of leading zeros
    images = tf.cast(images, 'float16')
    Xmax = 255 # tf.math.reduce_max(images_train)
    Xmin = 0   # tf.math.reduce_min(images_train) 
    images = (images-Xmin)/(Xmax-Xmin)
    return images


def FullyConnected(task, lr):
  if task=='race':
    n=7
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    dataset_train = tf.data.Dataset.from_tensor_slices((images_train,race_train))
    dataset_train = dataset_train.batch(16)
    dataset_test = tf.data.Dataset.from_tensor_slices((images_test,race_test))
    dataset_test = dataset_test.batch(16)
  elif task=='age':
    n=9
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    dataset_train = tf.data.Dataset.from_tensor_slices((images_train,age_train))
    dataset_train = dataset_train.batch(16)
    dataset_test = tf.data.Dataset.from_tensor_slices((images_test,age_test))
    dataset_test = dataset_test.batch(16)
  elif task=='gender':
    n=1
    loss = 'binary_crossentropy'
    activation = 'sigmoid'
    dataset_train = tf.data.Dataset.from_tensor_slices((images_train,gender_train))
    dataset_train = dataset_train.batch(16)
    dataset_test = tf.data.Dataset.from_tensor_slices((images_test,gender_test))
    dataset_test = dataset_test.batch(16)
  Model = tf.keras.Sequential()
  Model.add(tf.keras.layers.Flatten(input_shape=(32,32, 1)))
  Model.add(tf.keras.layers.Dense(1024, activation='tanh'))
  Model.add(tf.keras.layers.Dense(512, activation='sigmoid'))
  Model.add(tf.keras.layers.Dense(100, activation='relu'))
  Model.add(tf.keras.layers.Dense(n, activation=activation))
  Model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                loss=loss,
                metrics=['accuracy'])
  return Model, dataset_train, dataset_test

def Convolutional(task, lr):
  if task=='race':
    n=7
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    dataset_train = tf.data.Dataset.from_tensor_slices((images_train,race_train))
    dataset_train = dataset_train.batch(16)
    dataset_test = tf.data.Dataset.from_tensor_slices((images_test,race_test))
    dataset_test = dataset_test.batch(16)
  elif task=='age':
    n=9
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    dataset_train = tf.data.Dataset.from_tensor_slices((images_train,age_train))
    dataset_train = dataset_train.batch(16)
    dataset_test = tf.data.Dataset.from_tensor_slices((images_test,age_test))
    dataset_test = dataset_test.batch(16)
  elif task=='gender':
    n=1
    loss = 'binary_crossentropy'
    activation = 'sigmoid'
    dataset_train = tf.data.Dataset.from_tensor_slices((images_train,gender_train))
    dataset_train = dataset_train.batch(16)
    dataset_test = tf.data.Dataset.from_tensor_slices((images_test,gender_test))
    dataset_test = dataset_test.batch(16)

  convModel = tf.keras.Sequential()
  convModel.add(tf.keras.layers.Conv2D(40,5,activation='ReLU'))
  convModel.add(tf.keras.layers.MaxPool2D())
  convModel.add(tf.keras.layers.Flatten())
  convModel.add(tf.keras.layers.Dense(100,activation='ReLU'))
  convModel.add(tf.keras.layers.Dense(n,activation=activation))
  convModel.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                loss=loss,
                metrics=['accuracy'])
  return convModel, dataset_train, dataset_test

def Convolutional2(task, lr):
  if task=='race':
    n=7
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    dataset_train = tf.data.Dataset.from_tensor_slices((images_train,race_train))
    dataset_train = dataset_train.batch(16)
    dataset_test = tf.data.Dataset.from_tensor_slices((images_test,race_test))
    dataset_test = dataset_test.batch(16)
  elif task=='age':
    n=9
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    dataset_train = tf.data.Dataset.from_tensor_slices((images_train,age_train))
    dataset_train = dataset_train.batch(16)
    dataset_test = tf.data.Dataset.from_tensor_slices((images_test,age_test))
    dataset_test = dataset_test.batch(16)
  elif task=='gender':
    n=1
    loss = 'binary_crossentropy'
    activation = 'sigmoid'
    dataset_train = tf.data.Dataset.from_tensor_slices((images_train,gender_train))
    dataset_train = dataset_train.batch(16)
    dataset_test = tf.data.Dataset.from_tensor_slices((images_test,gender_test))
    dataset_test = dataset_test.batch(16)

  conv2Model = tf.keras.Sequential()
  conv2Model.add(tf.keras.layers.Conv2D(16,3,activation='ReLU'))
  conv2Model.add(tf.keras.layers.Conv2D(32,3,activation='ReLU'))
  conv2Model.add(tf.keras.layers.MaxPool2D())
  conv2Model.add(tf.keras.layers.Conv2D(64,3,activation='ReLU'))
  conv2Model.add(tf.keras.layers.Conv2D(128,3,activation='ReLU'))
  conv2Model.add(tf.keras.layers.MaxPool2D())
  conv2Model.add(tf.keras.layers.Flatten())
  conv2Model.add(tf.keras.layers.Dense(100,activation='ReLU'))
  conv2Model.add(tf.keras.layers.Dense(n,activation=activation))
  conv2Model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                loss=loss,
                metrics=['accuracy'])
  
  return conv2Model, dataset_train, dataset_test

def getLabels():
    num = 1000
    train_labels = pd.read_csv('fairface_label_train.csv')
    file_train, age_train, gender_train, race_train = train_labels.iloc[:num,0].tolist(), train_labels.iloc[:num,1].to_numpy().reshape(-1,1), train_labels.iloc[:num,2].to_numpy(), train_labels.iloc[:num,3].to_numpy().reshape(-1,1)
    test_labels= pd.read_csv('fairface_label_val.csv')
    file_test, age_test, gender_test, race_test = test_labels.iloc[:num,0].tolist(), test_labels.iloc[:num,1].to_numpy().reshape(-1,1), test_labels.iloc[:num,2].to_numpy(), test_labels.iloc[:num,3].to_numpy().reshape(-1,1)
    
    age_train, age_test = OneHotEncoder(sparse=False).fit_transform(age_train), OneHotEncoder(sparse=False).fit_transform(age_test)
    gender_train, gender_test = LabelEncoder().fit_transform(gender_train), LabelEncoder().fit_transform(gender_test)
    race_train, race_test = OneHotEncoder(sparse=False).fit_transform(race_train), OneHotEncoder(sparse=False).fit_transform(race_test)
    return file_train, file_test, age_train, age_test, gender_train, gender_test, race_train, race_test



file_train, file_test, age_train, age_test, gender_train, gender_test, race_train, race_test = getLabels()
data_dir_train = r"C:\Unreal Engine\Projects\COSC525\FromHub\project3\train_imgs\\"
data_dir_test = r"C:\Unreal Engine\Projects\COSC525\FromHub\project3\val_imgs\\"

images_train = load_image(file_train, data_dir_train)
images_test = load_image(file_test, data_dir_test)


#model, dataset_train, dataset_test = FullyConnected('race', 0.001)
#model, dataset_train, dataset_test = Convolutional('race', 0.001)
model, dataset_train, dataset_test = Convolutional2('gender', 0.1)


history = model.fit(dataset_train,validation_data=dataset_test, epochs=10)
  
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()
  
  
  
  
  
  
  
  
  
  