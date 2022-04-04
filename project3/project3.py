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
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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


def FullyConnected(task, lr,num_epochs=10):
  if task=='race':
    n=7
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    dataset_train = tf.data.Dataset.from_tensor_slices((images_train,race_train))
    dataset_train = dataset_train.batch(16)
    dataset_test = tf.data.Dataset.from_tensor_slices((images_test,race_test))
    dataset_test = dataset_test.batch(16)
    #y_train, y_test = race_train, race_test
  elif task=='age':
    n=9
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    dataset_train = tf.data.Dataset.from_tensor_slices((images_train,age_train))
    dataset_train = dataset_train.batch(16)
    dataset_test = tf.data.Dataset.from_tensor_slices((images_test,age_test))
    dataset_test = dataset_test.batch(16)
    #y_train, y_test = age_train, age_test
  elif task=='gender':
    n=1
    loss = 'binary_crossentropy'
    activation = 'sigmoid'
    dataset_train = tf.data.Dataset.from_tensor_slices((images_train,gender_train))
    dataset_train = dataset_train.batch(16)
    dataset_test = tf.data.Dataset.from_tensor_slices((images_test,gender_test))
    dataset_test = dataset_test.batch(16)
    #y_train, y_test = gender_train, gender_test
  Model = tf.keras.Sequential()
  Model.add(tf.keras.layers.Flatten(input_shape=(32,32, 1)))
  Model.add(tf.keras.layers.Dense(1024, activation='tanh'))
  Model.add(tf.keras.layers.Dense(512, activation='sigmoid'))
  Model.add(tf.keras.layers.Dense(100, activation='relu'))
  Model.add(tf.keras.layers.Dense(n, activation=activation))
  Model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                loss=loss,
                metrics=['accuracy'])
  
  history = Model.fit(dataset_train,validation_data=dataset_test, epochs=num_epochs)
  #history = Model.fit(x=images_train, y=y_train, validation_data=(images_test,y_test), batch_size=16, epochs=num_epochs)
  return history

def Convolutional(task, lr, num_epochs=10):
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
  
  history = convModel.fit(dataset_train,validation_data=dataset_test, epochs=num_epochs)
  
  return history

def Convolutional2(task, lr, num_epochs=10):
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
  conv2Model.add(tf.keras.layers.BatchNormalization())
  conv2Model.add(tf.keras.layers.Conv2D(16,3,activation='ReLU'))
  conv2Model.add(tf.keras.layers.BatchNormalization())
  conv2Model.add(tf.keras.layers.Conv2D(32,3,activation='ReLU'))
  conv2Model.add(tf.keras.layers.BatchNormalization())
  conv2Model.add(tf.keras.layers.MaxPool2D())
  conv2Model.add(tf.keras.layers.Conv2D(64,3,activation='ReLU'))
  conv2Model.add(tf.keras.layers.BatchNormalization())
  conv2Model.add(tf.keras.layers.Conv2D(128,3,activation='ReLU'))
  conv2Model.add(tf.keras.layers.BatchNormalization())
  conv2Model.add(tf.keras.layers.MaxPool2D())
  conv2Model.add(tf.keras.layers.Flatten())
  conv2Model.add(tf.keras.layers.Dense(100,activation='ReLU'))
  conv2Model.add(tf.keras.layers.BatchNormalization())
  conv2Model.add(tf.keras.layers.Dense(n,activation=activation))
  conv2Model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                loss=loss,
                metrics=['accuracy'])
  
  history = conv2Model.fit(dataset_train,validation_data=dataset_test, epochs=num_epochs)
  
  return history

def getLabels():
    num = -1
    train_labels = pd.read_csv('fairface_label_train.csv')
    file_train, age_train, gender_train, race_train = train_labels.iloc[:num,0].tolist(), train_labels.iloc[:num,1].to_numpy().reshape(-1,1), train_labels.iloc[:num,2].to_numpy(), train_labels.iloc[:num,3].to_numpy().reshape(-1,1)
    test_labels= pd.read_csv('fairface_label_val.csv')
    file_test, age_test, gender_test, race_test = test_labels.iloc[:num,0].tolist(), test_labels.iloc[:num,1].to_numpy().reshape(-1,1), test_labels.iloc[:num,2].to_numpy(), test_labels.iloc[:num,3].to_numpy().reshape(-1,1)
    
    age_train, age_test = OneHotEncoder(sparse=False).fit_transform(age_train), OneHotEncoder(sparse=False).fit_transform(age_test)
    gender_train, gender_test = LabelEncoder().fit_transform(gender_train), LabelEncoder().fit_transform(gender_test)
    race_train, race_test = OneHotEncoder(sparse=False).fit_transform(race_train), OneHotEncoder(sparse=False).fit_transform(race_test)
    return file_train, file_test, age_train, age_test, gender_train, gender_test, race_train, race_test

def task(num):
    if num==1:
        print(f'Training on Race:\n')
        history_race = FullyConnected('race', 0.001, num_epochs=20)
        print(f'\nTraining on Age:\n')
        history_age = FullyConnected('age', 0.001, num_epochs=20)
        print(f'\nTraining on Gender:\n')
        history_gender = FullyConnected('gender', 0.001, num_epochs=20)
    elif num==2:
        print(f'Training on Race:\n')
        history_race = Convolutional('race', 0.001, num_epochs=20)
        print(f'\nTraining on Age:\n')
        history_age = Convolutional('age', 0.001, num_epochs=20)
        print(f'\nTraining on Gender:\n')
        history_gender = Convolutional('gender', 0.001, num_epochs=20)
    elif num==3:
        print(f'Training on Race:\n')
        history_race = Convolutional2('race', 0.001, num_epochs=20)
        print(f'\nTraining on Age:\n')
        history_age = Convolutional2('age', 0.001, num_epochs=20)
        print(f'\nTraining on Gender:\n')
        history_gender = Convolutional2('gender', 0.001, num_epochs=20)
    #elif num==4:
    
    return history_race, history_age, history_gender
        
file_train, file_test, age_train, age_test, gender_train, gender_test, race_train, race_test = getLabels() # separates processed labels from csv
# Image folders must be stored in another folder in the root directory with project3.py for... some reason.
# I create train_imgs & val_imgs that hold their respective image folders
data_dir_train = r"C:\Unreal Engine\Projects\COSC525\FromHub\project3\train_imgs\\"
data_dir_test = r"C:\Unreal Engine\Projects\COSC525\FromHub\project3\val_imgs\\"

images_train = load_image(file_train, data_dir_train) # get image tensors corresponding to labels
images_test = load_image(file_test, data_dir_test)


# Training, returns accuracy/loss for training/validation
# FC_race, FC_age, FC_gender = task(1)
# Conv_race, Conv_age, Conv_gender = task(2)
Conv2_race, Conv2_age, Conv2_gender = task(3)
  
def graph(history_race,history_age,history_gender, model):
    plt.plot(history_race.history['accuracy'])
    plt.plot(history_race.history['val_accuracy'])
    plt.title('Race (' + model + ')')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history_race.history['loss'])
    plt.plot(history_race.history['val_loss'])
    plt.title('Race (' + model + ')')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()
    plt.plot(history_age.history['accuracy'])
    plt.plot(history_age.history['val_accuracy'])
    plt.title('Age (' + model + ')')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history_age.history['loss'])
    plt.plot(history_age.history['val_loss'])
    plt.title('Age (' + model + ')')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()
    plt.plot(history_gender.history['accuracy'])
    plt.plot(history_gender.history['val_accuracy'])
    plt.title('Gender (' + model + ')')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history_gender.history['loss'])
    plt.plot(history_gender.history['val_loss'])
    plt.title('Gender (' + model + ')')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()
  
# graph(FC_race,FC_age,FC_gender,'FC')
# graph(Conv_race,Conv_age,Conv_gender,'Conv')
# graph(Conv2_race,Conv2_age,Conv2_gender,'Conv2')
  
  
  
  
  
  
  