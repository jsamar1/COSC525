# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:14:26 2022

@author: Riley
"""
from pyexpat import model
import numpy as np
import os
import PIL
#from PIL import Image
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from IPython.display import Image
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
from keras import metrics

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
    y_train, y_test = race_train, race_test
  elif task=='age':
    n=9
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    y_train, y_test = age_train, age_test
  elif task=='gender':
    n=1
    loss = 'binary_crossentropy'
    activation = 'sigmoid'
    y_train, y_test = gender_train, gender_test
    
  Model = tf.keras.Sequential()
  Model.add(tf.keras.layers.Flatten(input_shape=(32,32, 1)))
  Model.add(tf.keras.layers.Dense(1024, activation='tanh'))
  Model.add(tf.keras.layers.Dense(512, activation='sigmoid'))
  Model.add(tf.keras.layers.Dense(100, activation='relu'))
  Model.add(tf.keras.layers.Dense(n, activation=activation))
  Model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                loss=loss,
                metrics=['accuracy'])
  
  early_stopping_monitor = tf.keras.callbacks.EarlyStopping(patience=4)
  history = Model.fit(x=images_train, y=y_train, validation_data=(images_test,y_test), batch_size=batch_size, epochs=num_epochs, callbacks=[early_stopping_monitor])

  return history

def Convolutional(task, lr, num_epochs=10):
  if task=='race':
    n=7
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    y_train, y_test = race_train, race_test
  elif task=='age':
    n=9
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    y_train, y_test = age_train, age_test
  elif task=='gender':
    n=1
    loss = 'binary_crossentropy'
    activation = 'sigmoid'
    y_train, y_test = gender_train, gender_test

  convModel = tf.keras.Sequential()
  convModel.add(tf.keras.layers.Conv2D(40,5,activation='ReLU'))
  convModel.add(tf.keras.layers.MaxPool2D())
  convModel.add(tf.keras.layers.Flatten())
  convModel.add(tf.keras.layers.Dense(100,activation='ReLU'))
  convModel.add(tf.keras.layers.Dense(n,activation=activation))
  convModel.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                loss=loss,
                metrics=['accuracy'])
  early_stopping_monitor = tf.keras.callbacks.EarlyStopping(patience=4)
  history = convModel.fit(x=images_train, y=y_train, validation_data=(images_test,y_test), batch_size=batch_size, epochs=num_epochs, callbacks=[early_stopping_monitor])
  
  return history

def Convolutional2(task, lr, num_epochs=10):
  if task=='race':
    n=7
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    y_train, y_test = race_train, race_test
  elif task=='age':
    n=9
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    y_train, y_test = age_train, age_test
  elif task=='gender':
    n=1
    loss = 'binary_crossentropy'
    activation = 'sigmoid'
    y_train, y_test = gender_train, gender_test

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
  
  early_stopping_monitor = tf.keras.callbacks.EarlyStopping(patience=4)
  history = conv2Model.fit(x=images_train, y=y_train, validation_data=(images_test,y_test), batch_size=batch_size, epochs=num_epochs, callbacks=[early_stopping_monitor])
  
  return history

def branchConvolution(lr, num_epochs=10):
  input = tf.keras.Input(shape=(32,32,1), name='image')
  x = tf.keras.layers.Conv2D(40, 5, activation='ReLU')(input)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Flatten()(x)
  y = tf.keras.layers.Dense(100, activation='ReLU')(x)
  z = tf.keras.layers.Dense(100, activation='ReLU')(x)
  branchRace = tf.keras.layers.Dense(7, activation='ReLU')(y)
  branchAge = tf.keras.layers.Dense(9, activation='ReLU')(z)
  model = tf.keras.Model(inputs=input, outputs=[branchRace, branchAge], name="BranchNN")

  model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )

  early_stopping_monitor = tf.keras.callbacks.EarlyStopping(patience=4)
  history = model.fit(x=images_train, y=[race_train, age_train], validation_data=(images_test,[race_test, age_test]), batch_size=batch_size, epochs=num_epochs, callbacks=[early_stopping_monitor])

  return history

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """
    #Extract mean and log of variance
    z_mean, z_log_var = args
    #get batch size and length of vector (size of latent space)
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    #Return sampled number (need to raise var to correct power)
    return z_mean + K.exp(z_log_var) * epsilon

def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae"):
    """Plots labels and MNIST digits as a function of the 2D latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "faces_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 10
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = (n - 1) * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

def VAE(latent_dim, lr, num_epochs=10):
  # build encoder model
  inputs = tf.keras.Input(shape=(32,32,1), name='encoder_input')
  # x = tf.keras.Dense(100, activation='relu', name="encoder_hidden_layer")(latent_inputs)
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu', strides=2, padding='same')(inputs)
  x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', strides=2, padding='same')(x)
  shape = K.int_shape(x)
  x = tf.keras.layers.Flatten()(x)
  # x = tf.keras.layers.

  z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
  z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)

  # use reparameterization trick to push the sampling out as input
  z = tf.keras.layers.Lambda(sampling, name='z')([z_mean, z_log_var])

  # instantiate encoder model
  encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder_output')
  encoder.summary()


  # build decoder model
  latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
  x = tf.keras.layers.Dense(np.prod(shape[1:]))(latent_inputs)
  x = tf.keras.layers.Reshape(shape[1:])(x)
  x = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, activation='relu', strides=2, padding='same')(x)
  x = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, activation='relu', strides=2, padding='same')(x)
  outputs = tf.keras.layers.Conv2D(filters=1, kernel_size=3, activation='sigmoid', padding='same')(x)

  # instantiate decoder model
  decoder = tf.keras.Model(latent_inputs, outputs, name='decoder_output')
  decoder.summary()

  # instantiate VAE model
  outputs = decoder(encoder(inputs)[2])
  vae = tf.keras.Model(inputs, outputs, name='vae')

  #setting loss
  reconstruction_loss = mse(inputs, outputs)
  reconstruction_loss *=1
  kl_loss = K.exp(z_log_var) + K.square(z_mean) - z_log_var - 1
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= 0.001
  vae_loss = K.mean(reconstruction_loss + kl_loss)
  vae.add_loss(vae_loss)
  vae.compile(optimizer='adam')

  early_stopping_monitor = tf.keras.callbacks.EarlyStopping(patience=4)

  history = vae.fit(images_train, epochs=num_epochs, batch_size=batch_size, validation_data=(images_test, None), callbacks=[early_stopping_monitor])

  return history, vae, (encoder, decoder)

def getLabels(num=5000):
    train_labels = pd.read_csv('fairface_label_train.csv')
    file_train, age_train, gender_train, race_train = train_labels.iloc[:num,0].tolist(), train_labels.iloc[:num,1].to_numpy().reshape(-1,1), train_labels.iloc[:num,2].to_numpy(), train_labels.iloc[:num,3].to_numpy().reshape(-1,1)
    test_labels= pd.read_csv('fairface_label_val.csv')
    file_test, age_test, gender_test, race_test = test_labels.iloc[:num,0].tolist(), test_labels.iloc[:num,1].to_numpy().reshape(-1,1), test_labels.iloc[:num,2].to_numpy(), test_labels.iloc[:num,3].to_numpy().reshape(-1,1)
    
    age_train, age_test = OneHotEncoder(sparse=False).fit_transform(age_train), OneHotEncoder(sparse=False).fit_transform(age_test)
    gender_train, gender_test = LabelEncoder().fit_transform(gender_train), LabelEncoder().fit_transform(gender_test)
    race_train, race_test = OneHotEncoder(sparse=False).fit_transform(race_train), OneHotEncoder(sparse=False).fit_transform(race_test)
    return file_train, file_test, age_train, age_test, gender_train, gender_test, race_train, race_test

def task(num,lr=0.001,num_epochs=10):
    if num==1:
        print(f'Training on Race:\n')
        history_race = FullyConnected('race', lr, num_epochs=num_epochs)
        print(f'\nTraining on Age:\n')
        history_age = FullyConnected('age', lr, num_epochs=num_epochs)
        print(f'\nTraining on Gender:\n')
        history_gender = FullyConnected('gender', lr, num_epochs=num_epochs)
    elif num==2:
        print(f'Training on Race:\n')
        history_race = Convolutional('race', lr, num_epochs=num_epochs)
        print(f'\nTraining on Age:\n')
        history_age = Convolutional('age', lr, num_epochs=num_epochs)
        print(f'\nTraining on Gender:\n')
        history_gender = Convolutional('gender', lr, num_epochs=num_epochs)
    elif num==3:
        print(f'Training on Race:\n')
        history_race = Convolutional2('race', lr, num_epochs=num_epochs)
        print(f'\nTraining on Age:\n')
        history_age = Convolutional2('age', lr, num_epochs=num_epochs)
        print(f'\nTraining on Gender:\n')
        history_gender = Convolutional2('gender', lr, num_epochs=num_epochs)
    elif num==4:
      print(f'Training on Race and Age:\n')
      history_race = branchConvolution(lr, num_epochs=num_epochs)
      return history_race
    elif num==5:
      latent_dim = 5

      history, vae, models = VAE(latent_dim, lr, num_epochs)
      plot_results(models, images_train, batch_size)
      return history, vae
      pass
    
    return history_race, history_age, history_gender
        
# Image folders must be stored in another folder in the root directory with project3.py for... some reason.
# I create train_imgs & val_imgs that hold their respective image folders
# data_dir_train = r"C:\Unreal Engine\Projects\COSC525\FromHub\project3\train_imgs\\"
# data_dir_test = r"C:\Unreal Engine\Projects\COSC525\FromHub\project3\val_imgs\\"
data_dir_train = r"C:\Users\jsamar1\Downloads\project3_COSC525\\"
data_dir_test = r"C:\Users\jsamar1\Downloads\project3_COSC525\\"

file_train, file_test, age_train, age_test, gender_train, gender_test, race_train, race_test = getLabels(200) # separates processed labels from csv
images_train = load_image(file_train, data_dir_train) # get image tensors corresponding to labels
images_test = load_image(file_test, data_dir_test)


# Training, returns accuracy/loss for training/validation
batch_size = 128
lr = 0.01
num_epochs = 100
FC_race, FC_age, FC_gender = task(1, lr, num_epochs)
Conv_race, Conv_age, Conv_gender = task(2, lr, num_epochs)
Conv2_race, Conv2_age, Conv2_gender = task(3, lr, num_epochs)
branch_race_and_age = task(4, lr, num_epochs)
# history, model = task(5, lr, num_epochs)


def graphGeneral(history_race, model, filename, title):
  # loss graphic
  for key in history_race.history.keys():
    if('val' in key or 'accuracy' in key): continue

    plt.plot(history_race.history[key], label=key)
    plt.plot(history_race.history['val_' + key], label='validation_' + key)
    plt.title(filename + ' (' + title + ')')
    plt.xlabel('Epochs')
    plt.ylabel(key)
  plt.legend(loc='upper left')
  plt.savefig(filename+ '-loss' + title)
  plt.clf()

  # accuracy graphic 
  for key in history_race.history.keys():
    if('val' in key or 'loss' in key): continue

    plt.plot(history_race.history[key], label=key)
    plt.plot(history_race.history['val_' + key], label='validation_' + key)
    plt.title(filename + ' (' + title + ')')
    plt.xlabel('Epochs')
    plt.ylabel(key)

  plt.legend(loc='upper left')
  plt.savefig(filename + '-accuracy' + title)
  plt.clf()

def graph(history_race,history_age,history_gender, model):
    plt.plot(history_race.history['accuracy'], label='race accuracy')
    plt.plot(history_race.history['val_accuracy'], label='race val_accuracy')
    plt.title('Race (' + model + ')')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.savefig(model + 'race')
    # plt.clf()
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
  
graphGeneral(FC_race, 'FC', 'task1', 'Race')
graphGeneral(FC_age, 'FC', 'task1', 'Age')

graphGeneral(Conv_race, 'Conv', 'task2', 'Race')
graphGeneral(Conv_age, 'Conv', 'task2', 'Age')

graphGeneral(Conv2_race, 'Conv2d', 'task3', 'Race')
graphGeneral(Conv2_age, 'Conv2d', 'task3', 'Age')

graphGeneral(branch_race_and_age, 'BranchConv', 'task4', 'Race and Age')

print('here')
  
  
  
  
  
  