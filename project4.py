import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

def trainingData(window_size, stride, file_name='beatles.txt'):
    enc = OneHotEncoder(sparse=False)
    enc.n_features_in_ = 45
    words = pd.read_fwf(file_name).values.tolist()
    words = [word[0] for word in words] # List of strings
    x = np.empty((1,5,1))
    y= np.empty((1,5,1))
    for sentence in words:
        numbers = []
        for letter in sentence:
            numbers.append(ord(letter))
        sentence = np.array(numbers).reshape(-1,1)
        temp_x = np.array([sentence[stride*j:stride*j+window_size] for j in range(int((len(sentence)-window_size+1)/stride))])
        temp_y = np.array([sentence[stride*j+1:stride*j+window_size+1] for j in range(int((len(sentence)-window_size+1)/stride))])
        try: # Filters out windows that are dimensionless (weird indexing error)
            x = np.append(x,temp_x,axis=0)
            y = np.append(y,temp_y,axis=0)
        except:
            pass
    x,y = x[1:],y[1:]
    #x_enc = y_enc = np.zeros((x.shape[0],5,45))
    x_enc = enc.fit_transform(x.reshape(-1,1)).reshape(-1,5,45)
    y_enc = enc.fit_transform(y.reshape(-1,1)).reshape(-1,5,45)
    return x_enc,y_enc

x,y = trainingData(window_size=5,stride=3)
