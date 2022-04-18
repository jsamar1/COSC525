import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

def trainingData(window_size, stride, file_name='beatles.txt'):
    try:
        x = np.load('x_data'+str(window_size)+str(stride)+'.npy')
        y = np.load('y_data'+str(window_size)+str(stride)+'.npy')
        return x,y
    except:
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
        x_enc = enc.fit_transform(x.reshape(-1,1)).reshape(-1,5,45)
        y_enc = enc.fit_transform(y.reshape(-1,1)).reshape(-1,5,45)
        np.save('x_data'+str(window_size)+str(stride),x_enc)
        np.save('y_data'+str(window_size)+str(stride),y_enc)
        return x_enc,y_enc

def train(x,y,model,numEpochs,lr):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x,y,epochs=numEpochs,batch_size=32,verbose=1,validation_split=0.2,shuffle=True,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)])
    return history

def predict(given,model,temp,numOfChars):
    x = given
    for _ in range(numOfChars):
        out = model(x)
        x = out
    return x
    
x,y = trainingData(5,3)

simpleRNN = keras.Sequential()
simpleRNN.add(layers.SimpleRNN(units=100, activation='relu', return_sequences=True))
simpleRNN.add(layers.Dense(units=45, activation='softmax'))


hist = train(x,y,simpleRNN,numEpochs=2,lr=0.001)