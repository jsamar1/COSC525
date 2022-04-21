import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from joblib import dump, load
import re

NUMCHARS = 27

def trainingData(window_size, stride, file_name='beatles.txt'):
    global NUMCHARS
    try:
        x = np.load('x_data'+str(window_size)+str(stride)+'.npy')
        y = np.load('y_data'+str(window_size)+str(stride)+'.npy')
        clf = load('enc.joblib')
        print('load')
        NUMCHARS = x.shape[2]
        return x,y,clf
    except:
        enc = OneHotEncoder(sparse=False)
        words = pd.read_fwf(file_name).values.tolist()
        words = [word[0] for word in words] # List of strings
        # vocab = {l for word in words for l in word}
        # print(f'unique chars: {len(vocab)}')

        # string of all lines
        allWords = "\n".join(words)
        allWords = re.sub('[^a-zA-Z!?\',\n ]+', '', allWords)
        vocab = np.array(list({ord(l) for l in allWords})).reshape(-1,1)
        NUMCHARS = len(vocab)
        enc.fit(vocab)
        x = np.empty((1,5,1))
        y= np.empty((1,5,1))
        
        numbers = []
        for char in allWords:
            numbers.append(ord(char))
        
        sentence = np.array(numbers).reshape(-1,1)

        x = np.array([sentence[stride*j:stride*j+window_size] for j in range(int((len(sentence)-window_size+1)/stride))])
        y = np.array([sentence[stride*j+1:stride*j+window_size+1] for j in range(int((len(sentence)-window_size+1)/stride))])
        # try: # Filters out windows that are dimensionless (weird indexing error)
        #     x = np.append(x,temp_x,axis=0)
        #     y = np.append(y,temp_y,axis=0)
        # except:
        #     pass
        # x,y = x[1:],y[1:]
        x_enc = enc.fit_transform(x.reshape(-1,1))
        x_enc = x_enc.reshape(-1,window_size,NUMCHARS)
        y_enc = enc.fit_transform(y.reshape(-1,1)).reshape(-1,window_size,NUMCHARS)
        
        np.save('x_data'+str(window_size)+str(stride),x_enc)
        np.save('y_data'+str(window_size)+str(stride),y_enc)
        dump(enc, 'enc.joblib')
        print('save')
        return x_enc,y_enc,enc

def train(x,y,model,numEpochs,lr,temp=1):
    model = model(temp)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    print(x.shape,y.shape)
    history = model.fit(x,y,epochs=numEpochs,batch_size=64,verbose=1,validation_split=0.2,shuffle=True,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)])
    return history, model

def predict(given,model,numOfChars,temp=1): # currently passing only the last generated sequence
    # model = model(temp)
    input_ = given
    x = given
    for _ in range(numOfChars):

        out = model(x).numpy()[-1].reshape(1,5,NUMCHARS) 
        out = out.T==np.max(out.T,axis=0)
        out = out.T.astype('int')
        # if _ == 0:
        #     outs = temp
        # else:
        #     outs = np.append(outs,temp,axis=0)
        # x = temp
        x = np.append(x,out,axis=0)
    sentence = []
    for i in range(len(x)):
        nums = enc.inverse_transform(x[i])
        for num in nums:
            num = int(num[0])
            letter = chr(num)
            sentence.append(letter)
    return ''.join(sentence)
    

def simpleRNN(temp):
    simpleRNN = keras.Sequential()
    simpleRNN.add(layers.SimpleRNN(units=100, activation='relu', return_sequences=True))

    simpleRNN.add(layers.Dense(units=NUMCHARS, activation='linear'))

    simpleRNN.add(layers.Rescaling(1/temp))
    simpleRNN.add(layers.Softmax())
    return simpleRNN

def LSTM(temp):
    LSTM = keras.Sequential()
    # LSTM.add(layers.LSTM(units=100, activation='tanh', return_sequences=True))
    # LSTM.add(layers.Dropout(0.2))
    LSTM.add(layers.LSTM(units=100, activation='tanh', return_sequences=True))
    LSTM.add(layers.Dropout(0.2))

    LSTM.add(layers.LSTM(units=100, activation='tanh', return_sequences=True))
    LSTM.add(layers.Dropout(0.2))
    LSTM.add(layers.Dense(units=NUMCHARS, activation='linear'))

    LSTM.add(layers.Rescaling(1/temp))
    LSTM.add(layers.Softmax())
    return LSTM

# def trainingData(window_size, stride, file_name='beatles.txt'):
x,y,enc = trainingData(5,3)
# a,b,encoder = trainingData(10, 2)

#hist = train(x,y,simpleRNN,numEpochs=5,lr=0.001)
hist, lModel = train(x,y,LSTM,numEpochs=5,lr=0.003)


pred = predict(x[0].reshape(1,5,NUMCHARS),lModel,10,1)
print(pred)

hist, sModel = train(x,y,simpleRNN,numEpochs=5,lr=0.003)


pred = predict(x[0].reshape(1,5,NUMCHARS),sModel,10,1)
print(pred)