import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from joblib import dump, load
import re
import sys


NUMCHARS = 27

def trainingData(window_size, stride, file_name='beatles.txt'):
    global NUMCHARS
    try:
        x = np.load('x_data'+str(window_size)+str(stride)+'.npy') # Load x data
        y = np.load('y_data'+str(window_size)+str(stride)+'.npy') # Load y data
        clf = load('enc.joblib') # Load encoder
        print('load')
        NUMCHARS = x.shape[2] # Set number of chars
        return x,y,clf
    except:
        enc = OneHotEncoder(sparse=False)
        words = pd.read_fwf(file_name).values.tolist() # Read file
        words = [word[0] for word in words] # List of strings
        # vocab = {l for word in words for l in word}
        # print(f'unique chars: {len(vocab)}')

        # string of all lines
        allWords = "\n".join(words)
        #allWords = re.sub('[^a-zA-Z!?\',\n ]+', '', allWords)
        removeChars = ['6', '8', '2', '3', '4', '7', ':','5','1','0','9','!','?'] # Remove these chars
        allWords = allWords.translate({ord(x): '' for x in removeChars}) # Remove chars
        vocab = np.array(list({ord(l) for l in allWords})).reshape(-1,1) # List of unique chars
        NUMCHARS = len(vocab)
        enc.fit(vocab) # Encode chars
        
        numbers = []
        for char in allWords:
            numbers.append(ord(char)) # List of ords
        
        sentence = np.array(numbers).reshape(-1,1) # Convert to array

        x = np.array([sentence[stride*j:stride*j+window_size] for j in range(int((len(sentence)-window_size+1)/stride))]) # Create x data
        y = np.array([sentence[stride*j+1:stride*j+window_size+1] for j in range(int((len(sentence)-window_size+1)/stride))]) # Create y data

        x_enc = enc.fit_transform(x.reshape(-1,1)) # Encode x data
        x_enc = x_enc.reshape(-1,window_size,NUMCHARS) 
        y_enc = enc.fit_transform(y.reshape(-1,1)).reshape(-1,window_size,NUMCHARS) # Encode y data
        
        np.save('x_data'+str(window_size)+str(stride),x_enc) # Save x data
        np.save('y_data'+str(window_size)+str(stride),y_enc) # Save y data
        dump(enc, 'enc.joblib') # Save encoder
        print('save')
        return x_enc,y_enc,enc

def train(x,y,model,numEpochs,lr,hiddenSize,temp=1):
    model = model(hiddenSize,temp) # Create model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy']) 
    history = model.fit(x,y,validation_split=0.2,epochs=numEpochs,verbose=1,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)]) # Train model
    return history, model

def predict(given,model,numOfChars):
    x = given[0][-1].reshape(1, 1, NUMCHARS)
    for _ in range(numOfChars):
        out = model(x.reshape(1,-1,NUMCHARS), training=False).numpy()[-1].reshape(1,-1,NUMCHARS)  # Predict next window of characters
        target = out[0][-1].reshape(1,-1,NUMCHARS)
        target = target.T==np.max(target.T,axis=0) # Convert to one-hot
        target = target.T.astype('int') # Convert to int
        out = out.T==np.max(out.T,axis=0) # Convert to one-hot
        out = out.T.astype('int') # Convert to int
        x = np.append(x,target,axis=0) # Add to input

    sentence = []
    for i in range(len(x)):
        nums = enc.inverse_transform(x[i]) # Convert back to ords
        for num in nums:
            num = int(num[0]) 
            letter = chr(num) # Convert to char
            sentence.append(letter) # Add to sentence
    return ''.join(sentence) # Return sentence
    
def graphGeneral(history, model, filename, title):
  # loss graphic
  for key in history.history.keys():
    if('val' in key or 'accuracy' in key): continue

    plt.plot(history.history[key], label=key)
    plt.plot(history.history['val_' + key], label='validation_' + key)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(key)
  plt.legend(loc='upper left')
  plt.savefig(filename+ '-loss')
  plt.clf()

def simpleRNN(hiddenSize,temp):
    simpleRNN = keras.Sequential()
    simpleRNN.add(layers.SimpleRNN(units=hiddenSize, activation='relu', return_sequences=True)) 

    simpleRNN.add(layers.Dense(units=NUMCHARS, activation='linear')) 

    simpleRNN.add(layers.Rescaling(1/temp))
    simpleRNN.add(layers.Softmax())
    return simpleRNN

def LSTM(hiddenSize,temp):
    LSTM = keras.Sequential()
    LSTM.add(layers.LSTM(units=hiddenSize, activation='tanh', return_sequences=True))
    LSTM.add(layers.Dropout(0.2))

    LSTM.add(layers.Dense(units=NUMCHARS, activation='linear'))

    LSTM.add(layers.Rescaling(1/temp))
    LSTM.add(layers.Softmax())
    return LSTM

# Define command line argument execution
if len(sys.argv) > 1:
    file = sys.argv[1]
    modelType = sys.argv[2]
    hiddenSize = int(sys.argv[3])
    windowSize = int(sys.argv[4])
    stride = int(sys.argv[5])
    temp = int(sys.argv[6])

    # Load and parse data
    x,y,enc = trainingData(windowSize, stride, file)

    # Train model
    if modelType == 'simple':
        hist, Model = train(x,y,simpleRNN,numEpochs=50,lr=0.001,hiddenSize=hiddenSize,temp=temp)
    elif modelType == 'lstm':
        hist, Model = train(x,y,LSTM,numEpochs=50,lr=0.001,hiddenSize=hiddenSize,temp=temp)
    else:
        print('Invalid model type')
        sys.exit()

    # Initialize prediction with first x window
    pred1 = predict(x[0].reshape(1,-1,NUMCHARS),Model,150)
    print(pred1)
    # pred2 = predict(x[15].reshape(1,-1,NUMCHARS),Model,150)
    # pred3 = predict(x[67].reshape(1,-1,NUMCHARS),Model,150)
    # # print(pred[9::10])

    # with open(str(modelType)+str(hiddenSize)+str(windowSize)+'2.txt', 'w') as f:
    #     f.write(pred1)
    #     f.write('\n----------------------------------------------------------------\n')
    #     f.write(pred2)
    #     f.write('\n----------------------------------------------------------------\n')
    #     f.write(pred3)

    
    graphGeneral(hist, Model, str(modelType)+str(hiddenSize)+str(windowSize), "Epoch-Loss")

else: # If command line arguments are not given
    # def trainingData(window_size, stride, file_name='beatles.txt'):
    x,y,enc = trainingData(10,5)

    # a,b,encoder = trainingData(10, 2)

    #hist = train(x,y,simpleRNN,numEpochs=5,lr=0.001)
    hist, lModel = train(x,y,LSTM,numEpochs=15,lr=0.003, hiddenSize=100)


    pred = predict(x[0].reshape(1,-1,NUMCHARS),lModel,100)
    print(pred)
    # print('\n\n')


    # hist, sModel = train(x,y,simpleRNN,numEpochs=55,lr=0.003,hiddenSize=100)


    # pred = predict(x[0].reshape(1,-1,NUMCHARS),sModel,150)
    # print(pred)