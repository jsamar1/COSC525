import numpy as np
import sys
import matplotlib.pyplot as plt

"""
For this entire file there are a few constants:
activation:
0 - linear
1 - logistic (only one supported)
loss:
0 - sum of square errors
1 - binary cross entropy
"""
LOSS_FUNCTION = 1
ACTIVATION_FUNCTION = 1
EPOCHS_VALUE = 1000
# A class which represents a single neuron
class Neuron:
    #initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self,activation, input_num, lr, weights=None):
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.weights = weights.flatten()
        
    #This method returns the activation of the net
    def activate(self,net):
        sigmoid = 1/(1+np.exp(-net))
        if self.activation == 0:
            output = net
        elif self.activation == 1:
            output = sigmoid
        else:
            print('Incorrect activation. Choose 0 or 1')
        #print('activate')   
        return output

    #Calculate the output of the neuron should save the input and output for back-propagation.   
    def calculate(self,input):
        self.input = input
        output = self.weights.dot(input)        #calculate w*x
        self.output = self.activate(output)     #activate and store output in neuron
        
        return self.output

    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):
        if self.activation == 0:
            deriv = 1                           #linear derivative
        if self.activation == 1:
            deriv = self.output*(1-self.output) #sigmoid derivative
        return deriv
    
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        self.delta = wtimesdelta*self.activationderivative()*np.array(self.input) #dE/do*do/dn*dn/dw = dE/dw
        test = wtimesdelta*self.activationderivative() #dE/do*do/dn = dE/dn
        return self.delta, test  #saves delta in neuron to use in updateweight()
    
    #Simply update the weights using the partial derivatives and the leranring weight
    def updateweight(self):
        self.weights = self.weights - self.lr*self.delta
        return self.weights

#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        self.numOfNeurons = numOfNeurons
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.weights = weights

        if np.size(self.weights) == self.input_num*self.numOfNeurons:  #changes number of neurons in the first layer to be accurate
            self.numOfNeurons = self.input_num
        self.perceptron = [Neuron(self.activation,self.input_num,self.lr,self.weights[i]) for i in range(self.numOfNeurons)]        
        
    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        self.input = input #saves input in layer
        outputs = []  
        for i in range(self.numOfNeurons):
            perceptron = self.perceptron[i]
            value = perceptron.calculate(input)  #calculates the value of the neuron
            outputs.append(value)
        return outputs
        
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        delta = 0
        for i in range(self.numOfNeurons):
            perceptron = self.perceptron[i]
            grad = perceptron.calcpartialderivative(wtimesdelta[i])[1]*self.weights[i,:] #dE/dn*dn/h = dE/dh
            delta += grad #sums the gradients for each weight from each neuron 
            self.weights[i,:] = perceptron.updateweight()
        return delta
 
class ConvolutionalLayer:
    def __init__(self,numOfKernels, kernelSize, activation, inputSize, lr, weights=None):
        self.numOfKernels = numOfKernels
        self.kernelSize = kernelSize
        self.activation = activation
        self.inputSize = inputSize # 3 dimensional
        self.lr = lr
        # if not weights:
        #     weights = np.random.rand(kernalSize,kernelSize,inputSize[2],numOfKernels)
        self.weights = weights
        self.numOfNeurons = ((inputSize[0]-kernelSize+1)**2)
        self.kernels = [] #each sublist holds the neurons of a kernel
        for i in range(numOfKernels):
            self.perceptron = [Neuron(self.activation,self.inputSize,self.lr,self.weights[:,:,:,i]) for j in range(self.numOfNeurons)]
            self.kernels.append(self.perceptron)
            
    def calculate(self,input):
        self.input = input
        window = np.lib.stride_tricks.sliding_window_view(input,(self.kernelSize,self.kernelSize,self.inputSize[2]))
        window = window.reshape(self.numOfKernels,self.numOfNeurons,self.kernelSize**2*self.inputSize[2])
        
        #height = width = self.inputSize[0] - kernelSize + 1 # inputSize - kernelSize + 1
        outputs = []
        for i, kernel in enumerate(self.kernels):
            for j, perceptron in enumerate(kernel):
                #print(window[i].shape)
                x = window[i,j,:].flatten() #
                value = perceptron.calculate(x)
                outputs.append(value)
        return outputs

class MaxPoolingLayer:
    def __init__(self, poolSize, inputSize):
        self.poolSize = poolSize
        self.inputSize = inputSize
        self.numOfNeurons = ((inputSize[0]-poolSize)//poolSize + 1)
        
    def calculate(self,input):
        self.input = input
        window = np.lib.stride_tricks.sliding_window_view(input,(self.poolSize,self.poolSize,self.inputSize[2]))[::self.poolSize,::self.poolSize]
        window = window.reshape(self.numOfNeurons,self.numOfNeurons,self.poolSize**2,self.inputSize[2])
        #window should be indexed with [0,0,:,0] where it is [ith winddow,jth window,:,channel], the : selects all values in the window
        out = np.empty([window.shape[0],window.shape[1],self.inputSize[2]])
        self.idx = np.empty([window.shape[0],window.shape[1],self.inputSize[2]])
        for i in range(window.shape[0]):
            for j in range(window.shape[1]):
                for k in range(self.inputSize[2]):
                    reshaped = window[i,j,:,k].reshape(self.poolSize,self.poolSize)
                    out[i,j,k] = max(window[i,j,:,k])
                    self.idx[i,j,k] = np.argmax(window[i,j,:,k]) #index of max value in flattened window array
        return out    

    def calcwdeltas(self,wdelta):
        # window = np.lib.stride_tricks.sliding_window_view(self.input,(self.poolSize,self.poolSize,self.inputSize[2]),writeable=True)[::self.poolSize,::self.poolSize]
        # window = window.reshape(self.input.shape)
        # print(window.shape)
        # return window
        windowBack = np.empty((self.inputSize))
        for i in range(wdelta.shape[0]):
            for j in range(wdelta.shape[1]):
                for k in range(wdelta.shape[2]):
                    out = np.zeros((self.poolSize**2))
                    index = int(self.idx[i,j,k])
                    out[index] = wdelta[i,j,k]
                    out = out.reshape(self.poolSize,self.poolSize)
                    windowBack[i*self.poolSize:(i+1)*self.poolSize,j*self.poolSize:(j+1)*self.poolSize,k] = out
        return windowBack
        # for i in range(wdelta.shape[0]):
        #     for j in range(wdelta.shape[1]):
        #         for k in range(wdelta.shape[2]):
                    
        #             index = int(self.idx[i,j,k])
                    
        #             self.windowBack[i,j,index,k] = wdelta[i,j,k]
        # self.windowBack = self.windowBack.reshape(self.inputSize)
        # return self.windowBack
        
#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self,inputSize, loss, lr):
        self.inputSize = inputSize
        self.loss = loss
        self.lr = lr
        # self.layer = []
        # for i in range(self.numOfLayers):
        #     try: #handles the case where sys.argv[1] does not exist
        #         if i == (self.numOfLayers-1) and sys.argv[2]=='xor': 
        #             self.numOfNeurons = 1
        #     except:
        #         pass
        #     self.layer.append(FullyConnected(self.numOfNeurons,self.activation,self.inputSize,self.lr,self.weights[i])) #instantiate layers, changing output layer to 1 neuron for xor gate
    
    def addLayer(self,genre):
        if genre == 'Conv':
            layer = ConvolutionalLayer()
        if genre == 'FC':
            layer = FullyConnected(numOfNeurons, activation, input_num, lr)
        
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        input = np.append(input,1)
        outputs = [input]
        for i in range(self.numOfLayers):
            layer = self.layer[i]
            value = layer.calculate(input)
            input = value  #sets input to the next layer as the output to the previous layer           
            input.append(1) #adds the bias node to the input vector
            if i < self.numOfLayers-1:  #adds the bias node to the output vector for all but the last layer
                outputs.append(value[:])
            else:
                outputs.append(value[:self.numOfNeurons]) #appends the vector without the bias since its the last layer
        return outputs
    
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):
        ones = [1] * len(y)
        if self.loss == 0:
            error = 0.5*np.sum((y-yp)**2) #MSE
        if self.loss == 1:
            y = y[0]
            yp = yp[0]
            error = -(y*np.log(yp) + (1-y)*np.log(1-yp))
        return error
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):
        ones = [1] * len(y)
        if self.loss == 0:
            errorderiv = -(y-yp)  #yp is the predicted y values (outputs)
        if self.loss == 1:
            errorderiv = -y/yp + np.subtract(ones, y)/np.subtract(ones, yp)
        return errorderiv
    
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):
        outputs = self.calculate(x)
        error = self.calculateloss(outputs[-1],y)
        errorderiv = self.lossderiv(outputs[-1],y)
        wtimesdelta = errorderiv    #intializing wtimesdelta for the first calculation
        for i in range(self.numOfLayers):
            i = self.numOfLayers - i - 1 #shifts the index so that we move backward through the network
            layer = self.layer[i]
            wdeltas = layer.calcwdeltas(wtimesdelta)
            wtimesdelta = wdeltas
        return [error, self.weights]

if __name__=="__main__":
    if(len(sys.argv) < 2):
        print('a good place to test different parts of your codes')
        weights = np.ones([2,2,3,1]) # [kernelSize,kernelSize,inputSize[2],numOfkernels]
        x = np.arange(1,28).reshape(3,3,3)
        test = ConvolutionalLayer(1, 2, 0, [3,3,3], 0.1, weights)
        
        x = np.arange(1,65).reshape(4,4,4)
        test = MaxPoolingLayer(2,x.shape)
        maxs = test.calculate(x)
        backs = test.calcwdeltas(maxs)
    
        # w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])     #runs the example from class, uncomment the block to train
        # x=np.array([0.05,0.1])
        # y = np.array([0.01,0.99])

        # errors = []
        # model = NeuralNetwork(2,2,2,ACTIVATION_FUNCTION,LOSS_FUNCTION,0.001,w)
        # for i in range(EPOCHS_VALUE):
        #     error, w = model.train(x,y)
        #     errors.append(error)
        # plt.plot(errors)
        # plt.ylabel("Error Value")
        # plt.xlabel("Epochs")
        # plt.title('Example from Class')
        # plt.show()
    elif(sys.argv[2] == 'graphs'):
        learningRates = [.00001, .0001, .001, .01, .05, .1, .2, .3]
        lossFunctions = [0, 1] # 0 == means squared, 1 == binary cross entropy
        plotTitles = ["Mean Square Error Loss", "Binary Cross Entropy Loss"]
        actFuncs = ["Linear Activation", "Logistic Activation"]
        print('Create graphics of different learning rates with different loss functions')


        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])     #will put example above here eventually
        x=np.array([0.05,0.1])
        y = np.array([0.01, 0.99])
        # def __init__(self,numOfLayers,numOfNeurons, inputSize, activation, loss, lr, weights=None):
        for activationFunc in actFuncs:
            for loss in lossFunctions:
                print(f'Act: {activationFunc}')
                print(f'Loss: {loss}')
                if(loss == 1 and activationFunc == 'Linear Activation'):
                    print('skip')
                    continue
                plt.figure(loss)
                plt.title(plotTitles[loss] + " - " + activationFunc)
                plt.ylabel("Error Value")
                plt.xlabel("Epoch")
                for rate in learningRates:
                    errors = []
                    w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])    
                    model = NeuralNetwork(2,2,2,actFuncs.index(activationFunc),loss,rate,w)
                    for i in range(EPOCHS_VALUE):
                        error, w = model.train(x,y)
                        errors.append(error)
                    plt.plot(errors, label=str(rate))
                plt.legend(loc="upper right")
                plt.savefig(plotTitles[loss].lower().replace(" ", "") + "_" + activationFunc)
                plt.show()  
    elif (sys.argv[2]=='example'):
        learningRate = float(sys.argv[1])
        print('run example from class (single step)')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        x=np.array([0.05,0.1])  # bias added into x vector
        y = np.array([0.01, 0.99])
        ACITVATION_FUNCTION = 0 
        LOSS_FUNCTION = 0  # cannot have binary cross entropy with 2 output nodes
        learningRate = 0.5 # matches the learning rate in the example so weights match on backprop
        
        model = NeuralNetwork(2,2,2,ACTIVATION_FUNCTION,LOSS_FUNCTION,learningRate,w)


        # Perform a single step on the example from class
        error, weights = model.train(x,y)
        print(f'Error: {error}\nWeights: {[[w[:-1] for w in we] for we in weights]}')
        print(f'[ [w1, w2], [w3, w4], [w5, w6], [w7, w8] ]')

        errors = []
        model = NeuralNetwork(2,2,2,ACTIVATION_FUNCTION,LOSS_FUNCTION,learningRate,w)
        for i in range(EPOCHS_VALUE):
            error, w = model.train(x,y)
            errors.append(error)
        plt.plot(errors)
        plt.ylabel("Error Value")
        plt.xlabel("Epochs")
        plt.show()
        
    elif(sys.argv[2]=='and'):
        learningRate = float(sys.argv[1])
        xs = np.array([[0, 0],[0,1],[1,0],[1,1]])
        ys = np.array([[0], [0], [0], [1]])
        w=np.array([[[.15,.2,.25]]]) #wrap in lots of lists for layer/neuron indexing
        errors = []
        model = NeuralNetwork(1,1,2,ACTIVATION_FUNCTION,LOSS_FUNCTION,learningRate,w) #Extremely high learning rate of 2 for fast convergence, iterated manually for optimal lr
        for i in range(EPOCHS_VALUE):
            suberror = []
            for x,y in zip(xs,ys):
                loss, w = model.train(x,y)
                suberror.append(loss)
            errors.append(np.average(suberror))

        plt.plot(errors)
        plt.ylabel("Error Value")
        plt.xlabel("Epochs")
        plt.title('AND with single perceptron')
        plt.show()

        print('\nSingle Perceptron for AND')
        testpts = np.array([[0, 0],[0,1],[1,0],[1,1]])
        outputs = []
        for i in range(4):
            out = model.calculate(testpts[i])
            outputs.append(np.mean(out[-1]))
        print(outputs)
        
    elif(sys.argv[2]=='xor'):
        learningRate = float(sys.argv[1])
        xs = np.array([[0, 0],[0,1],[1,0],[1,1]])
        ys = np.array([[0], [1], [1], [0]])
        w1 = np.random.normal(loc=np.sqrt(2/8),size=(1,5,3))[0]      #for networks where input size != number of neurons, you must initialize the first layer weights separately
        w2 = np.random.normal(loc=np.sqrt(2/11),size=(1,5,6))[0]     #with dimensions (numOfLayers,numOfOutputs in next layer,numOfNeurons including bias)
        w=[w1,w2]                                                    #it is a gaussian distribution defined with the xavier initialization
        errors = []
        
        model = NeuralNetwork(2,5,2,ACTIVATION_FUNCTION,LOSS_FUNCTION,learningRate,w) 
        
        for i in range(EPOCHS_VALUE):
            suberror = []
            for x,y in zip(xs,ys):
                loss, w = model.train(x,y)
                suberror.append(loss)
            errors.append(np.average(suberror))
        
        plt.plot(errors)
        plt.ylabel("Error Value")
        plt.xlabel("Epochs")
        plt.title('XOR with 2 layers, 5 hidden units')
        plt.show()
        
        #predict
        print(f'\nHidden layer for XOR')
        testpts = np.array([[0, 0],[0,1],[1,0],[1,1]])
        outputs = []
        for i in range(4):
            out = model.calculate(testpts[i])
            outputs.append(np.mean(out[-1]))
        print(outputs)
        print('If bad accuracy (should be [0, 1, 1, 0]), rerun the example for different weights')
        
        w3=np.array([[[1.2,1.4,1.6]]]) #single perceptron training on xor data
        errors1 = []
        model = NeuralNetwork(1,1,2,ACTIVATION_FUNCTION,LOSS_FUNCTION,learningRate,w3)
        for i in range(EPOCHS_VALUE):
            suberror = []
            for x,y in zip(xs,ys):
                loss, w = model.train(x,y)
                suberror.append(loss)
            errors1.append(np.average(suberror))
        plt.plot(errors1)
        plt.ylabel("Error Value")
        plt.xlabel("Epochs")
        plt.title('XOR with single perceptron')
        plt.show()

        print(f'\nSingle Percepton for XOR')
        outputs = []
        for i in range(4):
            out = model.calculate(testpts[i])
            outputs.append(np.mean(out[-1]))
        print(outputs)
        print('Bad accuracy attributed to high bias and underfitting')
        
