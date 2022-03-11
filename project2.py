import numpy as np
import sys
import matplotlib.pyplot as plt
from parameters import generateExample1, generateExample2, generateExample3
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
    def __init__(self,activation, input_num, lr, weights, b=None):
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.weights = weights.flatten()
        if b != None:
            self.b = b
        else:
            self.b = 0 #placeholder value to return for FC layers where b is included in self.weights
        
    #This method returns the activation of the net
    def activate(self,net):
        sigmoid = 1/(1+np.exp(-net))
        if self.activation == 0:
            self.output = net
        elif self.activation == 1:
            self.output = sigmoid
        else:
            print('Incorrect activation. Choose 0 or 1')
        #print('activate')   
        return self.output

    #Calculate the output of the neuron should save the input and output for back-propagation.   
    def calculate(self,input):
        self.input = input
        self.output = self.weights.dot(input) + self.b       #calculate w*x+b
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
        self.deltaB = wtimesdelta*self.activationderivative()*1 #dE/do*do/dn*dn/db
        grad = wtimesdelta*self.activationderivative() #dE/do*do/dn = dE/dn
        return grad  #saves delta in neuron to use in updateweight()
    
    #Simply update the weights using the partial derivatives and the leranring weight
    def updateweight(self):
        self.weights = self.weights - self.lr*self.delta
        if self.b != 0: 
            self.b = self.b - self.lr*self.deltaB
        return self.weights,self.deltaB

    def get_bias(self):
        return self.b

#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        self.numOfNeurons = numOfNeurons
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.weights = weights
        self.perceptron = [Neuron(self.activation,self.input_num,self.lr,self.weights[i]) for i in range(self.numOfNeurons)]        
        
    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        input = np.append(input,1) # adds bias node
        self.input = input #saves input in layer
        outputs = []  
        for i in range(self.numOfNeurons):
            perceptron = self.perceptron[i]
            value = perceptron.calculate(input)  #calculates the value of the neuron
            value = perceptron.activate(value)   # activation
            outputs.append(value)
        return outputs
        
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        delta = 0
        for i in range(self.numOfNeurons):
            perceptron = self.perceptron[i]
            grad = perceptron.calcpartialderivative(wtimesdelta[i])*self.weights[i,:] #dE/dn*dn/h = dE/dh
            delta += grad #sums the gradients for each weight from each neuron 
            self.weights[i,:] = perceptron.updateweight()[0]
        return delta

    def get_weights(self):
        return self.weights

    def get_bias(self, index=None):
        return self.perceptron[0].get_bias()
 
class ConvolutionalLayer:
    def __init__(self,numOfKernels, kernelSize, activation, inputSize, lr, weights=None,b=None):
        self.numOfKernels = numOfKernels
        self.kernelSize = kernelSize
        self.activation = activation
        self.inputSize = inputSize # 3 dimensional
        self.lr = lr
        self.weights = weights
        self.b = b
        self.numOfNeurons = ((inputSize[0]-kernelSize+1)**2)
        self.kernels = [] #each sublist holds the neurons of a kernel
        for i in range(numOfKernels):
            self.perceptron = [Neuron(self.activation,self.inputSize,self.lr,self.weights[:,:,:,i],self.b[i]) for j in range(self.numOfNeurons)]
            self.perceptron = np.array(self.perceptron).reshape((inputSize[0]-kernelSize+1),(inputSize[0]-kernelSize+1)) # reshape perceptrons into grid corresponding to window
            self.kernels.append(self.perceptron)
            
    def calculate(self,input):
        self.input = input
        height = width = self.inputSize[0] - self.kernelSize + 1
        window = np.lib.stride_tricks.sliding_window_view(input,(self.kernelSize,self.kernelSize,self.inputSize[2])) # returns window view of input
        window = window.reshape(height,width,self.kernelSize**2*self.inputSize[2]) # reshape into 2-D grid corresponding to perceptron grid, with depth dimension = values in window
        outputs = np.empty((height,width,self.numOfKernels)) 
        for i in range(height):
            for j in range(width):
                for k in range(self.numOfKernels):
                    section = self.kernels[k] # gets perceptron in kernel
                    perceptron = section[i,j] # gets perceptron
                    x = window[i,j,:]         # gets values in window for perceptorn
                    value = perceptron.calculate(x)# w*X + b
                    value = perceptron.activate(value)
                    outputs[i,j,k] = value
        return outputs
    
    def calcwdeltas(self,wdelta):
        delta = np.zeros((self.inputSize[0],self.inputSize[1],self.inputSize[2],self.numOfKernels))
        grad_b = np.zeros((self.b.shape))
        height = width = wdelta.shape[0]
        for i in range(height):
            for j in range(width):
                for k in range(self.numOfKernels):
                    wdeltaSection = wdelta[i,j,k] # gets wdelta for each perceptron
                    section = self.kernels[k]     # gets perceptrons in kernel
                    perceptron = section[i,j]     # gets perceptron
                    perceptron.weights = self.weights[:,:,:,k].flatten()
                    grad = perceptron.calcpartialderivative(wdeltaSection)*self.weights[:,:,:,k] # dE/dn*dn/dh = dE/dh
                    grad = grad.reshape(self.kernelSize,self.kernelSize,self.inputSize[2]) # reshape into window view
                    grad_b[k] += wdeltaSection*1 # gradient of bias is always wdelta
                    delta[i:i+self.kernelSize,j:j+self.kernelSize,:, k] += grad #add dE/dh to delta to give proper summation
                    weights,self.b[k] = perceptron.updateweight() 
                    self.weights[:,:,:,k] = weights.reshape(self.weights[:,:,:,k].shape) # reshapes weights and store them in layer
        return delta

    def get_weights(self):
        return self.weights
        
    def get_bias(self, kernelIndex=0):
        return self.kernels[kernelIndex][0][0].get_bias()

class MaxPoolingLayer:
    def __init__(self, poolSize, inputSize):
        self.poolSize = poolSize
        self.inputSize = inputSize
        self.numOfNeurons = ((inputSize[0]-poolSize)//poolSize + 1)
        
    def calculate(self,input):
        self.input = input
        window = np.lib.stride_tricks.sliding_window_view(input,(self.poolSize,self.poolSize,self.inputSize[2]))[::self.poolSize,::self.poolSize] # window view with non overlapping windows using index stepping
        window = window.reshape(self.numOfNeurons,self.numOfNeurons,self.poolSize**2,self.inputSize[2])
        #window should be indexed [ith window,jth window,:,channel], the : selects all values in the window
        out = self.idx = np.empty([self.numOfNeurons,self.numOfNeurons,self.inputSize[2]]) #stores output and index of max in grid of output size
        for i in range(self.numOfNeurons):
            for j in range(self.numOfNeurons):
                for k in range(self.inputSize[2]):
                    out[i,j,k] = max(window[i,j,:,k])
                    self.idx[i,j,k] = np.argmax(window[i,j,:,k]) #index of max value in flattened window array
        return out    

    def calcwdeltas(self,wdelta):
        windowBack = np.empty((self.inputSize))
        for i in range(self.numOfNeurons):
            for j in range(self.numOfNeurons):
                for k in range(wdelta.shape[2]):
                    out = np.zeros((self.poolSize**2))
                    index = int(self.idx[i,j,k])
                    out[index] = wdelta[i,j,k]
                    out = out.reshape(self.poolSize,self.poolSize)
                    windowBack[i*self.poolSize:(i+1)*self.poolSize,j*self.poolSize:(j+1)*self.poolSize,k] = out # put window with wdelta back into empty array
        return windowBack

class FlattenLayer:
    def __init__(self,inputSize):
        self.inputSize = [int(num) for num in inputSize]
        
    def calculate(self,input):
        self.input = input
        return input.flatten()
    
    def calcwdeltas(self,wdelta):
        wdelta = wdelta[:-1] # removes bias wdelta
        return wdelta.reshape(self.inputSize)
        
#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self,inputSize, loss, lr):
        self.inputSize = inputSize
        self.loss = loss
        self.lr = lr
        self.layer = []
        
    def addConv(self,numOfKernels,kernelSize,activation,lr,weights=None, b=None):
        if np.size(weights) == 1:
            if weights == None:
                weights = np.random.rand(kernelSize,kernelSize,self.inputSize[2],numOfKernels)
        if np.size(b) == 1:
            if b == None:
                    b = np.random.rand((numOfKernels))
        self.layer.append(ConvolutionalLayer(numOfKernels, kernelSize, activation, self.inputSize, lr, weights, b))
        self.inputSize = [self.inputSize[0]-kernelSize+1,self.inputSize[0]-kernelSize+1,numOfKernels] # sets input size for next layer
        
    def addFC(self,numOfNeurons, activation, lr, weights=None):
        self.inputSize = int(self.inputSize)
        if np.size(weights) == 1:
            if weights == None:
                weights = np.random.rand(1,self.inputSize+1)
        input_num = self.inputSize
        self.layer.append(FullyConnected(numOfNeurons, activation, input_num, lr, weights))
        self.inputSize = numOfNeurons # sets input size for next layer
        
    def addMaxPool(self, poolSize):
        self.layer.append(MaxPoolingLayer(poolSize, self.inputSize))
        outputSize = [self.inputSize[0]/poolSize,self.inputSize[1]/poolSize,self.inputSize[2]]
        self.inputSize = outputSize # sets input size for next layer
        
    def addFlattenLayer(self):
        self.layer.append(FlattenLayer(self.inputSize))
        self.inputSize = self.inputSize[0]*self.inputSize[1]*self.inputSize[2] # sets input size for next layer
        
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        for i in range(len(self.layer)):
            layer = self.layer[i]
            value = layer.calculate(input)
            input = value  #sets input to the next layer as the output to the previous layer           
        return value
    
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
            errorderiv = -2*(y-yp)  #yp is the predicted y values (outputs)
        if self.loss == 1:
            errorderiv = -y/yp + np.subtract(ones, y)/np.subtract(ones, yp)
        return errorderiv
    
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):
        output = self.calculate(x)
        error = self.calculateloss(output,y)
        errorderiv = self.lossderiv(output,y)
        wtimesdelta = errorderiv    #intializing wtimesdelta for the first calculation
        for i in range(len(self.layer)):
            i = len(self.layer) - i - 1 #shifts the index so that we move backward through the network
            layer = self.layer[i]
            wtimesdelta = layer.calcwdeltas(wtimesdelta)
        output = self.calculate(x)
        error = self.calculateloss(output,y)
        return [error, output]

    def get_weights(self):
        weights = [0]*len(self.layer)
        for i,layer in enumerate(self.layer):
            if(isinstance(layer, ConvolutionalLayer) or isinstance(layer, FullyConnected)):
                weights[i] = layer.get_weights()
        return np.array(weights, dtype=object)
    
    def get_bias(self, layerIndex, kernelIndex=0):
        return self.layer[layerIndex].get_bias(kernelIndex)

if __name__=="__main__":
    if(len(sys.argv) < 2):
        print('a good place to test different parts of your codes')
        
        print('\nExample 1')
        weights0,b0,weights2,input,output = generateExample1()
        input = np.expand_dims((input),axis=2)
        # weights2 = weights2.reshape(10,1)
        # b2 = weights2[-1]
        # weights2 = weights2[:-1]

        net = NeuralNetwork([5,5,1], 0, 100)
        net.addConv(1, 3, 1, 100, weights0, b0)
        net.addFlattenLayer()
        net.addFC(1, 1, 1, weights2)

        outBefore = net.calculate(input)
        error,out = net.train(input,output)
        outAfter = net.calculate(input)

        print(f'model output before:\n{outBefore}')
        print(f'\nmodel output after:\n{outAfter}')
        t = net.get_weights()
        print(f'\n1st convolution layer, 1st kernel weights:')
        print(f'{t[0][0,:,:,0][:,0]}')
        print(f'{t[0][1,:,:,0][:,0]}')
        print(f'{t[0][2,:,:,0][:,0]}')
        print(f'\n1st convolution layer, 1st kernel bias:\n{net.get_bias(0,0)}')
        print(f'\nFully Connected layer weights:')
        print(f'{t[2][0][:9]}')
        print(f'\nFully Connected layer bias:')
        print(f'{t[2][0][-1]}')
        
        print('\nExample 2')
        l1k1,l1k2,l1b1,l1b2,l2c1,l2c2,l2b,l3,l3b,input, output = generateExample2()
        l1k1 = np.expand_dims(l1k1,axis=(2,3))
        l1k2 = np.expand_dims(l1k2,axis=(2,3))
        weights0 = np.concatenate((l1k1,l1k2),axis=3)
        b0 = np.concatenate((l1b1,l1b2),axis=0)
        l2c1 = np.expand_dims(l2c1,axis=(2,3))
        l2c2 = np.expand_dims(l2c2,axis=(2,3))
        b1 = l2b
        weights1 = np.concatenate((l2c1,l2c2),axis=2)
        l3b = np.expand_dims((l3b),axis=0)
        weights3 = np.concatenate((l3,l3b),axis=1)
        input = np.expand_dims((input),axis=2)
        
        net = NeuralNetwork([7,7,1], 0, 100)
        net.addConv(2, 3, 1, 100, weights0, b0)
        net.addConv(1, 3, 1, 100, weights1, b1)
        net.addFlattenLayer()
        net.addFC(1, 1, 100, weights3)

        outBefore = net.calculate(input)
        error,out = net.train(input,output)
        outAfter = net.calculate(input)

        print(f'model output before:\n{outBefore}')
        print(f'model output after:\n{outAfter}')

        t = net.get_weights()
        print(f'\n1st convolution layer, 1st kernel weights:')
        print(f'{t[0][0,:,:,0][:,0]}')
        print(f'{t[0][1,:,:,0][:,0]}')
        print(f'{t[0][2,:,:,0][:,0]}')
        print(f'\n1st convolution layer, 1st kernel bias:\n{net.get_bias(0,0)}')
        print(f'\n1st convolution layer, 2nd kernel weights:') 

        print(f'{t[0][0,:,:,1][:,0]}')
        print(f'{t[0][1,:,:,1][:,0]}')
        print(f'{t[0][2,:,:,1][:,0]}')
        print(f'\n1st convolution layer, 2nd kernel bias:\n{net.get_bias(0,1)}')
        print(f'\n2nd convolution layer weights:')
        print(f'{t[1][0,:,:,0][:,0]}')
        print(f'{t[1][1,:,:,0][:,0]}')
        print(f'{t[1][2,:,:,0][:,0]}')
        print(f'{t[1][0,:,:,0][:,1]}')
        print(f'{t[1][1,:,:,0][:,1]}')
        print(f'{t[1][2,:,:,0][:,1]}')
        print(f'\n2nd convolution layer bias:\n{net.get_bias(1,0)}')
        print(f'\nFully Connected layer weights:')
        print(f'{t[3][0][:9]}')
        print(f'\nFully Connected layer bias:')
        print(f'{t[3][0][-1]}')

        print('\nExample 3')
        l1k1,l1k2,l1b1,l1b2,l3,l3b,input,output = generateExample3()
        
        l1k1 = np.expand_dims(l1k1,axis=(2,3))
        l1k2 = np.expand_dims(l1k2,axis=(2,3))
        weights0 = np.concatenate((l1k1,l1k2),axis=3)
        b0 = np.concatenate((l1b1,l1b2),axis=0)
        input = np.expand_dims((input),axis=2)
        l3b = np.expand_dims((l3b),axis=0)
        weights3 = np.concatenate((l3,l3b),axis=1)
        
        net = NeuralNetwork([8,8,1], 0, 100) # Example 3, incredibly sensitive to LR and weight initializiation.
        net.addConv(2,3,1,100, weights0, b0)
        net.addMaxPool(2)
        net.addFlattenLayer()
        net.addFC(1, 1, 100, weights3)

        outBefore = net.calculate(input)
        error,out = net.train(input,output)
        outAfter = net.calculate(input)

        print(f'model output before:\n{outBefore}')
        print(f'\nmodel output after:\n{outAfter}')


        t = net.get_weights()
        print(f'\n1st convolution layer, 1st kernel weights:')
        print(f'{t[0][0,:,:,0][:,0]}')
        print(f'{t[0][1,:,:,0][:,0]}')
        print(f'{t[0][2,:,:,0][:,0]}')
        print(f'\n1st convolution layer, 1st kernel bias:\n{net.get_bias(0,0)}')
        print(f'\n1st convolution layer, 2nd kernel weights:') 
        print(f'{t[0][0,:,:,1][:,0]}')
        print(f'{t[0][1,:,:,1][:,0]}')
        print(f'{t[0][2,:,:,1][:,0]}')
        print(f'\n1st convolution layer, 2nd kernel bias:\n{net.get_bias(0,1)}')
        print(f'\nFully Connected layer weights:')
        print(f'{t[3][0][:9]}')
        print(f'\nFully Connected layer bias:')
        print(f'{t[3][0][-1]}')
        
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
        model = NeuralNetwork(2,0,2,ACTIVATION_FUNCTION,LOSS_FUNCTION,learningRate,w)
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
        
