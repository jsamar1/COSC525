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


# A class which represents a single neuron
class Neuron:
    #initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self,activation, input_num, lr, weights=None):
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.weights = weights
        # if self.weights.any() == None:
        #     self.weights = np.random.rand(1,self.input_num)
        print('constructor 1')    
        
    #This method returns the activation of the net
    def activate(self,net):
        sigmoid = 1/(1+np.exp(-net))
        if self.activation == 0:
            output = net
        elif self.activation == 1:
            output = sigmoid
        else:
            print('Incorrect activation. Choose 0 or 1')
        print('activate')   
        return output

    #Calculate the output of the neuron should save the input and output for back-propagation.   
    def calculate(self,input):
        self.input = input
        output = self.weights.dot(input)        #calculate w*x
        self.output = self.activate(output)     #activate and store output in neuron
        print('calculate 1')
        return self.output

    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):
        if self.activation == 0:
            deriv = 1                           #linear derivative
        if self.activation == 1:
            deriv = self.output*(1-self.output) #sigmoid derivative
        print('activationderivative')
        return deriv
    
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        self.delta = wtimesdelta*self.activationderivative()*np.array(self.input) #dE/do*do/dn*dn/dw = dE/dw
        return self.delta  #saves delta in neuron to use in updateweight()
        print('calcpartialderivative')
    
    #Simply update the weights using the partial derivatives and the leranring weight
    def updateweight(self):
        self.weights = self.weights - self.lr*self.delta
        return self.weights
        print('updateweight')

#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        self.numOfNeurons = numOfNeurons
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.weights = weights
        #Initalize weights if none given
        # if self.weights.any() == None:
        #     self.weights = np.random.rand(self.input_num,self.numOfNeurons)
        self.perceptron = [Neuron(self.activation,self.input_num,self.lr,self.weights[i]) for i in range(self.numOfNeurons)]
        print('constructor 2') 
        
        
    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        self.input = input #saves input in layer
        outputs = []  
        for i in range(self.numOfNeurons):
            perceptron = self.perceptron[i]
            value = perceptron.calculate(input)  #calculates the value of the neuron
            outputs.append(value)
        print('calculate 2')
        return outputs
        
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        delta = 0
        for i in range(self.numOfNeurons):
            perceptron = self.perceptron[i]
            grad = perceptron.calcpartialderivative(wtimesdelta[i])*self.weights[i,:]/self.input  #dE/dw * dn_o/do_h / (dn/dw) = dE/do_h
            delta += grad #sums the gradients for each weight from each neuron 
            self.weights[i,:] = perceptron.updateweight()
        return delta
        print('calcwdeltas') 
           
#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self,numOfLayers,numOfNeurons, inputSize, activation, loss, lr, weights=None):
        self.numOfLayers = numOfLayers
        self.numOfNeurons = numOfNeurons
        self.inputSize = inputSize
        self.activation = activation
        self.loss = loss
        self.lr = lr
        self.weights = weights
        self.layer = [FullyConnected(self.numOfNeurons,self.activation,self.inputSize,self.lr,self.weights[:][:][i]) for i in range(self.numOfLayers)] #instantiate layers
        print('constructor 3')
    
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
                outputs.append(value[:self.numOfLayers]) #appends the vector without the bias since its the last layer
        print('calculate 3')
        return outputs
    
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):
        if self.loss == 0:
            error = 0.5*np.sum((y-yp)**2) #MSE
        if self.loss == 1:
            pass
        return error
        print('calculate loss')
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):
        if self.loss == 0:
            print(yp,y)
            errorderiv = -(y-yp)  #yp is the predicted y values (outputs)
        return errorderiv
        print('lossderiv')
    
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
        print('train')

if __name__=="__main__":
    if (len(sys.argv)<2):
        print('a good place to test different parts of your code')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])     #runs the example from class, uncomment the block to train
        print(w.shape)
        x=np.array([0.05,0.1])  #bias added into x vector
        y = np.array([0.01,0.99])
        model = NeuralNetwork(2,2,2,1,0,0.5,w)
        print(model.train(x,y))
        # errors = []
        # for i in range(100):
        #     model = NeuralNetwork(2,2,2,1,0,0.5,w)
        #     w = model.train(x,y)[1]
        #     errors.append(model.train(x,y)[0])
        # plt.plot(errors)
        
    elif (sys.argv[1]=='example'):
        print('run example from class (single step)')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])     #will put example above here eventually
        x=np.array([0.05,0.1,1])  #bias added into x vector
        model = NeuralNetwork(2,2,2,0,0,0.5,w)
        model.calculate(x)
        np.array([0.01,0.99])
        
    elif(sys.argv[1]=='and'):
        xs = np.array([[0, 0],[0,1],[1,0],[1,1]])
        ys = np.array([[0], [0], [0], [1]])
        w=np.array([[[.15,.2,.25]]]) #wrap in lots of lists for layer/neuron indexing
        errors = []
        for i in range(100):
            suberror = []
            for x,y in zip(xs,ys):
                model = NeuralNetwork(1,1,2,1,0,2,w) #Extremely high learning rate of 2 for fast convergence, iterated manually for optimal lr
                loss, w = model.train(x,y)
                suberror.append(loss)
            errors.append(np.average(suberror))
        plt.plot(errors)
        print('learn and')
        
    elif(sys.argv[1]=='xor'):
        xs = np.array([[0, 0],[0,1],[1,0],[1,1]])
        ys = np.array([[0], [1], [1], [0]])
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]]) #these weights correlate to 2 output neurons when we should have 1 for xor
        #w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45]]]) #network structure is 3x1 -> 2x1 -> 1x1
        #w = np.random.normal(loc=np.sqrt(2/4),size=(2,2,3))    #experimenting with xavier initialization
        errors = []
        for i in range(2):
            suberror = []
            for x,y in zip(xs,ys):
                model = NeuralNetwork(2,2,2,1,0,0.1,w) 
                loss, w = model.train(x,y)
                suberror.append(loss)
            errors.append(np.average(suberror))
        plt.plot(errors)
        
        #predict
        testpt = np.array([0, 0])
        model = NeuralNetwork(2,2,2,1,0,0.1,w) 
        outputs = model.calculate(testpt)
        #print(outputs[-1])
        print('learn xor')