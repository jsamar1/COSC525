import numpy as np
import sys
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
        if self.weights.any() == None:
            self.weights = np.random.rand(1,self.input_num)
        print('constructor 1')    
        
    #This method returns the activation of the net
    def activate(self,net):
        self.net = net
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
        output = self.weights.dot(self.input)
        output = self.activate(output)
        print('calculate 1')
        return output

    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):
        if self.activation == 0:
            deriv = 1
        if self.activation == 1:
            deriv = self.net*(1-self.net)
        print('activationderivative')   
        return deriv
    
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        return self.input #the derivative w.r.t. w for w*X is X (the input)
        print('calcpartialderivative') 
    
    #Simply update the weights using the partial derivatives and the leranring weight
    def updateweight(self):
        self.weights = self.weights - self.lr*self.input
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
        if self.weights.any() == None:
            self.weights = np.random.rand(self.input_num,self.numOfNeurons)
        print('constructor 2') 
        
        
    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        outputs = []
        #print(self.numOfNeurons)
        for i in range(self.numOfNeurons):
            perceptron = Neuron(self.activation,self.input_num,self.lr,self.weights[i,:])
            #calculates the value of the neuron
            value = perceptron.calculate(input)
            outputs.append(value)
        #print(outputs)
        print('calculate 2')
        return outputs
        
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        # delta = np.zeros_like(self.weights) #initalizing the delta variable to be the same shape as the weights
        # for i in range(self.numOfNeurons):
        #     perceptron = Neuron(self.activation,self.input_num,self.lr,self.weights[i,:])
        #     delta += perceptron.calcpartialderivative()  #adds the delta of each neuron together
        delta = []
        for i in range(self.numOfNeurons):
             perceptron = Neuron(self.activation,self.input_num,self.lr,self.weights[i,:])
             delta.append(perceptron.calcpartialderivative(0))  #I don't think the wtimesdelta argument is used? subbed for 0
             perceptron.updateweight()
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
        if self.weights.any() == None:
            self.weights = np.random.rand(self.inputSize,self.numOfNeurons,self.numOfLayers)
        print('constructor 3')
    
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        outputs = []
        for i in range(self.numOfLayers):
            layer = FullyConnected(self.numOfNeurons,self.activation,self.inputSize,self.lr,self.weights[:][:][i])
            value = layer.calculate(input)
            outputs.append(list(value))
            input = value  #sets input to the next layer as the output to the previous layer           
            input.append(1) #adds the bias node to the input vector
            layer.calcwdeltas(0)
        print('calculate 3')
        #print(outputs)
        return outputs
    
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):
        print('calculate loss')
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):
        print('lossderiv')
    
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):
        print('train')

if __name__=="__main__":
    if (len(sys.argv)<2):
        print('a good place to test different parts of your code')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        x=np.array([0.05,0.1,1])  #bias added into x vector
        model = NeuralNetwork(2,2,2,1,0,0.1,w)
        model.calculate(x)
        
    elif (sys.argv[1]=='example'):
        print('run example from class (single step)')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        x=np.array([0.05,0.1,1])  #bias added into x vector
        model = NeuralNetwork(2,2,2,0,0,0.1,w)
        model.calculate(x)
        
        np.array([0.01,0.99])
        
    elif(sys.argv[1]=='and'):
        print('learn and')
        
    elif(sys.argv[1]=='xor'):
        print('learn xor')