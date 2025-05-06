import numpy as np
from app.neural_network.activation_functions.sigmoid import SigmoidFunction

class Layer:
    def __init__(self, input_size, output_size, activation_function):
        self.input_size = input_size
        self.output_size = output_size
        
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / (input_size + output_size))
        self.biases = np.zeros((1, output_size))
        
        if activation_function == "sigmoid":
            self.activation_function = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation_function == "relu":
            self.activation_function = self.relu
            self.activation_derivative = self.relu_derivative
        elif activation_function == "tanh":
            self.activation_function = self.tanh
            self.activation_derivative = self.tanh_derivative
        else:  # linear
            self.activation_function = lambda x: x
            self.activation_derivative = lambda x: 1
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        s = Layer.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2
    
    def forward(self, input_data):
        self.input_data = input_data
        self.z = np.dot(input_data, self.weights) + self.biases
        return self.activation_function(self.z)
    
    def backward(self, output_gradient, learning_rate):
        activation_gradient = self.activation_derivative(self.z)
        grad_z = output_gradient * activation_gradient
        
        grad_weights = np.dot(self.input_data.T, grad_z)
        grad_biases = np.sum(grad_z, axis=0, keepdims=True)
        
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        
        return np.dot(grad_z, self.weights.T)



