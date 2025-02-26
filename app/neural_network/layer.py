import numpy as np
from app.neural_network.activation_functions.sigmoid import SigmoidFunction

class Layer:
    def __init__(self, input_size, output_size, activation_function):
        self.input_size = input_size
        self.output_size = output_size  # Добавляем атрибут output_size
        
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        
        if activation_function == "sigmoid":
            self.activation_function = SigmoidFunction()
        # Добавьте другие функции активации по необходимости

    def forward(self, input_data):
        self.input_data = input_data
        self.z = np.dot(input_data, self.weights) + self.biases
        return self.activation_function.function(self.z)

    def backward(self, output_gradient, learning_rate):
        activation_gradient = self.activation_function.backward(self.z)
        grad_z = output_gradient * activation_gradient
        
        # Обновление параметров
        self.weights -= np.dot(self.input_data.T, grad_z) * learning_rate
        self.biases -= np.sum(grad_z, axis=0, keepdims=True) * learning_rate
        
        return np.dot(grad_z, self.weights.T)