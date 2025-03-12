import numpy as np
from app.neural_network.activation_functions.sigmoid import SigmoidFunction
from app.neural_network.layer import Layer
import pickle
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.output_layer_labels = []
        self.encoder = None

    def save_model(self, filename):
            with open(filename, 'wb') as file:
                pickle.dump(self, file)
                
    def add_layer(self, output_size, activation_function, input_size=None):
        if not self.layers and input_size is None:
            raise ValueError("Для первого слоя необходимо указать input_size")
            
        if input_size is None:
            input_size = self.layers[-1].output_size
            
        layer = Layer(input_size, output_size, activation_function)
        self.layers.append(layer)
        
    def add_encoder(self, encoder):
        self.encoder = encoder

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, output_gradient, learning_rate):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, learning_rate)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                output = self.forward(X[i:i+1])
                loss = self.loss(y[i:i+1], output)
                print(f'Epoch {epoch + 1}/{epochs}, Sample {i + 1}/{X.shape[0]}, Loss: {loss}')
                output_gradient = self.loss_derivative(y[i:i+1], output)
                self.backward(output_gradient, learning_rate)

    @staticmethod
    def loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)  # MSE

    @staticmethod
    def loss_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

