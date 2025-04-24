import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from app.neural_network.layer  import Layer
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.output_layer_labels = []
        self.encoder = None
        self.scaler = StandardScaler()  # Для нормализации данных
    
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
    
    def train(self, X, y, epochs=100, learning_rate=0.01, batch_size=32, verbose=True):
        X = self.scaler.fit_transform(X)
        
        if self.encoder is not None:
            y = self.encoder.transform(y)
        
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                output = self.forward(batch_X)
                loss = self.loss(batch_y, output)
                epoch_loss += loss
                output_gradient = self.loss_derivative(batch_y, output)
                self.backward(output_gradient, learning_rate)
            
            if verbose and (epoch % 10 == 0 or epoch == epochs-1):
                avg_loss = epoch_loss / (n_samples / batch_size)
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
    
    @staticmethod
    def loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def loss_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

