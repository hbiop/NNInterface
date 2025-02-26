import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QLineEdit, QComboBox, QTableWidget,
                            QTableWidgetItem, QTextEdit, QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from app.files_preprocessing.file_preprocessor import DataPreprocessor
from app.neural_network.neural_network import NeuralNetwork

class TrainingThread(QThread):
    update_progress = pyqtSignal(int, float)
    finished = pyqtSignal()

    def __init__(self, nn, X, y, epochs, lr):
        super().__init__()
        self.nn = nn
        self.X = X
        self.y = y
        self.epochs = epochs
        self.lr = lr

    def run(self):
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(self.X.shape[0]):
                output = self.nn.forward(self.X[i:i+1])
                loss = self.nn.loss(self.y[i:i+1], output)
                output_gradient = self.nn.loss_derivative(self.y[i:i+1], output)
                self.nn.backward(output_gradient, self.lr)
                total_loss += loss
            avg_loss = total_loss / self.X.shape[0]
            self.update_progress.emit(epoch+1, avg_loss)
        self.finished.emit()

class NeuralNetworkGUI(QMainWindow):
    def __init__(self, preprocessor: DataPreprocessor):
        super().__init__()
        self.preprocessor = preprocessor
        self.nn = NeuralNetwork()
        self.initUI()
        
        # Получаем предобработанные данные
        self.X = self.preprocessor.preprocessed_data
        self.y = self.preprocessor.preprocessed_label


    def initUI(self):
        self.setWindowTitle('Neural Network Interface')
        self.setGeometry(300, 300, 800, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Section for adding layers
        self.setup_layer_controls(layout)
        
        # Training controls
        self.setup_training_controls(layout)
        
        # Prediction controls
        self.setup_prediction_controls(layout)
        # Остальная часть инициализации интерфейса без изменений
        # ... (как в вашем оригинальном коде)
    def setup_layer_controls(self, layout):
        layer_group = QWidget()
        layer_layout = QVBoxLayout(layer_group)

        # Layer configuration
        config_layout = QHBoxLayout()
        self.output_size = QLineEdit()
        self.activation = QComboBox()
        self.activation.addItems(['sigmoid', 'relu', 'tanh'])
        
        add_btn = QPushButton("Добавить слой")
        add_btn.clicked.connect(self.add_layer)

        config_layout.addWidget(QLabel("Выходной размер:"))
        config_layout.addWidget(self.output_size)
        config_layout.addWidget(QLabel("Функция активации:"))
        config_layout.addWidget(self.activation)
        config_layout.addWidget(add_btn)

        # Layers table
        self.layers_table = QTableWidget()
        self.layers_table.setColumnCount(2)
        self.layers_table.setHorizontalHeaderLabels(["Выходной размер", "Активация"])
        self.layers_table.setColumnWidth(0, 150)
        self.layers_table.setColumnWidth(1, 200)

        layer_layout.addLayout(config_layout)
        layer_layout.addWidget(self.layers_table)
        layout.addWidget(layer_group)
    
    def setup_training_controls(self, layout):
        train_group = QWidget()
        train_layout = QVBoxLayout(train_group)

        # Training parameters
        param_layout = QHBoxLayout()
        self.epochs = QLineEdit('100')
        self.lr = QLineEdit('0.01')

        param_layout.addWidget(QLabel("Эпохи:"))
        param_layout.addWidget(self.epochs)
        param_layout.addWidget(QLabel("Скорость обучения:"))
        param_layout.addWidget(self.lr)

        # Progress controls
        self.progress = QProgressBar()
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        train_btn = QPushButton("Начать обучение")
        train_btn.clicked.connect(self.start_training)

        train_layout.addLayout(param_layout)
        train_layout.addWidget(self.progress)
        train_layout.addWidget(self.log)
        train_layout.addWidget(train_btn)
        layout.addWidget(train_group)
    def setup_prediction_controls(self, layout):  
        predict_group = QWidget()
        predict_layout = QVBoxLayout(predict_group)

        self.input_data = QLineEdit()
        self.output_data = QLineEdit()
        self.output_data.setReadOnly(True)

        predict_btn = QPushButton("Сделать предсказание")
        predict_btn.clicked.connect(self.predict)

        predict_layout.addWidget(QLabel("Входные данные (через запятую):"))
        predict_layout.addWidget(self.input_data)
        predict_layout.addWidget(QLabel("Результат:"))
        predict_layout.addWidget(self.output_data)
        predict_layout.addWidget(predict_btn)
        layout.addWidget(predict_group)
    def add_layer(self):
        try:
            output_size = int(self.output_size.text())
            activation = self.activation.currentText()

            if not self.nn.layers:
                input_size = self.X.shape[1]
                self.nn.add_layer(output_size, activation, input_size)
            else:
                self.nn.add_layer(output_size, activation)

            row = self.layers_table.rowCount()
            self.layers_table.insertRow(row)
            self.layers_table.setItem(row, 0, QTableWidgetItem(str(output_size)))
            self.layers_table.setItem(row, 1, QTableWidgetItem(activation))

        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Некорректные значения параметров слоя")
            # Остальная часть метода без изменений
            # ... (как в вашем оригинальном коде)

    def start_training(self):
        if len(self.nn.layers) == 0:
            QMessageBox.warning(self, "Ошибка", "Добавьте хотя бы один слой!")
            return

        try:
            epochs = int(self.epochs.text())
            lr = float(self.lr.text())

            self.progress.setMaximum(epochs)
            # Используем предобработанные данные
            self.thread = TrainingThread(self.nn, self.X, self.y, epochs, lr)
            self.thread.update_progress.connect(self.update_training)
            self.thread.finished.connect(self.training_finished)
            self.thread.start()

        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Некорректные параметры обучения")
    
    def update_training(self, epoch, loss):
        self.progress.setValue(epoch)
        self.log.append(f"Эпоха {epoch}, Loss: {loss:.4f}")

    def training_finished(self):
        QMessageBox.information(self, "Обучение", "Обучение завершено!")

    def predict(self):
        try:
            data = np.array([float(x) for x in self.input_data.text().split(',')])
            print("data", data, "shape", data.shape)
            print("x", self.X, "shape", self.X.shape)
            if data.shape[0] != self.X.shape[1]:
                raise ValueError("Неверное количество признаков")
            
            prediction = self.nn.forward(data.reshape(1, -1))
            self.output_data.setText(str(prediction.flatten()))
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка предсказания: {str(e)}")

if __name__ == '__main__':
    # Пример использования с тестовыми данными
    X = np.random.rand(100, 10)
    y = np.random.rand(100, 1)
    print("x", X)
    print("y", y)
    app = QApplication(sys.argv)
    window = NeuralNetworkGUI(X, y)
    window.show()
    sys.exit(app.exec_())