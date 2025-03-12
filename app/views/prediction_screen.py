import sys
import pickle
import numpy as np
from app.neural_network.layer import Layer
from app.neural_network.neural_network import NeuralNetwork

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QTextEdit, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from app.files_preprocessing.file_preprocessor import DataPreprocessor
class PredictionWindow(QMainWindow):
    def __init__(self, preprocessor: DataPreprocessor):
        super().__init__()
        self.model = None
        self.preprocessor = preprocessor
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Neural Network Predictor')
        self.setGeometry(100, 100, 600, 400)

        # Главный виджет и layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Панель управления
        control_layout = QHBoxLayout()
        self.load_btn = QPushButton('Load Model', self)
        self.load_btn.clicked.connect(self.load_model)
        self.model_info = QLabel('No model loaded')
        self.model_info.setAlignment(Qt.AlignLeft)
        
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.model_info)
        layout.addLayout(control_layout)

        # Поле ввода
        input_layout = QVBoxLayout()
        input_label = QLabel('Enter features (comma separated):')
        self.input_field = QLineEdit()
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_field)
        layout.addLayout(input_layout)

        # Кнопка предсказания
        self.predict_btn = QPushButton('Predict', self)
        self.predict_btn.clicked.connect(self.make_prediction)
        self.predict_btn.setEnabled(False)
        layout.addWidget(self.predict_btn)

        # Область вывода
        output_layout = QVBoxLayout()
        output_label = QLabel('Prediction Result:')
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_area)
        layout.addLayout(output_layout)

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Open Model File', '', 'Pickle Files (*.pkl)'
        )

        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.model_info.setText(f"Loaded: {file_path.split('/')[-1]}")
                self.predict_btn.setEnabled(True)
                self.show_message(
                    f"Model successfully loaded!\n"
                    f"Layers: {len(self.model.layers)}\n"
                    f"Output labels: {self.model.output_layer_labels}"
                )
            except Exception as e:
                self.show_error(f"Error loading model: {str(e)}")
                self.model = None
                self.predict_btn.setEnabled(False)
                self.model_info.setText('No model loaded')

    def make_prediction(self):
        if not self.model:
            self.show_error("No model loaded!")
            return

        try:
            
            
            # Выполнение предсказания
            output = self.model.forward(self.preprocessor.data)
            
            # Форматирование результата
            if self.model.encoder and self.model.output_layer_labels:
                prediction = self.model.output_layer_labels[np.argmax(output)]
                confidence = f"{np.max(output)*100:.2f}%"
                result = f"Class: {prediction}\nConfidence: {confidence}"
            else:
                result = f"Network output:\n{output}"
            
            self.output_area.setText(result)
            
        except Exception as e:
            self.show_error(f"Prediction error: {str(e)}")

    def show_message(self, text):
        self.output_area.setText(text)

    def show_error(self, message):
        QMessageBox.critical(self, 'Error', message)