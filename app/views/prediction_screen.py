import sys
import pickle
import numpy as np
import pandas as pd
from app.neural_network.layer import Layer
from app.neural_network.neural_network import NeuralNetwork

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QTextEdit, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from app.files_preprocessing.file_preprocessor import DataPreprocessor
class PredictionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Прогнозирование с нейронной сетью")
        self.setGeometry(100, 100, 800, 600)
        
        # Инициализация атрибутов
        self.model = None
        self.preprocessor = None
        self.data = None
        
        # Создание интерфейса
        self.init_ui()
        
    def init_ui(self):
        # Основной виджет и layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Виджеты для загрузки модели
        self.model_label = QLabel("Модель не загружена")
        self.model_label.setAlignment(Qt.AlignCenter)
        load_model_btn = QPushButton("Загрузить модель и препроцессор")
        load_model_btn.clicked.connect(self.load_model)
        
        # Виджеты для загрузки данных
        self.data_label = QLabel("Данные не загружены")
        self.data_label.setAlignment(Qt.AlignCenter)
        load_data_btn = QPushButton("Загрузить данные для предсказания")
        load_data_btn.clicked.connect(self.load_data)
        
        # Кнопка предсказания
        predict_btn = QPushButton("Выполнить предсказание")
        predict_btn.clicked.connect(self.predict)
        
        # Поле для вывода результатов
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        
        # Добавление виджетов в layout
        layout.addWidget(QLabel("<h2>Прогнозирование с нейронной сетью</h2>"))
        layout.addWidget(load_model_btn)
        layout.addWidget(self.model_label)
        layout.addWidget(load_data_btn)
        layout.addWidget(self.data_label)
        layout.addWidget(predict_btn)
        layout.addWidget(QLabel("<h3>Результаты:</h3>"))
        layout.addWidget(self.result_text)
        
    def load_model(self):
        """Загрузка сохраненной модели и препроцессора"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл модели", "", "Pickle Files (*.pkl)"
        )
        
        if filepath:
            try:
                with open(filepath, 'rb') as f:
                    saved_objects = pickle.load(f)
                    self.model = saved_objects["neural_network"]
                    self.preprocessor = saved_objects["data_preprocessor"]
                self.model_label.setText(f"Модель загружена: {filepath}")
                QMessageBox.information(self, "Успех", "Модель и препроцессор успешно загружены!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель: {str(e)}")
    
    def load_data(self):
        """Загрузка данных для предсказания"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл с данными", "", 
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )
        
        if filepath:
            try:
                if filepath.endswith('.csv'):
                    self.data = pd.read_csv(filepath)
                elif filepath.endswith('.xlsx'):
                    self.data = pd.read_excel(filepath)
                else:
                    QMessageBox.warning(self, "Ошибка", "Неподдерживаемый формат файла")
                    return
                
                self.data_label.setText(f"Данные загружены: {filepath}\nРазмер: {self.data.shape}")
                self.result_text.append(f"Загружены данные:\n{self.data.head().to_string()}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить данные: {str(e)}")
    
    def predict(self):
        """Выполнение предсказания на загруженных данных"""
        if not self.model or not self.preprocessor:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите модель и препроцессор!")
            return
        
        if self.data is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные для предсказания!")
            return
        
        try:
            processed_data = self.preprocessor.preprocess_data_for_predict(self.data)
            print(processed_data)
            # Выполнение предсказания
            predictions = []
            for i in range(processed_data.shape[0]):
                output = self.model.forward(processed_data[i:i+1])
                print(self.model.layers[0])
                predictions.append(output)
            
            # Форматирование результатов
            results = pd.DataFrame({
                'Прогноз': [self.preprocessor.get_predicted_label(pred[0]) for pred in predictions],
                'Исходные данные': [str(row) for _, row in self.data.iterrows()]
            })
            
            # Вывод результатов
            self.result_text.append("\nРезультаты предсказания:")
            self.result_text.append(results.to_string())
            
            QMessageBox.information(self, "Готово", "Предсказание успешно выполнено!")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при выполнении предсказания: {str(e)}")
            self.result_text.append(f"\nОшибка: {str(e)}")