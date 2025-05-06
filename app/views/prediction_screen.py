import os
import sys
import pickle
import numpy as np
import pandas as pd
from app.neural_network.layer import Layer
from app.neural_network.neural_network import NeuralNetwork

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QTextEdit, QFileDialog, QMessageBox, QGroupBox)
from PyQt5.QtCore import Qt
from app.files_preprocessing.file_preprocessor import DataPreprocessor


class PredictionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Прогнозирование нейросетью")
        self.setGeometry(100, 100, 1000, 800)

        # Инициализация компонентов
        self.model = None
        self.preprocessor = None
        self.data = None
        self.init_ui()
        self.set_styles()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Кнопка возврата
        self.btn_back = QPushButton("← Назад")
        self.btn_back.setObjectName("back_button")
        self.btn_back.clicked.connect(self.close)

        # Заголовок
        lbl_title = QLabel("Прогнозирование с нейронной сетью")
        lbl_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #2c3e50;")

        # Группа модели
        grp_model = QGroupBox("Модель")
        self.lbl_model = QLabel("Модель не загружена")
        btn_load_model = QPushButton("Загрузить модель")
        btn_load_model.clicked.connect(self.load_model)

        model_layout = QVBoxLayout()
        model_layout.addWidget(self.lbl_model)
        model_layout.addWidget(btn_load_model)
        grp_model.setLayout(model_layout)

        # Группа данных
        grp_data = QGroupBox("Данные")
        self.lbl_data = QLabel("Данные не загружены")
        btn_load_data = QPushButton("Загрузить данные")
        btn_load_data.clicked.connect(self.load_data)

        data_layout = QVBoxLayout()
        data_layout.addWidget(self.lbl_data)
        data_layout.addWidget(btn_load_data)
        grp_data.setLayout(data_layout)

        # Кнопка прогноза
        btn_predict = QPushButton("Выполнить прогноз")
        btn_predict.setStyleSheet("background-color: #3498db;")
        btn_predict.clicked.connect(self.predict)

        # Результаты
        self.txt_results = QTextEdit()
        self.txt_results.setReadOnly(True)

        # Компоновка
        layout.addWidget(self.btn_back)
        layout.addWidget(lbl_title)
        layout.addWidget(grp_model)
        layout.addWidget(grp_data)
        layout.addWidget(btn_predict)
        layout.addWidget(self.txt_results)

    def set_styles(self):
        self.setStyleSheet("""
            QMainWindow { background: #ecf0f1; }
            QGroupBox { 
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 10px;
                padding: 15px;
            }
            QPushButton {
                background: #2ecc71;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 120px;
            }
            QPushButton:hover { background: #27ae60; }
            QPushButton#back_button { background: #e74c3c; }
            QPushButton#back_button:hover { background: #c0392b; }
            QTextEdit { 
                background: white;
                border: 1px solid #bdc3c7;
                font-family: Consolas;
                font-size: 12px;
            }
        """)

    def load_model(self):
        """Загрузка модели и препроцессора"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл модели", "", "Pickle Files (*.pkl)")

        if path:
            try:
                with open(path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.model = saved_data['neural_network']
                    self.preprocessor = saved_data['data_preprocessor']

                self.lbl_model.setText(f"Загружена модель:\n{os.path.basename(path)}")
                self.txt_results.append("✅ Модель успешно загружена")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки:\n{str(e)}")

    def load_data(self):
        """Загрузка данных для прогноза"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите данные", "", "CSV (*.csv);;Excel (*.xlsx)")

        if path:
            try:
                if path.endswith('.csv'):
                    self.data = pd.read_csv(path)
                else:
                    self.data = pd.read_excel(path)

                self.lbl_data.setText(f"Загружено {len(self.data)} записей")
                self.txt_results.append(f"=== Первые 5 строк ===\n{self.data.head().to_string()}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки:\n{str(e)}")

    def predict(self):
        """Выполнение прогнозирования"""
        if not all([self.model, self.preprocessor, self.data is not None]):
            QMessageBox.warning(self, "Ошибка", "Загрузите модель и данные!")
            return

        try:
            # Предобработка данных
            X = self.preprocessor.preprocess_data_for_predict(self.data)

            # Прогнозирование
            outputs, _ = self.model.forward(X)
            predictions = np.argmax(outputs, axis=1)
            confidences = np.max(outputs, axis=1)

            # Декодирование меток
            labels = [self.preprocessor.classes[p] for p in predictions]

            # Формирование результатов
            results = pd.DataFrame({
                'Прогноз': labels,
                'Уверенность': [f"{c * 100:.1f}%" for c in confidences]
            })

            self.txt_results.append("\n=== Результаты ===")
            self.txt_results.append(results.to_string())
            self.txt_results.append(f"\nВсего обработано: {len(results)} записей")

        except Exception as e:
            self.txt_results.append(f"\n❌ Ошибка: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка прогнозирования:\n{str(e)}")

    def close(self):
        """Возврат на предыдущий экран"""
        from app.views.main_window import MainWindow
        self.parent = MainWindow()
        self.parent.show()
        super().close()