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
        self.setWindowTitle("Нейросетевое прогнозирование")
        self.setGeometry(100, 100, 1000, 800)

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
        main_layout = QVBoxLayout(central_widget)

        # Стилизация
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                font-size: 14px;
                margin: 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                margin: 5px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton#back_button {
                background-color: #f44336;
            }
            QPushButton#back_button:hover {
                background-color: #d32f2f;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #ddd;
                padding: 10px;
                font-family: Consolas;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
            }
        """)
        self.back_button = QPushButton("← Назад")
        self.back_button.setObjectName("back_button")
        self.back_button.clicked.connect(self.back)
        # Заголовок
        title_label = QLabel("Нейросетевое прогнозирование")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #333;")
        title_label.setAlignment(Qt.AlignCenter)

        # Группа для загрузки модели
        model_group = QGroupBox("Загрузка модели")
        model_layout = QVBoxLayout()

        self.model_label = QLabel("Модель не загружена")
        self.model_label.setWordWrap(True)
        load_model_btn = QPushButton("Загрузить модель и препроцессор")
        load_model_btn.clicked.connect(self.load_model)

        model_layout.addWidget(self.model_label)
        model_layout.addWidget(load_model_btn)
        model_group.setLayout(model_layout)

        # Группа для загрузки данных
        data_group = QGroupBox("Загрузка данных")
        data_layout = QVBoxLayout()

        self.data_label = QLabel("Данные не загружены")
        self.data_label.setWordWrap(True)
        load_data_btn = QPushButton("Загрузить данные для прогнозирования")
        load_data_btn.clicked.connect(self.load_data)

        data_layout.addWidget(self.data_label)
        data_layout.addWidget(load_data_btn)
        data_group.setLayout(data_layout)

        # Кнопка прогнозирования
        predict_btn = QPushButton("Выполнить прогнозирование")
        predict_btn.setStyleSheet("background-color: #2196F3;")
        predict_btn.clicked.connect(self.predict)

        # Группа для результатов
        result_group = QGroupBox("Результаты прогнозирования")
        result_layout = QVBoxLayout()

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)

        result_layout.addWidget(self.result_text)
        result_group.setLayout(result_layout)

        # Добавление всех элементов в основной layout
        main_layout.addWidget(self.back_button)
        main_layout.addWidget(title_label)
        main_layout.addWidget(model_group)
        main_layout.addWidget(data_group)
        main_layout.addWidget(predict_btn)
        main_layout.addWidget(result_group)

        # Установка растягивания для области результатов
        main_layout.setStretchFactor(result_group, 1)

    def back(self):
        from app.views.main_window import MainWindow
        self.w = MainWindow()
        self.w.show()
        self.close()

    def load_model(self):
        """Загрузка сохраненной модели и препроцессора"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл модели", "", "Файлы Pickle (*.pkl)"
        )

        if filepath:
            try:
                with open(filepath, 'rb') as f:
                    saved_objects = pickle.load(f)
                    self.model = saved_objects["neural_network"]
                    self.preprocessor = saved_objects["data_preprocessor"]
                self.model_label.setText(f"Модель загружена:\n{filepath}")
                QMessageBox.information(self, "Успех", "Модель и препроцессор успешно загружены!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель:\n{str(e)}")

    def load_data(self):
        """Загрузка данных для прогнозирования"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл с данными", "",
            "CSV файлы (*.csv);;Excel файлы (*.xlsx);;Все файлы (*)"
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

                self.data_label.setText(f"Данные загружены:\n{filepath}\nРазмер: {self.data.shape}")
                self.result_text.append(f"=== Загруженные данные ===\n{self.data.head().to_string()}\n")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить данные:\n{str(e)}")

    def predict(self):
        """Выполнение прогнозирования на загруженных данных"""
        if not self.model or not self.preprocessor:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите модель и препроцессор!")
            return

        if self.data is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные для прогнозирования!")
            return

        try:
            processed_data = self.preprocessor.preprocess_data_for_predict(self.data)

            # Выполнение прогнозирования для всего набора данных сразу
            predictions = self.model.forward(processed_data)

            results = pd.DataFrame({
                'Прогноз': [self.preprocessor.get_predicted_label(max(pred)) for pred in predictions],
                'Уверенность': [f"{max(pred) * 100:.2f}%" for pred in predictions]
            })

            self.result_text.append("\n=== Результаты прогнозирования ===")
            self.result_text.append(results.to_string())

            # Добавляем статистику
            self.result_text.append(f"\nВсего записей: {len(results)}")

            QMessageBox.information(self, "Готово", "Прогнозирование успешно выполнено!")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при выполнении прогнозирования:\n{str(e)}")
            self.result_text.append(f"\n!!! Ошибка: {str(e)}")