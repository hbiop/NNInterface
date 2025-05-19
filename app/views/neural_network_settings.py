import json
import sqlite3
import sys
from datetime import datetime

import numpy as np
import pickle

from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QComboBox, QTableWidget,
                             QTableWidgetItem, QTextEdit, QMessageBox, QProgressBar, QFileDialog, QGroupBox, QDialog,
                             QTabWidget, QFormLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from app.database.database_manager import DatabaseManager
from app.database.training_logger import TrainingLogger
from app.files_preprocessing.file_preprocessor import DataPreprocessor
from app.neural_network.neural_network import NeuralNetwork


class TrainingThread(QThread):
    update_progress = pyqtSignal(int, float)
    finished = pyqtSignal()
    finish = pyqtSignal(bool)

    def __init__(self, nn, X, y, epochs, lr, logger, session_id):
        super().__init__()
        self.nn: NeuralNetwork = nn
        self.X = X
        self.y = y
        self.epochs = epochs
        self.lr = lr
        self.logger = logger
        self.session_id = session_id
        self.loss_history = []

    def run(self):
        for epoch in range(self.epochs):
            # Мини-батчи
            indices = np.random.permutation(self.X.shape[0])
            X_shuffled = self.X[indices]
            y_shuffled = self.y[indices]

            BATCH_SIZE = 20
            total_loss = 0

            for i in range(0, len(X_shuffled), BATCH_SIZE):
                batch_X = X_shuffled[i:i + BATCH_SIZE]
                batch_y = y_shuffled[i:i + BATCH_SIZE]

                # Прямое распространение и вычисление ошибки
                output, _ = self.nn.forward(batch_X)
                loss = np.mean((output - batch_y) ** 2)
                total_loss += loss

                # Обратное распространение
                self.nn.train(batch_X, batch_y, learning_rate=self.lr)

            # Расчет средней ошибки за эпоху
            avg_loss = total_loss / (len(X_shuffled) / BATCH_SIZE)
            self.loss_history.append(avg_loss)
            self.update_progress.emit(epoch + 1, avg_loss)
            self.logger.log_epoch(self.session_id, epoch + 1, avg_loss)

        final_loss = self.loss_history[-1] if self.loss_history else 0
        self.logger.complete_session(self.session_id, final_loss)
        self.finished.emit()
        self.finish.emit(True)


class NeuralNetworkGUI(QMainWindow):
    def __init__(self, preprocessor: DataPreprocessor):
        super().__init__()
        self.preprocessor = preprocessor
        self.initUI()
        print(self.preprocessor.preprocessed_data.shape)
        self.X = self.preprocessor.preprocessed_data
        self.nn = NeuralNetwork(self.X.shape[1])
        self.y = self.preprocessor.preprocessed_label

    def initUI(self):
        self.setWindowTitle('Интерфейс нейронной сети')
        self.setGeometry(300, 300, 900, 700)

        # Основной стиль
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
                font-family: Arial;
            }
            QGroupBox {
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton#export {
                background-color: #2196F3;
            }
            QPushButton#export:hover {
                background-color: #0b7dda;
            }
            QPushButton#back {
                background-color: #f44336;
            }
            QPushButton#back:hover {
                background-color: #d32f2f;
            }
            QTableWidget {
                background-color: white;
                border: 1px solid #ddd;
                selection-background-color: #e3f2fd;
            }
            QTextEdit, QLineEdit, QComboBox {
                background-color: white;
                border: 1px solid #ddd;
                padding: 5px;
                border-radius: 3px;
            }
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
            }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Кнопка возврата
        self.btn_back = QPushButton("← Вернуться на главную")
        self.btn_back.setObjectName("back")
        self.btn_back.clicked.connect(self.back_to_main_screen)
        layout.addWidget(self.btn_back)

        # Настройка разделов
        self.setup_layer_controls(layout)
        self.setup_training_controls(layout)

        # Кнопка экспорта
        self.btn_export = QPushButton("Экспорт модели")
        self.btn_export.setObjectName("export")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export)
        layout.addWidget(self.btn_export)

    def export(self):
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Экспорт модели",
                "",
                "Файлы модели (*.pkl)"
            )

            if not filename:
                return

            objects_to_save = {
                "neural_network": self.nn,
                "data_preprocessor": self.preprocessor
            }

            with open(filename, "wb") as f:
                pickle.dump(objects_to_save, f)

            QMessageBox.information(self, "Успех", "Модель успешно экспортирована!")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка экспорта:\n{str(e)}")

    def back_to_main_screen(self):
        from app.views.main_window import MainWindow
        self.w = MainWindow()
        self.w.show()
        self.close()

    def setup_layer_controls(self, layout):
        layer_group = QGroupBox("Конфигурация слоев")
        layer_layout = QVBoxLayout(layer_group)

        # Настройка слоя
        config_layout = QHBoxLayout()
        self.output_size = QLineEdit()
        self.output_size.setPlaceholderText("Введите размер")
        self.activation = QComboBox()
        self.activation.addItems(['sigmoid', 'relu', 'tanh', 'linear'])

        add_btn = QPushButton("Добавить слой")
        add_btn.clicked.connect(self.add_layer)

        config_layout.addWidget(QLabel("Размер слоя:"))
        config_layout.addWidget(self.output_size)
        config_layout.addWidget(QLabel("Функция активации:"))
        config_layout.addWidget(self.activation)
        config_layout.addWidget(add_btn)

        # Таблица слоев
        self.layers_table = QTableWidget()
        self.layers_table.setColumnCount(2)
        self.layers_table.setHorizontalHeaderLabels(["Размер слоя", "Активация"])
        self.layers_table.setColumnWidth(0, 200)
        self.layers_table.setColumnWidth(1, 250)
        self.layers_table.verticalHeader().setVisible(False)
        self.layers_table.setSelectionBehavior(QTableWidget.SelectRows)

        layer_layout.addLayout(config_layout)
        layer_layout.addWidget(self.layers_table)
        layout.addWidget(layer_group)

    def setup_training_controls(self, layout):
        train_group = QGroupBox("Обучение модели")
        train_layout = QVBoxLayout(train_group)

        # Параметры обучения
        param_layout = QHBoxLayout()
        self.epochs = QLineEdit('100')
        self.epochs.setValidator(QIntValidator(1, 10000))
        self.lr = QLineEdit('0.01')
        self.lr.setValidator(QDoubleValidator(0.0001, 1.0, 4))

        param_layout.addWidget(QLabel("Количество эпох:"))
        param_layout.addWidget(self.epochs)
        param_layout.addWidget(QLabel("Скорость обучения:"))
        param_layout.addWidget(self.lr)

        # Прогресс и лог
        self.progress = QProgressBar()
        self.progress.setAlignment(Qt.AlignCenter)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Здесь будет отображаться процесс обучения...")

        train_btn = QPushButton("Начать обучение")
        train_btn.clicked.connect(self.start_training)

        train_layout.addLayout(param_layout)
        train_layout.addWidget(QLabel("Прогресс обучения:"))
        train_layout.addWidget(self.progress)
        train_layout.addWidget(QLabel("Лог обучения:"))
        train_layout.addWidget(self.log)
        train_layout.addWidget(train_btn)
        layout.addWidget(train_group)

    def add_layer(self):
        try:
            output_size = int(self.output_size.text())
            if output_size <= 0:
                raise ValueError("Размер слоя должен быть положительным числом")

            activation = self.activation.currentText()

            self.nn.add_layer(output_size, activation)

            row = self.layers_table.rowCount()
            self.layers_table.insertRow(row)
            self.layers_table.setItem(row, 0, QTableWidgetItem(str(output_size)))
            self.layers_table.setItem(row, 1, QTableWidgetItem(activation))

            self.output_size.clear()

        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", f"Некорректные параметры слоя:\n{str(e)}")

    def start_training(self):
        if len(self.nn.layers) == 0:
            QMessageBox.warning(self, "Ошибка", "Добавьте слои перед обучением!")
            return

        try:
            last_layer_size = self.nn.layers[-1]['weights'].shape[1]
            if last_layer_size != len(self.preprocessor.classes):
                self.nn.add_layer(len(self.preprocessor.classes), activation='sigmoid')

            epochs = int(self.epochs.text())
            lr = float(self.lr.text())

            # Получаем конфигурацию слоев
            layers_config = []
            for layer in self.nn.layers:
                layers_config.append({
                    'size': layer['weights'].shape[1],
                    'activation': layer['activation']
                })

            # Создаем логгер и начинаем новую сессию
            self.logger = TrainingLogger()
            session_id = self.logger.start_new_session(
                epochs=epochs,
                lr=lr,
                layers_config=layers_config
            )

            self.progress.setMaximum(epochs)
            self.progress.setValue(0)
            self.log.clear()

            self.thread = TrainingThread(
                self.nn,
                self.X,
                self.y,
                epochs,
                lr,
                self.logger,
                session_id
            )
            self.thread.update_progress.connect(self.update_training)
            self.thread.finished.connect(self.training_finished)
            self.thread.finish.connect(lambda: self.btn_export.setEnabled(True))
            self.thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка инициализации: {str(e)}")

    def update_training(self, epoch, loss):
        self.progress.setValue(epoch)
        self.log.append(f"Эпоха {epoch}/{self.progress.maximum()}, Ошибка: {loss:.6f}")

    def enable_export_button(self, flag: bool):
        self.btn_export.setEnabled(flag)

    def training_finished(self):
        QMessageBox.information(self, "Обучение завершено", "Модель успешно обучена!")
        self.log.append("\nОбучение завершено успешно!")
