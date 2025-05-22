import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from PyQt5.QtCore import QDir, Qt, QStringListModel, QModelIndex, QThread, pyqtSignal
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QMainWindow, QHBoxLayout, QSizePolicy, \
    QStackedWidget, QApplication, QListView, QFileSystemModel, QTreeView, QGroupBox, QComboBox, QTableWidget, \
    QFileDialog, QMessageBox, QTableWidgetItem, QHeaderView, QProgressDialog, QTextEdit, QScrollArea, QProgressBar

from app.files_preprocessing.file_preprocessor import DataPreprocessor
from app.files_preprocessing.readers.csv_reader import CsvReader
from app.files_preprocessing.readers.excel_reader import ExcelReader
from app.neural_network.neural_network import NeuralNetwork

@dataclass
class ModelsData:
    id: int
    title: str

@dataclass
class ModelData:
    id: int
    title: str
    description: str

@dataclass
class FilesData:
    id: int
    title: str

@dataclass
class FileData:
    id: int
    title: str
    file_type: str
    upload_date: str
    file_size: str
    file_path: str
    description: str

class DataBaseManager:
    def __init__(self, db_path='data_files.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Таблица для хранения информации о датасетах
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    upload_date TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_path TEXT,
                    description TEXT
                )
            ''')

            cursor.execute('''
                            CREATE TABLE IF NOT EXISTS models (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                title TEXT NOT NULL,
                                description TEXT NOT NULL,
                                creation_date TEXT NOT NULL,
                                parameters TEXT NOT NULL
                            )
                        ''')

            conn.commit()
    #Работа с файлами
    def get_all_files(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, filename FROM files")
            files_data = [FilesData(id = int(data[0]), title = str(data[1])) for data in cursor.fetchall()]
            conn.commit()
        return files_data

    def get_file_by_id(self, id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM files where id = {id}")
            data = cursor.fetchone()
            files_data = FileData(
                id = int(data[0]),
                title = str(data[1]),
                file_type = str(data[2]),
                upload_date = str(data[3]),
                file_size = str(data[4]),
                file_path = str(data[5]),
                description = str(data[6])
            )
            conn.commit()
        return files_data

    def save_file(self, file_path, title, description):
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        file_type = os.path.splitext(file_path)[1]
        upload_date = datetime.now().isoformat()
        description = ""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO files 
                (filename, file_type, upload_date, file_size, file_path, description)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (filename, file_type, upload_date, file_size, file_path, description))

            conn.commit()
            return cursor.lastrowid

    #Работа с моделями
    def save_model(self, title, description, parameters):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO models 
                (title, description, creation_date, parameters)
                VALUES (?, ?, ?, ?)
            ''', (title, description, datetime.now().isoformat(), parameters))

            conn.commit()
            return cursor.lastrowid

    def get_all_models(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title FROM models")
            files_data = [ModelsData(id = int(data[0]), title = str(data[1])) for data in cursor.fetchall()]
            conn.commit()
        return files_data

    def get_model_by_id(self, id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT id, title, description FROM models where id = {id}")
            data = cursor.fetchone()
            models_data = ModelData(
                id=int(data[0]),
                title=str(data[1]),
                description=str(data[2])
            )
            conn.commit()
        return models_data

class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.username_label = QLabel("Имя пользователя:")
        self.username_input = QLineEdit()

        self.password_label = QLabel("Пароль:")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)

        self.login_button = QPushButton("Войти")

        layout = QVBoxLayout()
        layout.addWidget(self.username_label)
        layout.addWidget(self.username_input)
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_input)
        layout.addWidget(self.login_button)

        self.setLayout(layout)

class DataSets(QWidget):
    def __init__(self, callback, parent=None):
        super().__init__(parent)
        self.db = DataBaseManager()
        self.all_files = self.db.get_all_files()
        self.files_titles = [data.title for data in self.all_files]
        self.model = QStringListModel(self.files_titles)
        self.layout = QHBoxLayout()
        self.list_view = QListView()
        self.list_view.setModel(self.model)
        self.list_view.clicked.connect(self.on_item_clicked)
        self.vertical_widget = QWidget()
        self.vertical_widget.setFixedWidth(500)
        self.load_button = QPushButton("Загрузить новый набор данных")
        self.load_button.setFixedHeight(30)
        self.widget = QWidget()
        self.widget_layout = QVBoxLayout()
        self.title_label = QLabel()
        self.description_label = QLabel()
        self.size_label = QLabel()
        self.upload_date_label = QLabel()
        self.widget_layout.addWidget(self.title_label)
        self.widget_layout.addWidget(self.description_label)
        self.widget_layout.addWidget(self.size_label)
        self.widget_layout.addWidget(self.upload_date_label)
        self.widget.setLayout(self.widget_layout)
        self.vertical_layout = QVBoxLayout()
        self.vertical_layout.addWidget(self.widget)
        self.vertical_layout.addWidget(self.load_button)
        self.vertical_widget.setLayout(self.vertical_layout)
        self.layout.addWidget(self.list_view)
        self.layout.addWidget(self.vertical_widget)

        self.setLayout(self.layout)
        self.load_button.clicked.connect(callback)

    def on_item_clicked(self, index: QModelIndex):
        """Обработчик клика по элементу"""
        selected_item = index.row()
        file = self.db.get_file_by_id(self.all_files[selected_item].id)
        self.title_label.setText(f"Название {file.title}")
        self.description_label.setText(f"Описание {file.description}")
        self.size_label.setText(f"Размер файла {file.file_size}")
        self.upload_date_label.setText(f"Дата загрузки {file.upload_date}")

class DataSetsLoad(QWidget):
    def __init__(self, callback_back):
        super().__init__()
        self.db = DataBaseManager()
        self.preprocessor = DataPreprocessor()

        # Основной layout для всего виджета
        main_layout = QVBoxLayout(self)  # Главный layout устанавливается для самого виджета

        # Control Panel
        self.control_widget = QWidget()
        self.control_widget.setFixedHeight(50)
        self.control_layout = QHBoxLayout(self.control_widget)  # Layout устанавливается для control_widget
        self.btn_back = QPushButton("Назад")
        self.btn_back.clicked.connect(callback_back)
        self.btn_load = QPushButton("Загрузить данные")
        self.btn_load.clicked.connect(self.load_data)
        self.control_layout.addWidget(self.btn_back)
        self.control_layout.addWidget(self.btn_load)

        # File Description
        self.title_label = QLabel("Введите название:")
        self.title_field = QTextEdit()
        self.title_field.setFixedHeight(50)

        self.description_label = QLabel("Введите описание:")
        self.description_field = QTextEdit()
        self.description_field.setFixedHeight(300)

        # Table
        self.table = QTableWidget()
        self.table.setFixedHeight(300)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSortingEnabled(True)

        self.confirm_button = QPushButton("Подтвердить данные")
        self.confirm_button.setEnabled(False)
        # Создаем виджет для содержимого с прокруткой
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.addWidget(self.control_widget)
        self.scroll_layout.addWidget(self.title_label)
        self.scroll_layout.addWidget(self.title_field)
        self.scroll_layout.addWidget(self.description_label)
        self.scroll_layout.addWidget(self.description_field)
        self.scroll_layout.addWidget(self.table)
        self.scroll_layout.addWidget(self.confirm_button)

        # Настраиваем область прокрутки
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.scroll_content)
        self.scroll_area.setWidgetResizable(True)  # Разрешаем изменение размера содержимого

        # Добавляем scroll_area в главный layout
        main_layout.addWidget(self.scroll_area)

        # Устанавливаем stretch factors, чтобы table занимал все доступное пространство
        self.scroll_layout.setStretchFactor(self.table, 1)


    def load_data(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл с данными",
            "",
            "CSV файлы(*.csv);;Excel файлы (*.xlsx)",
            options=options
        )

        if not file_path:
            return

        try:
            self.file_path = file_path
            _, extension = os.path.splitext(file_path)
            file_type = extension[1:].upper()  # "csv" или "xlsx"

            if extension == ".csv":
                self.preprocessor.reader = CsvReader()
            elif extension == ".xlsx":
                self.preprocessor.reader = ExcelReader()
            else:
                raise ValueError("Неподдерживаемый формат файла")

            self.preprocessor.read_data(file_path)
            self.display_data()
            self.confirm_button.setEnabled(True)

            QMessageBox.information(self, "Успех", "Данные успешно загружены и сохранены!")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки",
                                 f"Не удалось загрузить данные:\n{str(e)}")

    def update_target_columns(self):
        """Обновляет список столбцов для выбора целевой переменной"""
        self.target_column_combo.clear()
        if self.preprocessor.data is not None:
            columns = self.preprocessor.data.columns.tolist()
            self.target_column_combo.addItems(columns)
            if columns:
                self.target_column_combo.setCurrentIndex(0)

    def display_data(self):
        data = self.preprocessor.data
        self.table.clear()

        self.table.setRowCount(data.shape[0])
        self.table.setColumnCount(data.shape[1])
        self.table.setHorizontalHeaderLabels(data.columns.tolist())

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                item = QTableWidgetItem(str(data.iat[i, j]))
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.table.setItem(i, j, item)

        # Автоматическое выравнивание столбцов
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

    def confirm_data(self):
        self.db.save_file(
            file_path=self.file_path,
            description=self.description_field.toPlainText(),
            title=self.title_field.toPlainText())

        self.callback_back()

class MainView(QMainWindow):
    def __init__(self):
        super().__init__()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.setMinimumSize(1200, 550)
        self.layout = QVBoxLayout()
        self.horizontal_widget = QWidget()
        self.horizontal_layout = QHBoxLayout()
        self.h_button_1 = QPushButton("Главная")
        self.h_button_1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.h_button_1.clicked.connect(self.go_to_main_page)
        self.h_button_2 = QPushButton("Наборы данных")
        self.h_button_2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.h_button_2.clicked.connect(self.go_to_dataset_page)
        self.h_button_3 = QPushButton("Модели")
        self.h_button_3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.h_button_3.clicked.connect(self.go_to_models_page)
        self.h_button_4 = QPushButton("Обучение")
        self.h_button_4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.h_button_5 = QPushButton("Предсказание")
        self.h_button_5.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.horizontal_layout.addWidget(self.h_button_1)
        self.horizontal_layout.addWidget(self.h_button_2)
        self.horizontal_layout.addWidget(self.h_button_3)
        self.horizontal_layout.addWidget(self.h_button_4)
        self.horizontal_layout.addWidget(self.h_button_5)
        self.horizontal_widget.setFixedHeight(50)
        self.horizontal_widget.setLayout(self.horizontal_layout)
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(Widget())
        self.stacked_widget.addWidget(DataSets(self.go_to_load_dataset_page))
        self.stacked_widget.addWidget(DataSetsLoad(self.go_to_dataset_page))
        self.stacked_widget.addWidget(Models(self.go_to_models_settings, self.go_to_model_training_screen))
        self.stacked_widget.addWidget(ModelsSettings(self.go_to_models_page))
        self.stacked_widget.addWidget(LearnModelScreen())
        self.layout.addWidget(self.horizontal_widget)
        self.layout.addWidget(self.stacked_widget)
        self.central_widget.setLayout(self.layout)

    def go_to_model_training_screen(self, model_id):
        self.stacked_widget.setCurrentIndex(5)
        training_screen = self.stacked_widget.widget(5)

        if hasattr(training_screen, 'set_model_id'):
            training_screen.set_model_id(model_id)
        else:
            print("Экран обучения не имеет метода set_model_id")

    def go_to_main_page(self):
        self.stacked_widget.setCurrentIndex(0)

    def go_to_models_settings(self):
        self.stacked_widget.setCurrentIndex(4)
    def go_to_models_page(self):
        self.stacked_widget.setCurrentIndex(3)

    def go_to_dataset_page(self):
        self.stacked_widget.setCurrentIndex(1)

    def go_to_load_dataset_page(self):
        self.stacked_widget.setCurrentIndex(2)


class TrainingThread(QThread):
    update_progress = pyqtSignal(int, float, float)  # epoch, loss, accuracy
    finished = pyqtSignal(dict)  # Результаты обучения
    error_occurred = pyqtSignal(str)  # Ошибки обучения

    def __init__(self, nn, X_train, y_train, X_val, y_val, epochs, lr, logger, session_id):
        super().__init__()
        self.nn = nn
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.epochs = epochs
        self.lr = lr
        self.logger = logger
        self.session_id = session_id
        self.loss_history = []
        self.val_loss_history = []
        self.accuracy_history = []
        self._is_running = True

    def run(self):
        try:
            best_loss = float('inf')
            best_weights = None

            for epoch in range(1, self.epochs + 1):
                if not self._is_running:
                    break

                # Обучение на мини-батчах
                train_loss = self._train_epoch()

                # Валидация
                val_loss, accuracy = self._validate()

                # Логирование
                self.loss_history.append(train_loss)
                self.val_loss_history.append(val_loss)
                self.accuracy_history.append(accuracy)

                # Сохранение лучших весов
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_weights = self.nn.get_weights()

                # Отправка прогресса
                self.update_progress.emit(epoch, train_loss, accuracy)
                self.logger.log_epoch(self.session_id, epoch, train_loss, val_loss, accuracy)

            # Завершение обучения
            results = {
                'best_loss': best_loss,
                'final_accuracy': self.accuracy_history[-1] if self.accuracy_history else 0,
                'loss_history': self.loss_history,
                'val_loss_history': self.val_loss_history,
                'accuracy_history': self.accuracy_history,
                'best_weights': best_weights
            }

            self.logger.complete_session(self.session_id, best_loss, self.accuracy_history[-1])
            self.finished.emit(results)

        except Exception as e:
            self.error_occurred.emit(f"Ошибка обучения: {str(e)}")
            import traceback
            traceback.print_exc()

    def _train_epoch(self):
        """Одна эпоха обучения"""
        indices = np.random.permutation(self.X_train.shape[0])
        X_shuffled = self.X_train[indices]
        y_shuffled = self.y_train[indices]

        BATCH_SIZE = 32
        total_loss = 0

        for i in range(0, len(X_shuffled), BATCH_SIZE):
            if not self._is_running:
                break

            batch_X = X_shuffled[i:i + BATCH_SIZE]
            batch_y = y_shuffled[i:i + BATCH_SIZE]

            # Прямое распространение и вычисление ошибки
            output, _ = self.nn.forward(batch_X)
            loss = self._compute_loss(output, batch_y)
            total_loss += loss

            # Обратное распространение
            self.nn.train(batch_X, batch_y, learning_rate=self.lr)

        return total_loss / (len(X_shuffled) / BATCH_SIZE)

    def _validate(self):
        """Валидация модели"""
        output, _ = self.nn.forward(self.X_val)
        loss = self._compute_loss(output, self.y_val)
        accuracy = self._compute_accuracy(output, self.y_val)
        return loss, accuracy

    def _compute_loss(self, output, y_true):
        """Вычисление функции потерь"""
        return np.mean((output - y_true) ** 2)

    def _compute_accuracy(self, output, y_true):
        """Вычисление точности"""
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:  # Для классификации
            preds = np.argmax(output, axis=1)
            true = np.argmax(y_true, axis=1)
            return np.mean(preds == true)
        else:  # Для регрессии
            return np.corrcoef(output.flatten(), y_true.flatten())[0, 1]

    def stop(self):
        """Остановка обучения"""
        self._is_running = False

class LearnModelScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.model_id = None
        self.preprocessor = DataPreprocessor()
        self.db = DataBaseManager()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Группа выбора данных
        data_group = QGroupBox("Выбор данных для обучения")
        data_layout = QVBoxLayout(data_group)

        self.file_combobox = QComboBox()
        self.file_combobox.setPlaceholderText("Выберите файл для обучения")
        data_layout.addWidget(QLabel("Файл с данными:"))
        data_layout.addWidget(self.file_combobox)

        self.target_column_combo = QComboBox()
        self.target_column_combo.setPlaceholderText("Выберите целевую переменную")
        data_layout.addWidget(QLabel("Целевая переменная:"))
        data_layout.addWidget(self.target_column_combo)

        # Группа параметров обучения
        train_group = QGroupBox("Параметры обучения")
        train_layout = QVBoxLayout(train_group)

        param_layout = QHBoxLayout()
        self.epochs = QLineEdit('100')
        self.epochs.setValidator(QIntValidator(1, 10000))
        self.lr = QLineEdit('0.01')
        self.lr.setValidator(QDoubleValidator(0.0001, 1.0, 4))

        param_layout.addWidget(QLabel("Количество эпох:"))
        param_layout.addWidget(self.epochs)
        param_layout.addWidget(QLabel("Скорость обучения:"))
        param_layout.addWidget(self.lr)

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

        layout.addWidget(data_group)
        layout.addWidget(train_group)
        self.setLayout(layout)

        self.load_files()

    def load_files(self):
        """Загружает список файлов из базы данных"""
        try:
            files = self.db.get_all_files()
            self.file_combobox.clear()
            for file in files:
                self.file_combobox.addItem(file.title, file.id)

            if not files:
                self.file_combobox.addItem("Нет доступных файлов", -1)
                self.file_combobox.setEnabled(False)

        except Exception as e:
            self.log.append(f"Ошибка загрузки файлов: {str(e)}")

    def set_model_id(self, model_id):
        """Устанавливает ID модели для обучения"""
        self.model_id = model_id

    def start_training(self):
        """Запускает обучение в отдельном потоке"""
        try:
            # Проверки и получение данных
            if not self._validate_inputs():
                return

            # Получение данных
            X_train, X_val, y_train, y_val = self._prepare_data()

            # Создание модели
            model = self._create_model(X_train.shape[1], y_train.shape[1])

            # Настройка логгера
            session_id = self.logger.start_session(
                model_id=self.model_id,
                file_id=self.file_combobox.currentData(),
                epochs=int(self.epochs.text()),
                lr=float(self.lr.text())
            )

            # Создание и запуск потока обучения
            self.thread = TrainingThread(
                nn=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=int(self.epochs.text()),
                lr=float(self.lr.text()),
                logger=self.logger,
                session_id=session_id
            )

            # Подключение сигналов
            self.thread.update_progress.connect(self.update_training_progress)
            self.thread.finished.connect(self.training_finished)
            self.thread.error_occurred.connect(self.training_error)

            # Блокировка интерфейса
            self.set_ui_enabled(False)
            self.progress.setValue(0)
            self.log.append("Начало обучения...")

            # Запуск потока
            self.thread.start()

        except Exception as e:
            self.log.append(f"Ошибка: {str(e)}")
            QMessageBox.critical(self, "Ошибка", str(e))

    def update_training_progress(self, epoch, loss, accuracy):
        """Обновление прогресса обучения"""
        self.progress.setValue(int((epoch / int(self.epochs.text())) * 100))
        self.log.append(
            f"Эпоха {epoch}/{self.epochs.text()}: "
            f"Ошибка={loss:.4f}, Точность={accuracy:.4f}"
        )

    def training_finished(self, results):
        """Завершение обучения"""
        self.log.append(
            f"Обучение завершено! Лучшая ошибка: {results['best_loss']:.4f}, "
            f"Финальная точность: {results['final_accuracy']:.4f}"
        )

        # Сохранение обученной модели
        self.save_trained_model(results['best_weights'])

        # Разблокировка интерфейса
        self.set_ui_enabled(True)
        QMessageBox.information(self, "Готово", "Обучение завершено успешно!")

    def training_error(self, message):
        """Обработка ошибки обучения"""
        self.log.append(message)
        self.set_ui_enabled(True)
        QMessageBox.critical(self, "Ошибка", message)

    def set_ui_enabled(self, enabled):
        """Блокировка/разблокировка интерфейса"""
        self.file_combobox.setEnabled(enabled)
        self.target_column_combo.setEnabled(enabled)
        self.epochs.setEnabled(enabled)
        self.lr.setEnabled(enabled)
        self.train_btn.setEnabled(enabled)

class Models(QWidget):
    def __init__(self, callback, callback_training, parent=None):
        super().__init__(parent)
        self.db = DataBaseManager()
        self.all_models = self.db.get_all_models()
        self.models_titles = [data.title for data in self.all_models]
        self.model = QStringListModel(self.models_titles)
        self.layout = QHBoxLayout()
        self.list_view = QListView()
        self.list_view.setModel(self.model)
        self.list_view.clicked.connect(self.on_item_clicked)
        self.vertical_widget = QWidget()
        self.vertical_widget.setFixedWidth(500)
        self.load_button = QPushButton("Создать новую модель")
        self.load_button.setFixedHeight(30)
        self.learn_model_button = QPushButton("Обучить модель")
        self.learn_model_button.clicked.connect(self.go_to_training_screen)
        self.callback_training = callback_training
        self.learn_model_button.setVisible(False)
        self.widget = QWidget()
        self.widget_layout = QVBoxLayout()
        self.title_label = QLabel()
        self.description_label = QLabel()
        self.upload_date_label = QLabel()
        self.widget_layout.addWidget(self.title_label)
        self.widget_layout.addWidget(self.description_label)
        self.widget_layout.addWidget(self.upload_date_label)
        self.widget_layout.addWidget(self.learn_model_button)
        self.widget.setLayout(self.widget_layout)
        self.vertical_layout = QVBoxLayout()
        self.vertical_layout.addWidget(self.widget)
        self.vertical_layout.addWidget(self.load_button)
        self.vertical_widget.setLayout(self.vertical_layout)
        self.layout.addWidget(self.list_view)
        self.layout.addWidget(self.vertical_widget)

        self.setLayout(self.layout)
        self.load_button.clicked.connect(callback)

    def go_to_training_screen(self):
        index = self.list_view.currentIndex().row()
        self.callback_training(index)

    def on_item_clicked(self, index: QModelIndex):
        """Обработчик клика по элементу"""
        selected_item = index.row()
        file = self.db.get_file_by_id(self.all_models[selected_item].id)
        self.title_label.setText(f"Название {file.title}")
        self.learn_model_button.setVisible(True)
        self.description_label.setText(f"Описание {file.description}")

class ModelsSettings(QWidget):
    def __init__(self, callback):
        super().__init__()
        self.db = DataBaseManager()
        # Главный горизонтальный layout
        main_layout = QHBoxLayout()

        # Левая часть - конфигурация сети
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Кнопка назад
        self.btn_back = QPushButton("Назад")
        self.callback = callback
        self.btn_back.clicked.connect(callback)
        left_layout.addWidget(self.btn_back)

        # Группа конфигурации слоев
        layer_group = QGroupBox("Конфигурация слоев")
        layer_layout = QVBoxLayout(layer_group)

        # Панель добавления нового слоя
        config_layout = QHBoxLayout()
        self.output_size = QLineEdit()
        self.output_size.setPlaceholderText("Введите размер")
        self.activation = QComboBox()
        self.activation.addItems(['sigmoid', 'relu', 'tanh', 'linear'])
        add_btn = QPushButton("Добавить слой")
        add_btn.clicked.connect(self.add_layer)
        delete_btn = QPushButton("Удалить слой")
        delete_btn.clicked.connect(self.delete_layer)
        config_layout.addWidget(QLabel("Размер слоя:"))
        config_layout.addWidget(self.output_size)
        config_layout.addWidget(QLabel("Функция активации:"))
        config_layout.addWidget(self.activation)
        config_layout.addWidget(add_btn)
        config_layout.addWidget(delete_btn)
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
        left_layout.addWidget(layer_group)

        # Правая часть - название и описание
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Группа метаданных
        meta_group = QGroupBox("Метаданные сети")
        meta_layout = QVBoxLayout(meta_group)

        # Поле названия
        meta_layout.addWidget(QLabel("Название сети:"))
        self.network_name = QLineEdit()
        self.network_name.setPlaceholderText("Введите название нейронной сети")
        meta_layout.addWidget(self.network_name)

        # Поле описания
        meta_layout.addWidget(QLabel("Описание:"))
        self.description = QTextEdit()
        self.description.setPlaceholderText("Введите описание архитектуры...")
        meta_layout.addWidget(self.description)

        # Кнопка сохранения
        save_btn = QPushButton("Сохранить конфигурацию")
        save_btn.clicked.connect(self.save_config)
        meta_layout.addWidget(save_btn)

        right_layout.addWidget(meta_group)

        # Добавляем обе панели в главный layout
        main_layout.addWidget(left_panel, stretch=2)
        main_layout.addWidget(right_panel, stretch=1)

        self.setLayout(main_layout)

    def save_config(self):
        title = self.network_name.text().strip()
        description = self.description.toPlainText().strip()

        if not title:
            QMessageBox.warning(self, "Ошибка", "Введите название сети")
            return

        if self.layers_table.rowCount() == 0:
            QMessageBox.warning(self, "Ошибка", "Добавьте хотя бы один слой")
            return

        layers = []
        for row in range(self.layers_table.rowCount()):
            size = int(self.layers_table.item(row, 0).text())
            activation = self.layers_table.item(row, 1).text()
            layers.append({
                "size": size,
                "activation": activation
            })
        parameters = json.dumps(layers, ensure_ascii=False)
        self.db.save_model(title,description, parameters)
        self.callback()


    def delete_layer(self):
        selected_row = self.layers_table.currentRow()
        if selected_row >= 0:
            self.layers_table.removeRow(selected_row)
        else:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, выберите строку для удаления")

    def add_layer(self):
        try:
            output_size = int(self.output_size.text())
            if output_size <= 0:
                raise ValueError("Размер слоя должен быть положительным числом")

            activation = self.activation.currentText()

            row = self.layers_table.rowCount()
            self.layers_table.insertRow(row)
            self.layers_table.setItem(row, 0, QTableWidgetItem(str(output_size)))
            self.layers_table.setItem(row, 1, QTableWidgetItem(activation))

            self.output_size.clear()

        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", f"Некорректные параметры слоя:\n{str(e)}")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainView()
    window.show()
    sys.exit(app.exec_())