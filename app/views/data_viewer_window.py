import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTableWidget, QLabel, QComboBox, QMessageBox, \
    QFileDialog, QTableWidgetItem, QApplication, QProgressDialog, QGroupBox, QHBoxLayout, QHeaderView

from app.files_preprocessing.readers.csv_reader import CsvReader
from app.files_preprocessing.file_preprocessor import DataPreprocessor
from app.files_preprocessing.readers.excel_reader import ExcelReader
from app.views.neural_network_settings import NeuralNetworkGUI


class DataViewer(QWidget):
    def __init__(self):
        super().__init__()

        # Настройки окна
        self.setWindowTitle('Просмотр и подготовка данных')
        self.setMinimumSize(900, 700)

        # Основной layout
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(15)

        # Стилизация
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial;
                font-size: 14px;
                color: #333;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 4px;
                min-width: 150px;
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
            QPushButton#confirm_button {
                background-color: #2196F3;
            }
            QPushButton#confirm_button:hover {
                background-color: #0b7dda;
            }
            QTableWidget {
                background-color: white;
                border: 1px solid #ddd;
                gridline-color: #eee;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
            QLabel {
                font-weight: bold;
                margin-bottom: 5px;
            }
            QComboBox {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                min-width: 200px;
            }
            QGroupBox {
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 20px;
                font-weight: bold;
                background-color: white;
            }
        """)

        # Панель управления
        control_panel = QHBoxLayout()

        self.back_button = QPushButton('← На главный экран')
        self.back_button.setObjectName("back_button")
        self.back_button.clicked.connect(self.back)

        self.load_button = QPushButton('Загрузить данные')
        self.load_button.clicked.connect(self.load_data)

        control_panel.addWidget(self.back_button)
        control_panel.addWidget(self.load_button)
        control_panel.addStretch()

        self.layout.addLayout(control_panel)

        # Группа выбора целевого столбца
        target_group = QGroupBox("Настройка целевой переменной")
        target_layout = QVBoxLayout()

        self.target_column_label = QLabel("Целевой столбец:")
        self.target_column_combo = QComboBox()
        self.target_column_combo.setPlaceholderText("Выберите столбец")

        target_layout.addWidget(self.target_column_label)
        target_layout.addWidget(self.target_column_combo)
        target_group.setLayout(target_layout)

        self.layout.addWidget(target_group)

        # Таблица данных
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSortingEnabled(True)

        self.layout.addWidget(self.table)

        # Кнопка подтверждения
        self.confirm_button = QPushButton('Подтвердить и продолжить')
        self.confirm_button.setObjectName("confirm_button")
        self.confirm_button.clicked.connect(self.confirm_data)
        self.confirm_button.setEnabled(False)

        self.layout.addWidget(self.confirm_button, 0, Qt.AlignRight)

        self.setLayout(self.layout)

        # Инициализация препроцессора
        self.preprocessor = DataPreprocessor(CsvReader())

    def back(self):
        from app.views.main_window import MainWindow
        self.w = MainWindow()
        self.w.show()
        self.close()

    def load_data(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл с данными",
            "",
            "CSV файлы (*.csv);;Excel файлы (*.xlsx);;Все файлы (*)",
            options=options
        )

        if not file_path:
            return

        try:
            _, extension = os.path.splitext(file_path)
            print(f"Тип файла: {extension}")

            if extension == ".csv":
                self.preprocessor.reader = CsvReader()
            elif extension == ".xlsx":
                self.preprocessor.reader = ExcelReader()
            else:
                raise ValueError("Неподдерживаемый формат файла")

            self.preprocessor.read_data(file_path)
            self.display_data()
            self.update_target_columns()
            self.confirm_button.setEnabled(True)

            QMessageBox.information(self, "Успех", "Данные успешно загружены!")

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

        # Настройка таблицы
        self.table.setRowCount(data.shape[0])
        self.table.setColumnCount(data.shape[1])
        self.table.setHorizontalHeaderLabels(data.columns.tolist())

        # Заполнение данными
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                item = QTableWidgetItem(str(data.iat[i, j]))
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.table.setItem(i, j, item)

        # Автоматическое выравнивание столбцов
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

    def confirm_data(self):
        if self.preprocessor.data is None:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, сначала загрузите данные!")
            return

        selected_column = self.target_column_combo.currentText()
        if not selected_column:
            QMessageBox.warning(self, "Ошибка", "Необходимо выбрать целевой столбец!")
            return

        try:
            # Прогресс-диалог
            progress = QProgressDialog("Подготовка данных...", "Отмена", 0, 0, self)
            progress.setWindowTitle("Обработка данных")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            QApplication.processEvents()  # Обновляем UI

            # Предобработка данных
            self.preprocessor.automatic_preprocess_data(selected_column)

            progress.close()

            # Переход к обучению модели
            self.w = NeuralNetworkGUI(self.preprocessor)
            self.w.show()
            self.close()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка обработки",
                                 f"Не удалось подготовить данные:\n{str(e)}")