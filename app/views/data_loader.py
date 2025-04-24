
import os
import pickle
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTableWidget, QLabel, QComboBox, QMessageBox, \
    QFileDialog, QTableWidgetItem, QHBoxLayout

from app.files_preprocessing.readers.csv_reader import CsvReader
from app.files_preprocessing.file_preprocessor import DataPreprocessor
from app.files_preprocessing.readers.excel_reader import ExcelReader
from app.views.neural_network_settings import NeuralNetworkGUI
from app.views.prediction_screen import PredictionWindow


class DataLoader(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Загрузка данных')
        self.setMinimumSize(800, 600)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(15)

        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial;
                font-size: 14px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 4px;
                min-width: 120px;
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
                padding: 5px;
                border: none;
            }
        """)

        self.back_button = QPushButton('← На главный экран')
        self.back_button.setObjectName("back_button")
        self.back_button.clicked.connect(self.back)

        self.load_button = QPushButton('Загрузить данные')
        self.load_button.clicked.connect(self.load_data)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)

        self.confirm_button = QPushButton('Подтвердить данные')
        self.confirm_button.setObjectName("confirm_button")
        self.confirm_button.clicked.connect(self.confirm_data)
        self.confirm_button.setEnabled(False)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.back_button)
        control_layout.addWidget(self.load_button)
        control_layout.addStretch()
        control_layout.addWidget(self.confirm_button)

        self.layout.addLayout(control_layout)
        self.layout.addWidget(self.table)

        self.setLayout(self.layout)

        self.preprocessor: DataPreprocessor = DataPreprocessor(CsvReader())

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

        _, extension = os.path.splitext(file_path)
        print(f"Тип файла: {extension}")

        if extension == ".csv":
            self.preprocessor.reader = CsvReader()
        elif extension == ".xlsx":
            self.preprocessor.reader = ExcelReader()
        else:
            QMessageBox.warning(self, "Ошибка", "Неподдерживаемый формат файла!")
            return

        try:
            self.preprocessor.read_data(file_path)
            self.display_data()
            self.confirm_button.setEnabled(True)
            QMessageBox.information(self, "Успех", "Данные успешно загружены!")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки данных:\n{str(e)}")

    def display_data(self):
        data = self.preprocessor.data
        self.table.clear()

        self.table.setRowCount(data.shape[0])
        self.table.setColumnCount(data.shape[1])
        self.table.setHorizontalHeaderLabels(data.columns.tolist())

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                item = QTableWidgetItem(str(data.iat[i, j]))
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)  # Делаем ячейки нередактируемыми
                self.table.setItem(i, j, item)

        self.table.resizeColumnsToContents()

    def confirm_data(self):
        if self.preprocessor.data is None:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, сначала загрузите данные!")
            return

        try:
            self.w = PredictionWindow(self.preprocessor)
            self.w.show()
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось обработать данные:\n{str(e)}")