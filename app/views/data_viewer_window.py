import os

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTableWidget, QLabel, QComboBox, QMessageBox, QFileDialog, QTableWidgetItem

from app.files_preprocessing.readers.csv_reader import CsvReader
from app.files_preprocessing.file_preprocessor import DataPreprocessor
from app.files_preprocessing.readers.excel_reader import ExcelReader
from app.views.neural_network_settings import NeuralNetworkGUI


class DataViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Data Viewer')
        self.layout = QVBoxLayout()
        
        # Кнопка загрузки данных
        self.load_button = QPushButton('Загрузить данные')
        self.load_button.clicked.connect(self.load_data)
        self.layout.addWidget(self.load_button)

        # ComboBox для выбора целевого столбца
        self.target_column_label = QLabel("Выберите целевой столбец:")
        self.target_column_combo = QComboBox()
        self.layout.addWidget(self.target_column_label)
        self.layout.addWidget(self.target_column_combo)

        # Таблица с данными
        self.table = QTableWidget()
        self.layout.addWidget(self.table)

        # Кнопка подтверждения
        self.confirm_button = QPushButton('Подтвердить данные')
        self.confirm_button.clicked.connect(self.confirm_data)
        self.layout.addWidget(self.confirm_button)

        self.preprocessor: DataPreprocessor = DataPreprocessor(CsvReader())
        self.setLayout(self.layout)

    def load_data(self):
        # ... (существующий код загрузки данных)
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл c данными", "",
                                                   "CSV Files (*.csv);;All Files (*)", options=options)
        _, extension = os.path.splitext(file_path)
        print(f"Тип файла: {extension}")
        if extension == ".csv":
            self.preprocessor.reader = CsvReader()
        if extension == ".xlsx":
            self.preprocessor.reader = ExcelReader()
        if file_path:
            self.preprocessor.read_data(file_path)
            self.display_data()
            # Обновляем ComboBox после загрузки данных
            self.update_target_columns()

    def update_target_columns(self):
        """Обновляет список столбцов в выпадающем списке"""
        self.target_column_combo.clear()
        if self.preprocessor.data is not None:
            columns = self.preprocessor.data.columns.tolist()
            self.target_column_combo.addItems(columns)

    def display_data(self):
        # Обновить таблицу с загруженными данными
        self.table.setRowCount(self.preprocessor.data.shape[0])
        self.table.setColumnCount(self.preprocessor.data.shape[1])
        self.table.setHorizontalHeaderLabels(self.preprocessor.data.columns.tolist())

        for i in range(self.preprocessor.data.shape[0]):
            for j in range(self.preprocessor.data.shape[1]):
                self.table.setItem(i, j, QTableWidgetItem(str(self.preprocessor.data.iat[i, j])))
    def confirm_data(self):
        if self.preprocessor.data is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные!")
            return

        selected_column = self.target_column_combo.currentText()
        print(selected_column)
        if not selected_column:
            QMessageBox.warning(self, "Ошибка", "Выберите целевой столбец!")
            return

        # Разделяем данные на признаки и целевую переменную
        
        
        # Предобработка данных
        self.preprocessor.automatic_preprocess_data(selected_column)
        
        # Передаем данные в нейросеть
        self.w = NeuralNetworkGUI(self.preprocessor)
        self.w.show()