
import os
import pickle
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTableWidget, QLabel, QComboBox, QMessageBox, QFileDialog, QTableWidgetItem

from app.files_preprocessing.readers.csv_reader import CsvReader
from app.files_preprocessing.file_preprocessor import DataPreprocessor
from app.files_preprocessing.readers.excel_reader import ExcelReader
from app.views.neural_network_settings import NeuralNetworkGUI
from app.views.prediction_screen import PredictionWindow
class DataLoader(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Data Viewer')
        self.layout = QVBoxLayout()
        
        self.load_button = QPushButton('Загрузить данные')
        self.load_button.clicked.connect(self.load_data)
        self.back_button = QPushButton('На главный экран')
        self.back_button.clicked.connect(self.back)
        self.layout.addWidget(self.back_button)
        self.layout.addWidget(self.load_button)

        self.table = QTableWidget()
        self.layout.addWidget(self.table)

        self.confirm_button = QPushButton('Подтвердить данные')
        self.confirm_button.clicked.connect(self.confirm_data)
        self.layout.addWidget(self.confirm_button)

        self.preprocessor: DataPreprocessor = DataPreprocessor(CsvReader())
        self.setLayout(self.layout)

    def back(self):
        from app.views.main_window import MainWindow
        self.w = MainWindow()
        self.w.show()
        self.close()

    def load_data(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл c данными", "",
                                                   "CSV Files (*.csv);;All Files (*.xlsx)", options=options)
        _, extension = os.path.splitext(file_path)
        print(f"Тип файла: {extension}")
        if extension == ".csv":
            self.preprocessor.reader = CsvReader()
        if extension == ".xlsx":
            self.preprocessor.reader = ExcelReader()
        if file_path:
            self.preprocessor.read_data(file_path)
            self.display_data()

    

    def display_data(self):
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

        #self.preprocessor.preprocess_data_for_predict()
        self.w = PredictionWindow(self.preprocessor)
        self.w.show()
        self.close()