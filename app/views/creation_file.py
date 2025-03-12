import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTableWidget, QTableWidgetItem,
    QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog, QHBoxLayout, QLineEdit, QMessageBox
)

class DataInputScreen(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Ввод данных для нейронной сети")
        self.setGeometry(100, 100, 800, 600)

        self.table_widget = QTableWidget(self)
        self.table_widget.setRowCount(0)  # Изначально нет строк
        self.table_widget.setColumnCount(0)  # Изначально нет столбцов

        # Кнопки для добавления строк и столбцов
        self.add_row_button = QPushButton("Добавить строку")
        self.add_row_button.clicked.connect(self.add_row)

        self.add_column_button = QPushButton("Добавить столбец")
        self.add_column_button.clicked.connect(self.add_column)

        # Кнопка для сохранения в Excel
        self.save_button = QPushButton("Сохранить в Excel")
        self.save_button.clicked.connect(self.save_to_excel)

        # Окно для ввода заголовка столбца
        self.column_name_input = QLineEdit(self)
        self.column_name_input.setPlaceholderText("Введите имя столбца")
        self.back_button = QPushButton("Назад")
        self.back_button.clicked.connect(self.btn_back_clicked)
        # Вертикальный макет
        layout = QVBoxLayout()
        layout.addWidget(self.back_button)
        layout.addWidget(QLabel("Введите данные в таблицу:"))
        layout.addWidget(self.table_widget)

        # Макет для кнопок
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.column_name_input)
        button_layout.addWidget(self.add_column_button)
        button_layout.addWidget(self.add_row_button)
        layout.addLayout(button_layout)
        layout.addWidget(self.save_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
    def btn_back_clicked(self):
        from app.views.main_window import MainWindow
        self.w = MainWindow()
        self.w.show()
        self.close()

    def add_row(self):
        current_row_count = self.table_widget.rowCount()
        self.table_widget.insertRow(current_row_count)  # Добавляем новую строку

    def add_column(self):
        column_name = self.column_name_input.text().strip()
        if column_name:
            current_column_count = self.table_widget.columnCount()
            self.table_widget.insertColumn(current_column_count)  # Добавляем новый столбец
            self.table_widget.setHorizontalHeaderItem(current_column_count, QTableWidgetItem(column_name))  # Устанавливаем заголовок

            # Очищаем поле ввода имени столбца
            self.column_name_input.clear()

    def save_to_excel(self):
        # Получаем данные из таблицы
        data = []
        for row in range(self.table_widget.rowCount()):
            row_data = []
            for column in range(self.table_widget.columnCount()):
                item = self.table_widget.item(row, column)
                row_data.append(item.text() if item else "")
            data.append(row_data)

        # Создаем DataFrame и сохраняем в Excel
        column_headers = []
        for col in range(self.table_widget.columnCount()):
            header_item = self.table_widget.horizontalHeaderItem(col)
            column_headers.append(header_item.text() if header_item else "")

        df = pd.DataFrame(data, columns=column_headers)
        file_name, _ = QFileDialog.getSaveFileName(self, "Сохранить файл", "", "Excel files (*.xlsx);;All files (*.*)")

        if file_name:
            df.to_excel(file_name, index=False)
            message_box = QMessageBox()
            message_box.setIcon(QMessageBox.Information)
            message_box.setText(f"Файл сохранён в {file_name}") 
            message_box.setWindowTitle("Information MessageBox") 
            message_box.setStandardButtons(QMessageBox.Ok) 
            message_box.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataInputScreen()
    window.show()
    sys.exit(app.exec_())