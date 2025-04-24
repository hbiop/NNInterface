import sys
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTableWidget, QTableWidgetItem,
    QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog, QHBoxLayout, QLineEdit, QMessageBox
)


class DataInputScreen(QMainWindow):
    def __init__(self):
        super().__init__()

        # Настройки окна
        self.setWindowTitle("Ручной ввод данных")
        self.setGeometry(100, 100, 900, 700)

        # Основной виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Главный layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Стилизация
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fa;
                font-family: 'Segoe UI', Arial;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
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
            QPushButton#save_button {
                background-color: #2196F3;
            }
            QPushButton#save_button:hover {
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
                font-size: 14px;
                font-weight: bold;
                color: #333;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                min-width: 200px;
            }
        """)

        # Кнопка возврата
        self.back_button = QPushButton("← Назад")
        self.back_button.setObjectName("back_button")
        self.back_button.clicked.connect(self.btn_back_clicked)

        # Заголовок
        title_label = QLabel("Введите данные для нейронной сети:")
        title_label.setStyleSheet("font-size: 16px; color: #2c3e50;")

        # Таблица данных
        self.table_widget = QTableWidget()
        self.table_widget.setRowCount(0)
        self.table_widget.setColumnCount(0)
        self.table_widget.setAlternatingRowColors(True)
        self.table_widget.setSelectionBehavior(QTableWidget.SelectItems)

        # Панель управления таблицей
        control_panel = QHBoxLayout()

        self.column_name_input = QLineEdit()
        self.column_name_input.setPlaceholderText("Введите название столбца")

        self.add_column_button = QPushButton("Добавить столбец")
        self.add_column_button.clicked.connect(self.add_column)

        self.add_row_button = QPushButton("Добавить строку")
        self.add_row_button.clicked.connect(self.add_row)

        control_panel.addWidget(self.column_name_input)
        control_panel.addWidget(self.add_column_button)
        control_panel.addWidget(self.add_row_button)

        # Кнопка сохранения
        self.save_button = QPushButton("Сохранить в Excel")
        self.save_button.setObjectName("save_button")
        self.save_button.clicked.connect(self.save_to_excel)

        # Добавление элементов в layout
        main_layout.addWidget(self.back_button)
        main_layout.addWidget(title_label)
        main_layout.addWidget(self.table_widget)
        main_layout.addLayout(control_panel)
        main_layout.addWidget(self.save_button, 0, Qt.AlignRight)

    def btn_back_clicked(self):
        from app.views.main_window import MainWindow
        self.w = MainWindow()
        self.w.show()
        self.close()

    def add_row(self):
        current_row_count = self.table_widget.rowCount()
        self.table_widget.insertRow(current_row_count)

        # Заполняем новые ячейки пустыми значениями
        for col in range(self.table_widget.columnCount()):
            self.table_widget.setItem(current_row_count, col, QTableWidgetItem(""))

    def add_column(self):
        column_name = self.column_name_input.text().strip()
        if not column_name:
            QMessageBox.warning(self, "Ошибка", "Введите название столбца!")
            return

        current_column_count = self.table_widget.columnCount()
        self.table_widget.insertColumn(current_column_count)
        self.table_widget.setHorizontalHeaderItem(current_column_count, QTableWidgetItem(column_name))
        self.column_name_input.clear()

        # Заполняем новый столбец пустыми значениями
        for row in range(self.table_widget.rowCount()):
            self.table_widget.setItem(row, current_column_count, QTableWidgetItem(""))

    def save_to_excel(self):
        if self.table_widget.columnCount() == 0:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения!")
            return

        try:
            # Собираем данные из таблицы
            data = []
            for row in range(self.table_widget.rowCount()):
                row_data = []
                for column in range(self.table_widget.columnCount()):
                    item = self.table_widget.item(row, column)
                    row_data.append(item.text() if item else "")
                data.append(row_data)

            # Получаем заголовки столбцов
            column_headers = []
            for col in range(self.table_widget.columnCount()):
                header_item = self.table_widget.horizontalHeaderItem(col)
                column_headers.append(header_item.text() if header_item else f"Столбец {col + 1}")

            # Создаем DataFrame
            df = pd.DataFrame(data, columns=column_headers)

            # Диалог сохранения файла
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить файл Excel",
                "",
                "Файлы Excel (*.xlsx);;Все файлы (*)"
            )

            if file_name:
                if not file_name.endswith('.xlsx'):
                    file_name += '.xlsx'

                df.to_excel(file_name, index=False)

                # Красивое сообщение об успешном сохранении
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("Сохранение завершено")
                msg.setText(f"Данные успешно сохранены в файл:\n{file_name}")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка сохранения",
                                 f"Не удалось сохранить файл:\n{str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataInputScreen()
    window.show()
    sys.exit(app.exec_())