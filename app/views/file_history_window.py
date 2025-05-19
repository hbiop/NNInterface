from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QMessageBox

from app.database.database_manager import DatabaseManager
from app.views.data_viewer_window import DataViewer


class FileHistoryWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('История загруженных файлов')
        self.setMinimumSize(900, 700)

        # Стилизация (можно использовать тот же стиль, что и в DataViewer)
        self.setup_ui()
        self.load_file_history()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Панель управления
        control_panel = QHBoxLayout()

        self.back_button = QPushButton('← На главный экран')
        self.back_button.setObjectName("back_button")
        self.back_button.clicked.connect(self.back_to_main)

        self.new_file_button = QPushButton('Загрузить новый файл')
        self.new_file_button.clicked.connect(self.open_data_viewer)

        control_panel.addWidget(self.back_button)
        control_panel.addWidget(self.new_file_button)
        control_panel.addStretch()

        layout.addLayout(control_panel)

        # Таблица с историей файлов
        self.files_table = QTableWidget()
        self.files_table.setColumnCount(5)
        self.files_table.setHorizontalHeaderLabels([
            "ID", "Имя файла", "Тип", "Дата загрузки", "Размер"
        ])
        self.files_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.files_table.doubleClicked.connect(self.open_selected_file)

        layout.addWidget(self.files_table)
        self.setLayout(layout)

    def load_file_history(self):
        try:
            db = DatabaseManager()
            files = db.get_all_files()

            self.files_table.setRowCount(len(files))
            for row_idx, file_data in enumerate(files):
                for col_idx, value in enumerate(file_data):
                    item = QTableWidgetItem(str(value))
                    self.files_table.setItem(row_idx, col_idx, item)

            self.files_table.resizeColumnsToContents()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить историю файлов: {str(e)}")

    def open_selected_file(self):
        selected_row = self.files_table.currentRow()
        if selected_row >= 0:
            file_id = int(self.files_table.item(selected_row, 0).text())
            self.open_data_viewer(file_id)

    def open_data_viewer(self, file_id=None):
        self.data_viewer = DataViewer(file_id)
        self.data_viewer.show()
        self.close()

    def back_to_main(self):
        from app.views.main_window import MainWindow
        self.w = MainWindow()
        self.w.show()
        self.close()