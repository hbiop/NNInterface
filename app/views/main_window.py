from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QSizePolicy

from app.views.choice_read_type_window import ChoiceReadTypeWindow
from app.views.creation_file import DataInputScreen
from app.views.data_viewer_window import DataViewer
from app.views.file_history_window import FileHistoryWindow
from app.views.prediction_screen import PredictionWindow
from app.views.data_loader import DataLoader
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Нейронная сеть")
        self.setMinimumSize(400, 300)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(50, 50, 50, 50)
        main_layout.setSpacing(30)

        title_label = QLabel("Нейронная сеть")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font: bold 24px;
                color: #2c3e50;
                margin-bottom: 40px;
            }
        """)

        self.train_btn = self.create_button("Обучить модель", "#3498db", self.on_train_clicked)
        self.predict_btn = self.create_button("Сделать предсказание", "#2ecc71", self.on_predict_clicked)
        self.create_file_btn = self.create_button("Создать файл", "#2ecc71", self.on_create_file_clicked)

        main_layout.addStretch(1)
        main_layout.addWidget(title_label)
        main_layout.addStretch(1)
        main_layout.addWidget(self.train_btn)
        main_layout.addWidget(self.predict_btn)
        main_layout.addWidget(self.create_file_btn)
        main_layout.addStretch(2)

        main_layout.setStretchFactor(title_label, 1)
        main_layout.setStretchFactor(self.train_btn, 2)
        main_layout.setStretchFactor(self.predict_btn, 2)
        main_layout.setStretchFactor(self.predict_btn, 2)

    def create_button(self, text, color, callback):
        button = QPushButton(text)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 15px;
                padding: 20px;
                font: bold 18px;
                min-height: 80px;
            }}
            QPushButton:hover {{
                background-color: {self.darken_color(color)};
            }}
        """)
        button.clicked.connect(callback)
        return button

    @staticmethod
    def darken_color(hex_color, factor=0.8):
        """Затемнение цвета для эффекта hover"""
        rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
        return f"#{int(rgb[0] * factor):02x}{int(rgb[1] * factor):02x}{int(rgb[2] * factor):02x}"

    def on_train_clicked(self):
        self.w = FileHistoryWindow()
        self.w.show()
        self.close()

    def on_create_file_clicked(self):
        self.w = DataInputScreen()
        self.w.show()
        self.close()

    def on_predict_clicked(self):
        self.w = PredictionWindow()
        self.w.show()
        self.close()