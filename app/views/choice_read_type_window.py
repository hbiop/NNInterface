from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QSizePolicy

from app.views.data_viewer_window import DataViewer


class ChoiceReadTypeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Assistant")
        self.setMinimumSize(400, 300)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(50, 50, 50, 50)
        main_layout.setSpacing(30)

        title_label = QLabel("Выберите откуда считать данные")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font: bold 24px;
                color: #2c3e50;
                margin-bottom: 40px;
            }
        """)

        self.train_btn = self.create_button("Из файла", "#3498db", self.on_train_clicked)
        self.predict_btn = self.create_button("Из базы данных", "#2ecc71", self.on_predict_clicked)
        self.back_btn = self.create_button("Назад", "#cc2f39", self.back)
        main_layout.addStretch(1)
        main_layout.addWidget(title_label)
        main_layout.addStretch(1)
        main_layout.addWidget(self.train_btn)
        main_layout.addWidget(self.predict_btn)
        main_layout.addWidget(self.back_btn)
        main_layout.addStretch(2)

        main_layout.setStretchFactor(title_label, 1)
        main_layout.setStretchFactor(self.train_btn, 2)
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
    def back(self):
        from app.views.main_window import sadfsafasfsafWindow
        self.w = sadfsafasfsafWindow()
        self.w.show()
        self.close()
    @staticmethod
    def darken_color(hex_color, factor=0.8):
        """Затемнение цвета для эффекта hover"""
        rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
        return f"#{int(rgb[0] * factor):02x}{int(rgb[1] * factor):02x}{int(rgb[2] * factor):02x}"


    def on_train_clicked(self):
        self.w = DataViewer()
        self.w.show()
        self.close()

    def on_predict_clicked(self):
        print("Запуск процесса предсказания...")
        # Здесь можно добавить логику предсказания