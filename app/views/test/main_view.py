import sys

from PyQt5.QtWidgets import QHBoxLayout, QMainWindow, QGridLayout, QTextEdit, QPushButton, QApplication, QWidget, \
    QVBoxLayout, QSizePolicy, QStackedLayout, QStackedWidget, QLabel, QLineEdit


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

class Widget2(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.username_label = QLabel("AAAAAAAAAAA пользователя:")
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
        self.stacked_widget.addWidget(Widget2())
        self.layout.addWidget(self.horizontal_widget)
        self.layout.addWidget(self.stacked_widget)
        self.central_widget.setLayout(self.layout)

    def go_to_main_page(self):
        self.stacked_widget.setCurrentIndex(0)

    def go_to_dataset_page(self):
        self.stacked_widget.setCurrentIndex(1)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainView()
    window.show()
    sys.exit(app.exec_())