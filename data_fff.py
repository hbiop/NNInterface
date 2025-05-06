import sys
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTableWidget, QTableWidgetItem,
                             QFileDialog, QSplitter, QVBoxLayout, QWidget, QLabel,
                             QComboBox, QPushButton)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        # Splitter for table and right panel
        self.splitter = QSplitter(QtCore.Qt.Horizontal)

        # Table setup
        self.table = QtWidgets.QTableWidget()
        self.splitter.addWidget(self.table)

        # Right panel setup
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)

        # Settings widgets
        self.column_label = QLabel("Selected Column: None")
        self.right_layout.addWidget(self.column_label)

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Histogram", "Line Chart", "Bar Chart"])
        self.right_layout.addWidget(self.plot_type_combo)

        self.update_plot_btn = QPushButton("Update Plot")
        self.update_plot_btn.clicked.connect(self.update_plot)
        self.right_layout.addWidget(self.update_plot_btn)

        # Matplotlib canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.right_layout.addWidget(self.canvas)

        self.splitter.addWidget(self.right_panel)
        self.splitter.setSizes([800, 400])

        # Main layout
        layout = QVBoxLayout(self.main_widget)
        layout.addWidget(self.splitter)

        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        open_action = file_menu.addAction('Open CSV')
        open_action.triggered.connect(self.load_csv)

        # Data storage
        self.df = None
        self.current_column = None

        # Connect header click
        self.table.horizontalHeader().sectionClicked.connect(self.on_header_clicked)

    def load_csv(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_name:
            self.df = pd.read_csv(file_name)
            self.populate_table()

    def populate_table(self):
        if self.df is not None:
            self.table.setRowCount(self.df.shape[0])
            self.table.setColumnCount(self.df.shape[1])
            self.table.setHorizontalHeaderLabels(self.df.columns)

            for row in range(self.df.shape[0]):
                for col in range(self.df.shape[1]):
                    item = QTableWidgetItem(str(self.df.iat[row, col]))
                    self.table.setItem(row, col, item)

    def on_header_clicked(self, index):
        self.current_column = self.df.columns[index]
        self.column_label.setText(f"Selected Column: {self.current_column}")
        self.update_plot()

    def update_plot(self):
        if self.df is None or self.current_column is None:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        plot_type = self.plot_type_combo.currentText()
        col_data = self.df[self.current_column]

        try:
            if plot_type == "Histogram":
                ax.hist(col_data.dropna(), bins=20)
                ax.set_title(f"Histogram of {self.current_column}")
            elif plot_type == "Line Chart":
                ax.plot(col_data)
                ax.set_title(f"Line Chart of {self.current_column}")
            elif plot_type == "Bar Chart":
                col_data.value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f"Bar Chart of {self.current_column}")

            ax.grid(True)
            self.canvas.draw()
        except Exception as e:
            print(f"Error plotting: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())