import pandas as pd

from app.files_preprocessing.readers.base_reader import DataReader

class ExcelReader(DataReader):
    def read_data(self, url):
        return pd.read_excel(url)
