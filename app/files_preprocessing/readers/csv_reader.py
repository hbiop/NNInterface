import pandas as pd

from app.files_preprocessing.readers.base_reader import DataReader




class CsvReader(DataReader):
    def read_data(self, url):
        return pd.read_csv(url)
