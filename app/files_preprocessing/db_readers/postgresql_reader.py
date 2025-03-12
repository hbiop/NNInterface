import psycopg2

from app.files_preprocessing.readers.base_reader import DataReader
from PyQt5.QtWidgets import QMessageBox



class PostgresqlReader(DataReader):
    def read_data(self,host, port, database, user, password, table):
        try:
            connection = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password
            )
            cursor = connection.cursor()
            cursor.execute(f"SELECT * from {table}")
            record = cursor.fetchall()
            print("Результат", record)
            connection.close()
        except:
            return "Произошла ошибка"