import json
import sqlite3
from datetime import datetime
import os
import pickle


class DatabaseManager:
    def __init__(self, db_path='data_files.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Таблица для хранения информации о файлах
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    upload_date TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    original_data BLOB,
                    file_path TEXT,
                    description TEXT
                )
            ''')

            # Таблица для хранения предобработанных данных
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS preprocessed_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    preprocess_date TEXT NOT NULL,
                    target_column TEXT NOT NULL,
                    preprocessor BLOB NOT NULL,
                    data_sample BLOB,
                    FOREIGN KEY(file_id) REFERENCES files(id)
                )
            ''')

            conn.commit()

    def save_file(self, file_path, file_type, data_frame=None):
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        upload_date = datetime.now().isoformat()

        # Сериализация данных, если они переданы
        original_data = None
        if data_frame is not None:
            original_data = pickle.dumps(data_frame)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO files 
                (filename, file_type, upload_date, file_size, original_data, file_path)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (filename, file_type, upload_date, file_size, original_data, file_path))

            conn.commit()
            return cursor.lastrowid

    def save_preprocessed_data(self, file_id, target_column, preprocessor, data_sample=None):
        preprocess_date = datetime.now().isoformat()
        preprocessor_blob = pickle.dumps(preprocessor)
        data_sample_blob = pickle.dumps(data_sample.head(100)) if data_sample is not None else None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO preprocessed_data 
                (file_id, preprocess_date, target_column, preprocessor, data_sample)
                VALUES (?, ?, ?, ?, ?)
            ''', (file_id, preprocess_date, target_column, preprocessor_blob, data_sample_blob))

            conn.commit()
            return cursor.lastrowid

    def get_all_files(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, filename, file_type, 
                       strftime('%Y-%m-%d %H:%M', upload_date) as upload_date, 
                       file_size
                FROM files
                ORDER BY upload_date DESC
            ''')
            return cursor.fetchall()

    def get_file_data(self, file_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM files WHERE id = ?', (file_id,))
            file_data = cursor.fetchone()

            if file_data and file_data[5]:  # original_data
                return pickle.loads(file_data[5])
            elif file_data and file_data[6]:  # file_path
                return file_data[6]
            return None

    def get_preprocessed_data(self, file_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM preprocessed_data 
                WHERE file_id = ?
                ORDER BY preprocess_date DESC
                LIMIT 1
            ''', (file_id,))
            data = cursor.fetchone()

            if data:
                return {
                    'preprocessor': pickle.loads(data[4]),
                    'data_sample': pickle.loads(data[5]) if data[5] else None
                }
            return None