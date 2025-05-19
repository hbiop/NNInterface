import sqlite3
from datetime import datetime


class TrainingLogger:
    def __init__(self, db_path='training_history.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    epochs INTEGER NOT NULL,
                    learning_rate REAL NOT NULL,
                    layers_config TEXT NOT NULL,
                    final_loss REAL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_epochs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    epoch INTEGER NOT NULL,
                    loss REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES training_sessions(id)
                )
            ''')
            conn.commit()

    def start_new_session(self, epochs, lr, layers_config):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO training_sessions 
                (start_time, epochs, learning_rate, layers_config)
                VALUES (?, ?, ?, ?)
            ''', (datetime.now().isoformat(), epochs, lr, str(layers_config)))
            conn.commit()
            return cursor.lastrowid

    def log_epoch(self, session_id, epoch, loss):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO training_epochs 
                (session_id, epoch, loss, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (session_id, epoch, loss, datetime.now().isoformat()))
            conn.commit()

    def complete_session(self, session_id, final_loss):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE training_sessions 
                SET end_time = ?, final_loss = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), final_loss, session_id))
            conn.commit()