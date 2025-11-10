from pymongo import MongoClient
from datetime import datetime
import os

# class Database:
#     def __init__(self):
#         # MongoDB connection
#         # Use environment variable or default to local
#         mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
#         self.client = MongoClient(mongo_uri)
#         self.db = self.client['stock_forecaster']
        
#         # Collections
#         self.historical = self.db['historical_data']
#         self.forecasts = self.db['forecasts']
#         self.metadata = self.db['metadata']
        
#         # Create indexes
#         self.historical.create_index([('symbol', 1), ('timestamp', -1)])
#         self.forecasts.create_index([('symbol', 1), ('timestamp', -1)])
    
#     def store_historical_data(self, data_list):
#         """Store historical price data"""
#         if data_list:
#             # Remove duplicates
#             for item in data_list:
#                 self.historical.update_one(
#                     {'symbol': item['symbol'], 'timestamp': item['timestamp']},
#                     {'$set': item},
#                     upsert=True
#                 )
#         return True
    
#     def get_historical_data(self, symbol, limit=None):
#         """Retrieve historical data for a symbol"""
#         query = {'symbol': symbol}
#         cursor = self.historical.find(query).sort('timestamp', -1)
        
#         if limit:
#             cursor = cursor.limit(limit)
        
#         data = list(cursor)
        
#         # Remove MongoDB _id field
#         for item in data:
#             item.pop('_id', None)
        
#         return data
    
#     def store_forecast(self, forecast_data):
#         """Store forecast results"""
#         result = self.forecasts.insert_one(forecast_data)
#         return str(result.inserted_id)
    
#     def get_forecasts(self, symbol, limit=10):
#         """Retrieve recent forecasts for a symbol"""
#         query = {'symbol': symbol}
#         cursor = self.forecasts.find(query).sort('timestamp', -1).limit(limit)
        
#         forecasts = list(cursor)
#         for item in forecasts:
#             item.pop('_id', None)
        
#         return forecasts
    
#     def store_metadata(self, key, value):
#         """Store metadata"""
#         self.metadata.update_one(
#             {'key': key},
#             {'$set': {'value': value, 'updated_at': datetime.now()}},
#             upsert=True
#         )
    
#     def get_metadata(self, key):
#         """Retrieve metadata"""
#         result = self.metadata.find_one({'key': key})
#         return result['value'] if result else None
    
#     def clear_old_data(self, symbol, days=365):
#         """Clear data older than specified days"""
#         from datetime import timedelta
#         cutoff_date = datetime.now() - timedelta(days=days)
        
#         result = self.historical.delete_many({
#             'symbol': symbol,
#             'timestamp': {'$lt': cutoff_date.isoformat()}
#         })
        
#         return result.deleted_count


# Alternative: SQL-based Database
class SQLDatabase:
    """Alternative implementation using SQLite/PostgreSQL"""
    
    def __init__(self, db_path='stock_forecaster.db'):
        import sqlite3
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()
    
    def create_tables(self):
        cursor = self.conn.cursor()
        
        # Historical data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                UNIQUE(symbol, timestamp)
            )
        ''')
        
        # Forecasts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                model TEXT NOT NULL,
                horizon INTEGER,
                timestamp TEXT NOT NULL,
                predictions TEXT,
                rmse REAL,
                mae REAL,
                mape REAL
            )
        ''')
        
        self.conn.commit()
    
    def store_historical_data(self, data_list):
        cursor = self.conn.cursor()
        
        for item in data_list:
            cursor.execute('''
                INSERT OR REPLACE INTO historical_data 
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                item['symbol'],
                item['timestamp'],
                item['open'],
                item['high'],
                item['low'],
                item['close'],
                item['volume']
            ))
        
        self.conn.commit()
        return True
    
    def get_historical_data(self, symbol, limit=None):
        cursor = self.conn.cursor()
        
        query = '''
            SELECT symbol, timestamp, open, high, low, close, volume
            FROM historical_data
            WHERE symbol = ?
            ORDER BY timestamp DESC
        '''
        
        if limit:
            query += f' LIMIT {limit}'
        
        cursor.execute(query, (symbol,))
        rows = cursor.fetchall()
        
        data = []
        for row in rows:
            data.append({
                'symbol': row[0],
                'timestamp': row[1],
                'open': row[2],
                'high': row[3],
                'low': row[4],
                'close': row[5],
                'volume': row[6]
            })
        
        return data
    
    def store_forecast(self, forecast_data):
        import json
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO forecasts 
            (symbol, model, horizon, timestamp, predictions, rmse, mae, mape)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            forecast_data['symbol'],
            forecast_data['model'],
            forecast_data['horizon'],
            forecast_data['timestamp'],
            json.dumps(forecast_data['predictions']),
            forecast_data['metrics']['rmse'],
            forecast_data['metrics']['mae'],
            forecast_data['metrics']['mape']
        ))
        
        self.conn.commit()
        return cursor.lastrowid