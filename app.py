import sqlite3
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime

import os

script_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(script_path))

from arima_ import * 
from create_db import *

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

DATABASE = '/home/myrto/Documents/FINT_project/forecast.db'

create_forecast_table(DATABASE)

def connect_to_db():
    conn = sqlite3.connect(DATABASE)
    return conn

def insert_forecast(forecast):
    conn = connect_to_db()
    cur = conn.cursor()
    # Convert the timestamp to string format
    timestamp_str = forecast['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
    try:
        cur.execute("INSERT INTO forecast (timestamp, value) VALUES (?, ?)", (timestamp_str, forecast['value']))
        conn.commit()
    except sqlite3.IntegrityError:
        # If a record with the same timestamp already exists, skip insertion
        pass
    finally:
        conn.close()


def run_arima_and_insert_forecast(prediction_days, train_days):
    forecast = main(prediction_days, train_days)
    forecast_series = pd.Series(forecast, index=forecast.index)
    for timestamp, value in forecast_series.items():
        insert_forecast({'timestamp': timestamp, 'value': value})


@app.route('/api/get_forecast', methods=['GET'])
def api_get_forecasts():
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM forecast")
    rows = cur.fetchall()
    forecasts = [{'timestamp': row[0], 'value': row[1]} for row in rows]
    conn.close()
    return jsonify(forecasts)

@app.route('/api/post_forecast', methods=['POST'])
def api_post_forecast():
    data = request.get_json()
    prediction_days = data.get('prediction_days')
    train_days = data.get('train_days')
    if prediction_days is None or train_days is None:
        return jsonify({'error': 'Prediction days and train days must be provided'}), 400
    
    forecast = run_arima_and_insert_forecast(prediction_days, train_days)
    
    return jsonify({'message': 'Forecast inserted successfully'}), 201

if __name__ == '__main__':
    run_arima_and_insert_forecast(prediction_days=2, train_days=120)  # Run ARIMA and insert forecast data into the database
    app.run(port=4000)
