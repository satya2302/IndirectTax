from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def create_monthly_data():
    # Fetch data from the API
    response = requests.get('http://localhost:5078/api/Usage')
    response.raise_for_status()
    api_data = response.json()

    # Convert API data to DataFrame
    df = pd.DataFrame(api_data)
    # Rename columns to match expected names
    df = df.rename(columns={
        'year': 'Year',
        'month': 'Month',
        'transactions': 'Transactions',
        'taxReturns': 'TaxReturns',
        'eFilings': 'EFilings',
        'users_Transactions': 'Users_Transactions',
        'users_TaxReturns': 'Users_Return',
        'users_EFilings': 'Users_efile'
    })
    # Ensure Year is integer (handles both int and string input)
    df['Year'] = df['Year'].astype(int)
    # Create 'ds' column in format YYYY-MMM-01
    df['ds'] = df['Year'].astype(str) + '-' + df['Month'].str[:3].str.upper() + '-01'
    # Reorder columns
    df = df[['ds', 'Year', 'Month', 'Transactions', 'TaxReturns', 'EFilings', 'Users_Transactions', 'Users_Return', 'Users_efile']]
    return df

def month_to_num(month):
    months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE',
              'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']
    month = month.upper()
    for i, m in enumerate(months):
        if m.startswith(month[:3]):
            return i + 1
    return 1

def train_and_forecast_rf(data, column):
    # Prepare features: year, month as numeric
    data = data.copy()
    data['Year'] = data['ds'].str[:4].astype(int)
    data['MonthNum'] = data['Month'].apply(month_to_num)
    # Add lag features (previous 1, 2, 3 months)
    for lag in [1, 2, 3]:
        data[f'lag_{lag}'] = data[column].shift(lag)
    data = data.dropna().reset_index(drop=True)
    feature_cols = ['Year', 'MonthNum', 'lag_1', 'lag_2', 'lag_3']
    X = data[feature_cols]
    y = data[column]
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    # Prepare to predict for 2024
    last_known = data.iloc[-3:][column].tolist()  # last 3 known values
    months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE',
              'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']
    preds = []
    for i, m in enumerate(months):
        year = 2024
        month_num = i + 1
        # For first 3 months, use last known + predicted lags
        if i == 0:
            lags = last_known[-1], last_known[-2], last_known[-3]
        elif i == 1:
            lags = preds[-1], last_known[-1], last_known[-2]
        elif i == 2:
            lags = preds[-1], preds[-2], last_known[-1]
        else:
            lags = preds[-1], preds[-2], preds[-3]
        X_pred = pd.DataFrame([[year, month_num, lags[0], lags[1], lags[2]]], columns=['Year', 'MonthNum', 'lag_1', 'lag_2', 'lag_3'])
        pred = model.predict(X_pred)[0]
        preds.append(pred)
    result = pd.DataFrame({
        'ds': [f'2024-{m[:3]}-01' for m in months],
        'yhat': preds
    })
    return result

@app.route('/predict', methods=['GET'])
def predict():
    data = create_monthly_data()
    metrics = [
        ('Transactions', 'transactions'),
        ('TaxReturns', 'taxReturns'),
        ('EFilings', 'eFilings')
    ]
    # Predict for each metric
    predictions = {}
    for metric, api_key in metrics:
        pred_df = train_and_forecast_rf(data, metric)
        predictions[api_key] = pred_df['yhat'].tolist()
    # Build response in requested format
    months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE',
              'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']
    result = []
    for i, month in enumerate(months):
        result.append({
            "year": 2026,
            "month": month,
            "transactions": int(round(predictions['transactions'][i])),
            "taxReturns": int(round(predictions['taxReturns'][i])),
            "eFilings": int(round(predictions['eFilings'][i]))
          
        })
    return jsonify(result)

@app.route('/predict-plot/<metric>', methods=['GET'])
def predict_plot(metric):
    # Supported metrics
    metric_map = {
        'usertransactions': ('Users_Transactions', 'Users Transactions'),
        'usertaxreturns': ('Users_Return', 'Users Tax Returns'),
        'userefilings': ('Users_efile', 'Users EFilings')
    }
    if metric not in metric_map:
        return jsonify({'error': 'Invalid metric'}), 400
    col, label = metric_map[metric]
    data = create_monthly_data()
    # Group by year and month for all years
    data['date'] = pd.to_datetime(data['ds'])
    data = data.sort_values('date')
    # Plot all years
    plt.figure(figsize=(12, 6))
    for year in sorted(data['Year'].unique()):
        year_data = data[data['Year'] == year]
        plt.plot(year_data['date'], year_data[col], marker='o', label=f'{label} ({year})')
    plt.title(f'{label} for All Years')
    plt.xlabel('Month')
    plt.ylabel(label)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == "__main__":
     app.run(debug=True, port=5000, use_reloader=False)
