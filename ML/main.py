from flask import Flask, request, jsonify
import pandas as pd
from prophet import Prophet

app = Flask(__name__)

# Sample data: replace this with your real multi-month dataset
def load_data():
    # Simulated monthly sales data
    data = {
        'ds': pd.date_range(start='2022-01-01', periods=24, freq='MS'),
        'y': [1000 + i*50 + (i%3)*200 for i in range(24)]
    }
    return pd.DataFrame(data)

# Train the model
def train_model(data):
    model = Prophet()
    model.fit(data)
    return model

# Predict future sales
def make_forecast(model, periods=6):
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(periods)

@app.route('/forecast', methods=['GET'])
def forecast():
    periods = int(request.args.get('periods', 6))  # default: next 6 months
    data = load_data()
    model = train_model(data)
    forecast_df = make_forecast(model, periods)
    result = forecast_df.to_dict(orient='records')
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
