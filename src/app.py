from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime

app = Flask(__name__)

# Load the trained model
data = pd.read_csv("D:/coding/ml/stock-price-predictor/src/components/data/AAPL_stock_data.csv")

model = load_model("D:/coding/ml/stock-price-predictor/src/model/stock_price_prediction_model.h5")
historical_prices = data["Close"].values.reshape(-1, 1)


# Initialize the scaler (set it up with the data range used during training)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(historical_prices)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Parse request JSON for the last 60 days of data
    data = request.get_json()
    last_60_days = np.array(data['last_60_days']).reshape(-1, 1)

    # Check if we have exactly 60 values, otherwise truncate or return an error
    if last_60_days.shape[0] < 60:
        return jsonify({'error': 'Please provide exactly 60 days of stock prices.'}), 400
    elif last_60_days.shape[0] > 60:
        last_60_days = last_60_days[-60:]  # Take only the last 60 days if more are provided

    # Scale the input data using the fitted scaler
    last_60_days_scaled = scaler.transform(last_60_days)
    input_data = np.reshape(last_60_days_scaled, (1, 60, 1))
    
    # Make predictions for the next 30 days
    predictions = []
    for _ in range(10):
        predicted_price_scaled = model.predict(input_data)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)
        predictions.append(predicted_price[0][0].item())
        
        # Update input_data with the new prediction
        input_data = np.append(input_data[:, 1:, :], [[predicted_price_scaled[0]]], axis=1)

    # Generate dates for the next 30 days
    last_date = datetime.datetime.now()
    future_dates = [(last_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 11)]

    # Send predictions with dates back as JSON
    return jsonify({'dates': future_dates, 'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)