from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import joblib
app = Flask(__name__)

# Load the pre-trained LSTM model
model = load_model('D:/coding/ml/stock-price-predictor/src/model/stock_price_prediction_model.h5')
model.summary()
# Initialize MinMaxScaler
scaler = joblib.load('scaler.pkl')
# scaler = MinMaxScaler(feature_range=(0,1))
# Function to generate future date labels
def generate_future_dates(last_date, days=30):
    return [last_date + timedelta(days=i) for i in range(1, days + 1)]

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data (last 60 days of stock prices)
        data = request.json.get('last_60_days')
        
        # Convert to numpy array and reshape for scaling
        data = np.array(data).reshape(-1, 1)
        
        # Scale the input data using the fitted scaler
        data_scaled = scaler.transform(data)
        
        # Prepare the input data for the LSTM model (only last 60 timesteps)
        input_data = np.reshape(data_scaled[-60:], (1, 60, 1))  # Shape (1, 60, 1)
        
        predictions = []
        
        # Generate the next 30 days of predictions by predicting once for all 30 days
        for i in range(30):
            print(f"Shape of input_data before prediction: {input_data.shape}")  # Debug print
            
            # Predict next value
            pred = model.predict(input_data)  # Predict next value
            predictions.append(pred[0][0])  # Append predicted value
            
            # Do NOT feed back predicted values for the next prediction (keep it constant)
            if i == 0:
                predicted_value = pred[0][0]
        
        # Inverse transform the predictions to get them back to the original scale
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Generate future dates (30 days from today)
        last_date = datetime.today()
        future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
        
        # Format dates as strings
        date_strings = [date.strftime('%Y-%m-%d') for date in future_dates]
        
        # Return the predictions and corresponding dates as JSON response
        return jsonify({
            'predicted_prices': predictions.flatten().tolist(),
            'dates': date_strings
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
