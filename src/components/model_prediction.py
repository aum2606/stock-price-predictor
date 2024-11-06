import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import numpy as np

def predict_and_plot(model, X_test, y_test, scaler, title="Model Prediction"):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_scaled, label="True Price")
    plt.plot(predictions, label="Predicted Price")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def evaluation_metrics(X_test,y_test,scaler,model):
    predictions = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predictions)
    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(real_prices, predicted_prices)
    mae = mean_absolute_error(real_prices, predicted_prices)
    r2 = r2_score(real_prices, predicted_prices)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R^2 Score: {r2}")
    return mse,mae,r2


def plot(X_test,y_test,training_data_len,model,data):
    predictions = model.predict(X_test)
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    # Visualize the data
    plt.figure(figsize=(16,6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()


def predict_for_future(model,scaled_data,days_to_predict,time_step,scaler):
    # Get the last 60 days of data
    last_sequence = scaled_data[-time_step:]
    last_sequence = last_sequence.reshape((1,time_step,1))
    days_to_predict = 10
    predicted_prices = []
    for i in range(days_to_predict):
        predicted_price_scaled = model.predict(last_sequence)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)

        # Save the predicted price
        predicted_prices.append(predicted_price[0][0])

        # Update the sequence with the predicted price
        new_sequence = np.append(last_sequence[:, 1:, :], [[predicted_price_scaled[0]]], axis=1)
        last_sequence = new_sequence

    plt.figure(figsize=(12, 6))
    plt.plot(predicted_prices, label="Predicted Prices")
    plt.title("Predicted Stock Prices for the Next 10 Days")
    plt.xlabel("Day")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()
    return predicted_prices
