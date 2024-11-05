from abc import ABC,abstractmethod
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler



class ModelBuilderStrategy(ABC):
    @abstractmethod
    def build_model(self,X_train):
        pass



class LSTMModelBuilder(ModelBuilderStrategy):
    def build_model(self,X_train):
        lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(units=50),
            Dense(units=1)
        ])
        return lstm_model
    

class RNNMdelBuilder(ModelBuilderStrategy):
    def build_model(self,X_train):
        rnn_model = Sequential([
            SimpleRNN(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            SimpleRNN(units=50),
            Dense(units=1)
        ])
        return rnn_model
    


class GRUModelBuilder(ModelBuilderStrategy):
    def build_model(self,X_train):
        gru_model = Sequential([
            GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            GRU(units=50),
            Dense(units=1)
        ])
        return gru_model
    


class ModelBuilder:
    def __init__(self, model_strategy: ModelBuilderStrategy):
        self.model_strategy = model_strategy

    def set_strategy(self,model_strategy:ModelBuilderStrategy):
        self.model_strategy = model_strategy

    def compile_model(self,X_train):
        model = self.model_strategy.build_model(X_train)
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    

if __name__=='__main__':
    data = pd.read_csv('D:\coding\ml\stock-price-predictor\src\components\data\AAPL_stock_data.csv')

    data = data[["Close"]]

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare training and testing datasets
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    # Create sequences for the models
    def create_sequences(data, time_step=60):
        x, y = [], []
        for i in range(time_step, len(data)):
            x.append(data[i-time_step:i, 0])
            y.append(data[i, 0])
        return np.array(x), np.array(y)

    time_step = 60
    x_train, y_train = create_sequences(train_data, time_step)
    x_test, y_test = create_sequences(test_data, time_step)

    # Reshape data for LSTM/RNN/GRU input
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    model_builder = ModelBuilder(LSTMModelBuilder())
    lstm_model = model_builder.compile_model(x_train)
    print(lstm_model.summary())

    model_builder.set_strategy(RNNMdelBuilder())
    rnn_model = model_builder.compile_model(x_train)
    print(rnn_model.summary())

    model_builder.set_strategy(GRUModelBuilder())
    gru_model = model_builder.compile_model(x_train)
    print(gru_model.summary())
