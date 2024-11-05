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
    # data = pd.read_csv('D:\coding\ml\stock-price-predictor\src\components\data\AAPL_stock_data.csv')

    # data = data[["Close"]]



    # data_preprocessor = PreProcessor(MinMaxScaling(feature_range=(0,1)))
    # scaled_data = data_preprocessor.apply_data_preprocessing(data=data)
    # print(scaled_data.shape)



    # splitter = DataSplitter(strategy=CustomDataSplittingStrategy())
    # X_train,X_test,y_train,y_test = splitter.split(scaled_data)
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)

    # model_builder = ModelBuilder(LSTMModelBuilder())
    # lstm_model = model_builder.compile_model(X_train)
    # # print(lstm_model.summary())

    # model_builder.set_strategy(RNNMdelBuilder())
    # rnn_model = model_builder.compile_model(X_train)
    # # print(rnn_model.summary())

    # model_builder.set_strategy(GRUModelBuilder())
    # gru_model = model_builder.compile_model(X_train)
    # # print(gru_model.summary())

    # predict_model = {}
    # predict_model['lstm_model'] = lstm_model
    # predict_model['rnn_model'] = rnn_model
    # predict_model['gru_model'] = gru_model

    # model_history = {}
    # for key,model in predict_model.items():
    #     print(model.summary())
    #     model_history[f"{key}_history"] = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0,validation_data=(X_test,y_test))
    #     print(f"model history completed for {key}")
    # for key in model_history.keys():
    #     print(key)

    # history = lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0,validation_data=(X_test,y_test))
    pass
