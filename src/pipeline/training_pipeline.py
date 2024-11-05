import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

from src.components.data_preprocessor import MinMaxScaling, PreProcessor
from src.components.data_splitter import CustomDataSplittingStrategy, DataSplitter, create_sequences
from src.components.model_building import GRUModelBuilder, LSTMModelBuilder, ModelBuilder, RNNMdelBuilder


data = pd.read_csv('D:\coding\ml\stock-price-predictor\src\components\data\AAPL_stock_data.csv')

def training_pipeline(data):
    print(data.head())
    #preprocessing the data
    data_preprocessor = PreProcessor(MinMaxScaling(feature_range=(0,1)))
    scaled_data = data_preprocessor.apply_data_preprocessing(data=data)
    print(scaled_data.shape)

    #Splitting the data
    splitter = DataSplitter(strategy=CustomDataSplittingStrategy())
    X_train,y_train,X_test,y_test = splitter.split(scaled_data)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    predict_models = {}

    model_builder = ModelBuilder(LSTMModelBuilder())
    lstm_model = model_builder.compile_model(X_train=X_train)
    predict_models['lstm_model'] = lstm_model

    model_builder.set_strategy(RNNMdelBuilder())
    rnn_model = model_builder.compile_model(X_train=X_train)
    predict_models['rnn_model'] = rnn_model

    model_builder.set_strategy(GRUModelBuilder())
    gru_model = model_builder.compile_model(X_train=X_train)
    predict_models['gru_model'] = gru_model


    model_history = {}

    for key,model in predict_models.items():
        print(model.summary())
        model_history[f"{key}_history"] = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0,validation_data=(X_test,y_test))
        print(f"model history completed for {key}")
    for key in model_history.keys():
        print(key)

    return model_history


if __name__=="__main__":
    ticker = "AAPL"  # Replace with desired ticker symbol
    data = pd.read_csv("D:/coding/ml/stock-price-predictor/src/components/data/AAPL_stock_data.csv")
    training_pipeline(data=data)