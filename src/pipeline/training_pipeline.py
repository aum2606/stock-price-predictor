import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.components.model_prediction import evaluation_metrics, predict_and_plot
import yfinance as yf
import numpy as np
from src.components.data_preprocessor import MinMaxScaling, PreProcessor
from src.components.data_splitter import CustomDataSplittingStrategy, DataSplitter, create_sequences
from src.components.model_building import GRUModelBuilder, LSTMModelBuilder, ModelBuilder, RNNMdelBuilder


data = pd.read_csv('D:\coding\ml\stock-price-predictor\src\components\data\AAPL_stock_data.csv')

def training_pipeline(data):
    print(data.head())

    #preprocessing the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    print(scaled_data.shape)

    # Splitting the data
    splitter = DataSplitter(strategy=CustomDataSplittingStrategy())
    X_train,y_train,X_test,y_test = splitter.split(scaled_data)

    #making a dictionary to store models
    # predict_models = {}

    #building models
    model_builder = ModelBuilder(LSTMModelBuilder())
    lstm_model = model_builder.compile_model(X_train=X_train)
    # predict_models['lstm_model'] = lstm_model

    # model_builder.set_strategy(RNNMdelBuilder())
    # rnn_model = model_builder.compile_model(X_train=X_train)
    # predict_models['rnn_model'] = rnn_model

    # model_builder.set_strategy(GRUModelBuilder())
    # gru_model = model_builder.compile_model(X_train=X_train)
    # predict_models['gru_model'] = gru_model

    print(lstm_model.summary())
    print(f"begining model training for lstm_model")
    print("-----------------------------------------------")
    history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    print(history)
    print(f"model history completed for lstm")
    mse,mae,r2 = evaluation_metrics(X_test,y_test,scaler,lstm_model)
    print(f"mse: {mse}")
    print(f"mae: {mae}")
    print(f"r2: {r2}")
    print("========================================================================")
    predict_and_plot(lstm_model,X_test,y_test,scaler,title=f"lstm prediction")
    #building dictonaries to store history of trained model
    # model_history = {}
    # max_r2 = {}
    #iterating over the models from model_history and fitting and performing evaluation of the data
    # for key,model in predict_models.items():
    #     print()
    #     print(f"begining model training for {key}")
    #     print("-----------------------------------------------")
    #     print(model.summary())
    #     model_history[f"{key}_history"] = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0,validation_data=(X_test,y_test))
    #     print(f"model history completed for {key}")
    #     mse,mae,r2 = evaluation_metrics(X_test,y_test,scaler,model)
    #     print(f"{key} => {mse}")
    #     print(f"{key} => {mae}")
    #     print(f"{key} => {r2}")
    #     print("========================================================================")
    #     predict_and_plot(model,X_test,y_test,scaler,title=f"{key} prediction")

    #     max_r2[key] = r2
    # key_max = max(max_r2,key=lambda x: max_r2[x])
    # print(f"the model with max r2 is {key_max}")

    # model = predict_models[key_max]
    return lstm_model,scaled_data


if __name__=="__main__":
    ticker = "AAPL"  # Replace with desired ticker symbol
    data = pd.read_csv("D:/coding/ml/stock-price-predictor/src/components/data/AAPL_stock_data.csv")
    model = training_pipeline(data=data)
    # model.save('D:/coding/ml/stock-price-predictor/src/model/stock_price_prediction_model.h5')
    print(f"the final model is {model.summary()}")