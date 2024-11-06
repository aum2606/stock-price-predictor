from sklearn.preprocessing import MinMaxScaler
from src.components.model_prediction import predict_for_future
from src.pipeline.analysis_pipeline import analysis_pipeline
from src.pipeline.training_pipeline import training_pipeline
import pandas as pd

def run(data):
    analysis_pipeline(data)
    model,scaled_data = training_pipeline(data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    print(scaled_data.shape)
    predict_for_future(model,scaled_data,10,60,scaler)

if __name__=="__main__":
    data = pd.read_csv('D:\coding\ml\stock-price-predictor\src\components\data\AAPL_stock_data.csv')
    run(data=data)