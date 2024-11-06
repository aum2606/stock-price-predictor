import pandas as pd
from src.pipeline.training_pipeline import training_pipeline


def evaluation_pipeline(data):
    model_history = training_pipeline(data=data)
    for key,model in model_history.items():
        print(f"Model: {key}")
        print(f"Accuracy: {model.history.history['accuracy'][-1]}")
        print(f"Loss: {model.history.history['loss'][-1]}")


if __name__ == "__main__":
    data = pd.read_csv("D:/coding/ml/stock-price-predictor/src/components/data/AAPL_stock_data.csv")
    evaluation_pipeline(data=data) 