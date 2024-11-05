from abc import ABC , abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.logger import logging



class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self,data):
        pass


def create_sequences(data,time_step=60):
    X, y = [], []
    for i in range(time_step,len(data)):
        X.append(data[i-time_step:i,0])
        y.append(data[i,0])
    return np.array(X), np.array(y)
      

class CustomDataSplittingStrategy(DataSplittingStrategy):
    def split_data(self, sclaed_data):
        train_size = int(len(sclaed_data)*0.8)
        train_data,test_data = sclaed_data[:train_size],sclaed_data[train_size:]
        time_step = 60
        X_train,y_train = create_sequences(train_data,time_step)
        X_test,y_test = create_sequences(test_data,time_step)
        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
        X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
        return X_train,y_train,X_test,y_test



class DataSplitter:
    def __init__(self,strategy:DataSplittingStrategy):
        """
            initialize strategy for data splitting

            Parameters:
                strategy (DataSplittingStrategy): the strategy to be used for data splitting
        """
        self._strategy = strategy

    def set_strategy(self,strategy:DataSplittingStrategy):
        """
            this method sets the strategy for data splitting

            Parameters:
                strategy (DataSplittingStrategy): the strategy to be used for data splitting
        """
        self._strategy = strategy

    def split(self,df:pd.DataFrame):
        """
            It executes the data splitting with current strategy

            Parameters:
                df (pd.DataFrame): the data to be split
                target_column (str): the column that contains the target variable

            Returns:
                X_train, X_test, y_train, y_test: The training and testing splits for features and target
        """
        logging.info(f"Splitting the data using selected strategy {self._strategy}")
        return self._strategy.split_data(df)



if __name__=="__main__":
    data = pd.read_csv('D:\coding\ml\stock-price-predictor\src\components\data\AAPL_stock_data.csv')
    data = data[["Close"]]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    splitter = DataSplitter(strategy=CustomDataSplittingStrategy())
    X_train,X_test,y_train,y_test = splitter.split(scaled_data)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)