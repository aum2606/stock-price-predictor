from abc import ABC,abstractmethod
import pandas as pd
from src.logger import logging
from sklearn.preprocessing import MinMaxScaler


class PreprocessingStrategy(ABC):
    @abstractmethod
    def apply_data_transformation(self,df:pd.DataFrame)->pd.DataFrame:
        """
            Abstract method to apply data transformation(feature engineering) to the dataframe

            Parameters:
                df -> pd.dataframe => the dataframe containing data for transformation

            Return:
                pd.dataframe -> transformed data
        """
        pass





class MinMaxScaling(PreprocessingStrategy):
    def __init__(self,feature_range=(0,1)):
        """
            Initialize the StandardScaling with specific feature to transform data

            Parameters:
                features -> list of features to be transformed
                feature_range -> tuple of (min,max) to scale the data to
        """
        self.scaler = MinMaxScaler(feature_range=feature_range) 


    def apply_data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Applies standard scaler transformation to the specified features in the dataframe

            Parameters:
                df -> pd.dataframe => the dataframe containing data for transformation

            Returns:
                pd.dataframe -> scaled features dataframe
        """
        logging.info(f"appyling MinMax scaler to the dataframe")
        data_transformed = data[["Close"]]
        data_transformed = self.scaler.fit_transform(data_transformed)
        logging.info("MinMax scaling complete")
        return data_transformed



class PreProcessor:
    def __init__(self,preprocessing_strategy: PreprocessingStrategy):
        """
            initializing the preprocessing strategy 

            Parameters:
                preprocessing_strategy -> PreprocessingStrategy instance
        """
        self.preprocessing_strategy = preprocessing_strategy

    
    def set_preprocessing_strategy(self,preprocessing_strategy:PreprocessingStrategy):
        """
            setting the preprocessing strategy for data preprocessing

            Parameters:
                preprocessing_strategy -> PreprocessingStrategy instance
        """
        logging.info("switching preprocessing strategy")
        self.preprocessing_strategy = preprocessing_strategy

    def apply_data_preprocessing(self,df:pd.DataFrame) ->pd.DataFrame:
        """
            applying data preprocessing using selected strategy

            Parameters:
                df -> pd.dataframe => the dataframe containing data for transformation
            
            Returns:
                pd.dataframe -> preprocessed dataframe
        """
        logging.info("applying preprocessing strategy")
        return self.preprocessing_strategy.apply_data_transformation(df=df)
    

if __name__=="__main__":
    data = pd.read_csv('D:\coding\ml\stock-price-predictor\src\components\data\AAPL_stock_data.csv')
    data_preprocessor = PreProcessor(MinMaxScaling(feature_range=(0,1)))
    data_transform_scaled = data_preprocessor.apply_data_preprocessing(df=data)
    print(data_transform_scaled.shape)