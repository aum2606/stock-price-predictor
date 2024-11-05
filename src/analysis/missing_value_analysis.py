from abc import ABC,abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class MissingValueAnalysisTemplate(ABC):
    def analyze(self,df:pd.DataFrame):
        """
            Perform a complete missing value analysis by identifying and visualizing missing values in the dataset

            Parameters:
                df -> pd.dataframe [dataframe]

            Returns:
                None: method performs analysis and visualizes missing values
        """
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self,df:pd.DataFrame):
        """
            this function identifies missing value in the dataset

            Parameters:
                df -> pd.dataframe [dataframe]
            
            Returns:
                None -> prints the count of missing values for each column
        """
        pass

    @abstractmethod
    def visualize_missing_values(self,df:pd.DataFrame):
        """
            this function visualizes missing values in the dataset

            Parameters:
                df -> pd.dataframe [dataframe]

            Returns:
                None -> create a visual chart of missing values
        """

        pass


class SimpleMissingValueAnalysis(MissingValueAnalysisTemplate):
    def identify_missing_values(self,df:pd.DataFrame):
        """
            print the count of missing values for each column in the dataset

            Parameters:
                df -> pd.dataframe [dataframe]

            Returns:
                None -> prints missing value counts to the console
        """
        print('Missing value analysis: ')
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])
    
    def visualize_missing_values(self, df: pd.DataFrame):
        """
            visualizing missing value in the dataset

            Parameters:
                df -> pd.dataframe [dataframe]

            Returns:
                None -> create a visual chart of missing value in dataset
        """
        print('visualizing the missing values...')
        plt.figure(figsize=(15,7))
        sns.heatmap(df.isnull(),annot=True,cmap='coolwarm')
        plt.title('Heatmap of missing values')
        plt.show()


if __name__=="__main__":
    data = pd.read_csv('D:\coding\ml\stock-price-predictor\src\components\data\AAPL_stock_data.csv')
    missing_value_analysis = SimpleMissingValueAnalysis()
    missing_value_analysis.identify_missing_values(data)
    missing_value_analysis.visualize_missing_values(data)
    print(data.head())