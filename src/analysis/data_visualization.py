from abc import ABC ,abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DataVisualizationStrategy(ABC):
    @abstractmethod
    def visualize(self, data):
        pass



class ClosingPriceVisualizerStratgey(DataVisualizationStrategy):
    def visualize(self, data):
        plt.figure(figsize=(12,6))
        plt.plot(data['Close'],label='Close price')
        plt.title("Closing price over time")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()




class VolumeVisualizerStrategy(DataVisualizationStrategy):
    def visualize(self, data):
        plt.figure(figsize=(12,6))
        plt.bar(data.index,data["Volume"],color='green')
        plt.title("Volume over time")
        plt.xlabel('date')
        plt.ylabel('volumne')
        plt.show()




class MovingAverageVisualizerStrategy(DataVisualizationStrategy):
    def visualize(self, data):
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        plt.figure(figsize=(15,8))
        plt.plot(data['Close'],label='Close price')
        plt.plot(data['MA50'],label='50 days moving average',linestyle='-')
        plt.plot(data['MA200'],label='200 days moving average',linestyle='-')
        plt.title("Closing price with 50 and 200 days moving average")
        plt.xlabel("Date")
        plt.ylabel('Price')
        plt.legend()
        plt.show()




class DailyReturnVisualzierStrategy(DataVisualizationStrategy):
    def visualize(self, data):
        data['Daily Return'] = data['Close'].pct_change()
        plt.figure(figsize=(12, 6))
        sns.histplot(data["Daily Return"].dropna(), bins=50, color="blue", kde=True)
        plt.title("Histogram of Daily Returns")
        plt.xlabel("Daily Return")
        plt.ylabel("Frequency")
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(data["Daily Return"], label="Daily Return")
        plt.title("Daily Return Over Time")
        plt.xlabel("Date")
        plt.ylabel("Daily Return")
        plt.legend()
        plt.show()




class CorrelationStrategy(DataVisualizationStrategy):
    def visualize(self, data):
        correlation = data[["Close", "Volume"]].corr()
        print("Correlation between Close Price and Volume:")
        print(correlation)
        sns.heatmap(correlation, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.show()



class DataVisualizer:
    def __init__(self,strategy:DataVisualizationStrategy):
        """
            initialize data visualizer with a specific strategy

            parameters:
                strategy -> strategy to be used for data visualizer

            Returns:
                None
        """
        self.strategy=strategy

    def set_strategy(self,strategy:DataVisualizationStrategy):
        """
            setting a new strategy for data visualizer

            parameters:
                strategy -> strategy to be used for data visualizer

            Returns:
                None
        """
        self.strategy=strategy

    def execute_visualizer(self,data):
        """
            executing the visualization using the current strategy

            parameters:
                data -> pd.dataframe [dataframe]
            
            Returns:
                None -> executes the strategy's inspection method
        """
        self.strategy.visualize(data)



if __name__=="__main__":
    data = pd.read_csv('D:\coding\ml\stock-price-predictor\src\components\data\AAPL_stock_data.csv')
    data_visualizer = DataVisualizer(strategy=ClosingPriceVisualizerStratgey())
    data_visualizer.execute_visualizer(data=data)
    data_visualizer.set_strategy(VolumeVisualizerStrategy())
    data_visualizer.execute_visualizer(data=data)
    data_visualizer.set_strategy(MovingAverageVisualizerStrategy())
    data_visualizer.execute_visualizer(data=data)
    data_visualizer.set_strategy(DailyReturnVisualzierStrategy())
    data_visualizer.execute_visualizer(data=data)
    data_visualizer.set_strategy(CorrelationStrategy())
    data_visualizer.execute_visualizer(data=data)