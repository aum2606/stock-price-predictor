import pandas as pd

from src.analysis.basic_data_inspection import DataInspector, DataTypesInspectionStrategy, SummaryStatisticsInspectionStrategy
from src.analysis.data_visualization import ClosingPriceVisualizerStratgey, CorrelationStrategy, DailyReturnVisualzierStrategy, DataVisualizer, MovingAverageVisualizerStrategy, VolumeVisualizerStrategy
from src.analysis.missing_value_analysis import SimpleMissingValueAnalysis


def analysis_pipeline(data):
    # Performing baisc data inspection
    print(data.head())

    print("=====================================================================================")
    print("Performing basic Data Inspection")
    data_inspector = DataInspector(DataTypesInspectionStrategy())
    data_inspector.executing_inspection(df=data)
    data_inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    data_inspector.executing_inspection(df=data)

    print("------------------------------------------------------------------------------------")
    print("Performing missing value analysis")
    #performing inspection to find missing values in the dataset
    missing_value_analysis = SimpleMissingValueAnalysis()
    missing_value_analysis.identify_missing_values(data)
    missing_value_analysis.visualize_missing_values(data)


    print("------------------------------------------------------------------------------------")
    print("Performing visualization on various aspects of the dataset")
    print("___________________________________________________________")
    #various data visualization strategies to visualize different aspect of the dataset
    print("------------------Closing price visualization---------------------")
    data_visualizer = DataVisualizer(strategy=ClosingPriceVisualizerStratgey())
    data_visualizer.execute_visualizer(data=data)

    print("------------------Volumne visualization---------------------")
    data_visualizer.set_strategy(VolumeVisualizerStrategy())
    data_visualizer.execute_visualizer(data=data)

    print("------------------Moving average visualization---------------------")
    data_visualizer.set_strategy(MovingAverageVisualizerStrategy())
    data_visualizer.execute_visualizer(data=data)

    print("------------------Daily returns visualization---------------------")
    data_visualizer.set_strategy(DailyReturnVisualzierStrategy())
    data_visualizer.execute_visualizer(data=data)

    print("------------------correlation between closing price and volume visualization---------------------")
    data_visualizer.set_strategy(CorrelationStrategy())
    data_visualizer.execute_visualizer(data=data)




if __name__=="__main__":
    data = pd.read_csv('D:\coding\ml\stock-price-predictor\src\components\data\AAPL_stock_data.csv')
    analysis_pipeline(data=data)