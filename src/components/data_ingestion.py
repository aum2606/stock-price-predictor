from abc import ABC,abstractmethod
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

class DataIngestor(ABC):
    @abstractmethod
    def ingest_data(self)->pd.DataFrame:
        pass



class yahooDataIngestor(DataIngestor):


    def ingest_data(self) -> pd.DataFrame:
        ticker =  'AAPL'
        start_date = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')  # Last 5 years
        end_date = datetime.now().strftime('%Y-%m-%d')
        filename = f"data/{ticker}_stock_data.csv"
        data = yf.download(ticker, start=start_date, end=end_date)

        # Save the data as a CSV file
        data.to_csv(filename)

        print(f"Dataset saved as {filename}")
        return data
    


class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor()->DataIngestor:
        """returns the appropriate data ingestor based on the file extension"""
        try:
            return yahooDataIngestor()
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    data_ingestor = DataIngestorFactory.get_data_ingestor()
    data = data_ingestor.ingest_data()
    print(data.head())