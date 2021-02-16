import yfinance as yf
import yahoo_fin.stock_info as ys
import pandas as pd
import numpy as np

"""
This class contains preliminary steps before data analysis.

1. Download historical price data of S&P500 constituents.
2. NaN elements removal(reference: Hansen Pei). 
3. Return calculation(reference: Hansen Pei).


@author: Jamie Keng
"""

class preliminary:
    
    def __init__(self):
        pass
    
    
    
    def download_sp500(self, start_date, end_date, interval= '1d', num =100):
        """

        Parameters
        ----------
        start_date : str
            Download start date string (YYYY-MM-DD) or _datetime.

        end_date : str
            Download end date string (YYYY-MM-DD) or _datetime.

        interval : str, OPTIONAL
            Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            default value is '1d'

        num : int 
            Number of randomly picked stocks out of S&P500 index.
            default value is 100.

        Returns
        -------
        hist_data : pandas dataframe
            The downloaded historical price dataframe. 

        """    

        # Get the S&P 500 stocks tickers.
        tickers_sp500 = ys.tickers_sp500()
        tickers_sp500 =list(np.random.choice(tickers_sp500, num))

        # Download the historical S&P 500 price data
        tickers = tickers_sp500
        start_date = start_date
        end_date =  end_date
        interval = interval  

        hist_data = yf.download(tickers,
                                start= start_date, 
                                end= end_date,
                                interval= interval,
                                group_by='column')['Close']

        return hist_data
    
    
 
    def remove_nuns(self, df, threshold=100):

        """
        Remove tickers with nuns in value over a threshold.

        Parameters
        ----------
        df : pandas dataframe
            Price time series dataframe

        threshold: int, OPTIONAL
            The number of null values allowed
            Default is 100

        Returns
        -------
        df : pandas dataframe
            Updated price time series without any nuns
        """
        null_sum_each_ticker = df.isnull().sum()
        tickers_under_threshold = \
            null_sum_each_ticker[null_sum_each_ticker <= threshold].index
        df = df[tickers_under_threshold]

        return df



    def get_returns_data(self, hist_data):
        
        """        
        Calculate the return of historical price data. 

        Parameters
        ----------
        hist_data : pandas dataframe
            The price data

        Returns
        -------
        returns_data : pandas dataframe
            The requested returns data.
        """

        returns_data = hist_data.pct_change()
        returns_data = returns_data.iloc[1:]

        return returns_data
    
    def stock_index(self, df): 
        """
        Stock Index Correspondance

        Parameters
        ----------
        df : pandas dataframe
            historical stock price dataset/return dataset 

        Returns
        -------
        stock_ind_dict : dictionary 
            keys : stock indexes in the dataframe 
            values : stock tickers 
        """
        
        indexes = list(enumerate(df.columns))

        stock_ind_dict = {}
        for asset_index, ticker in indexes:
            stock_ind_dict[asset_index] = ticker


        return stock_ind_dict





