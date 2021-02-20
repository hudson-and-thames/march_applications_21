import numpy as np
import yfinance as yf
import yahoo_fin.stock_info as ys
"""
This class contains preliminary steps before data analysis.
1. Download historical price data of S&P500 constituents.
2. NaN elements removal(reference: Hansen Pei). 
3. Return calculation(reference: Hansen Pei).
@author: Jamie Keng
"""

class Preliminary:

    @staticmethod
    def download_sp500(start_date, end_date, interval='1d', num=100):
        """
        Parameters
        ----------
        start_date -- Download start date string (YYYY-MM-DD) or _datetime.
        end_date --Download end date string (YYYY-MM-DD) or _datetime.

        interval -- Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        num -- Number of randomly picked stocks out of S&P500 index.
        """
        # Get the S&P 500 stocks tickers.
        tickers_sp500 = ys.tickers_sp500()
        tickers_sp500 = list(np.random.choice(tickers_sp500, num))

        # Download the historical S&P 500 price data
        tickers = tickers_sp500
        start_date = start_date
        end_date = end_date
        interval = interval
        hist_data = yf.download(tickers,
                                start=start_date,
                                end=end_date,
                                interval=interval,
                                group_by='column')['Close']
        return hist_data

    @staticmethod
    def remove_nuns(df, threshold=100):
        """Remove tickers with nuns in value over a threshold.

        Parameters
        ----------
        df : Price time series dataframe
        threshold: The number of null values allowed
        """
        null_sum_each_ticker = df.isnull().sum()
        tickers_under_threshold = \
            null_sum_each_ticker[null_sum_each_ticker <= threshold].index
        df = df[tickers_under_threshold]

        return df

    @staticmethod
    def get_returns_data(hist_data):
        """Calculate the return of historical price data.

        Parameters
        ----------
        hist_data -- The price data
        """
        returns_data = hist_data.pct_change()
        returns_data = returns_data.iloc[1:]

        return returns_data

    @staticmethod
    def stock_index(df):
        """Stock Index Correspondence

        Parameters
        ----------
        df : historical stock price dataset/return dataset
        """
        indexes = list(enumerate(df.columns))

        stock_ind_dict = {}
        for asset_index, ticker in indexes:
            stock_ind_dict[asset_index] = ticker

        return stock_ind_dict
