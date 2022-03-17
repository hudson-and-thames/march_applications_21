"""
Module for import data of S&P 500 stocks using yfinance and Wikipedia

@author: Joohwan Ko
"""
import pandas as pd
import yfinance as yf


class ImportData:
    """
    Class that imports data from yfinance and Wikipedia for the purpose of research
    """

    def __init__(self):
        pass

    @staticmethod
    def get_list_sp500():
        """
        Returns a list of constituents in S&P 500 Index from Wikipedia
        :return: (list) constituents in S&P 500 Index
        """
        # Read HTML from Wikipedia
        wiki_data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        # Only symbols(tickers) of the stocks are needed
        ticker_list = wiki_data[0]['Symbol'].values.tolist()
        return ticker_list

    @staticmethod
    def download_data(ticker_list, start_date, end_date):
        """
        Returns a dataframe of downloaded data of the given ticker list, start date and end date
        :param ticker_list: (list) a list of tickers
        :param start_date: (str) start date of data
        :param end_date: (str) end date of data
        :return: (pd.DataFrame) dataframe of stock prices
        """
        # Transform the ticker list into space separated string
        ticker_string = ' '.join(ticker_list)
        # Use yfinance to download data
        price_data = yf.download(
            tickers=ticker_string,
            start=start_date,
            end=end_date,
            group_by='ticker',
            auto_adjust=True,
            prepost=True,
            threads=True,
            proxy=None
        )
        return price_data

    @staticmethod
    def get_returns(price_data):
        """
        Returns a dataframe of stock returns for the given stock prices
        :param price_data: (pd.DataFrame) dataframe of stock prices
        :return: (pd.DataFrame) dataframe of stock returns
        """
        # Get 'Close' columns of the price data and change it to percent change which is equal to stock returns
        return_data = price_data.iloc[:, price_data.columns.get_level_values(1) == 'Close'].pct_change()
        # Drop Null Values
        return_data = return_data.dropna(axis=1, how='all')
        # As the dataframe has multi-level columns, drop it to single-level
        return_data = return_data.droplevel(1, axis=1)
        return return_data


class DataPreprocess():
    """
    Class that helps data preprocessing
    """

    def __init__(self):
        pass

    @staticmethod
    def get_rank(return_df):
        """
        Get rank of the dataframe for each stock
        :param return_df: (pd.DataFrame) stock returns
        :return: (pd.DataFrame) rank of each stock
        """
        rank_df = return_df.rank()
        return rank_df

    @staticmethod
    def most_correlated_stocks(rank_df, num_top_stocks):
        """
        Get top n correlated stocks for a given rank dataframe
        :param rank_df: (pd.DataFrame) rank of each stock
        :param num_top_stocks: (pd.DataFrame) number of top stocks to get
        :return: (pd.DataFrame, pd.DataFrame) rank of top correlated stocks and correlation matrix of those
        """
        # Get correlation matrix of the rank dataframe
        corr_df = rank_df.corr()
        # To see how highly correlated two stocks are, we use absolute value of correlations
        corr_abs_df = corr_df.abs()
        unstack_corr = corr_abs_df.unstack()
        sorted_corr = unstack_corr.sort_values(kind="quicksort").dropna()
        # Find top n number of correlated stocks
        num_stocks = len(corr_df)
        sorted_corr = sorted_corr[:-num_stocks]
        sorted_corr = sorted_corr.iloc[::-1]
        # Iterate these for loops until we find n most highly correlated stocks
        top_list = []
        for ticker1, ticker2 in sorted_corr.index:
            if ticker1 not in top_list:
                top_list.append(ticker1)
                if len(top_list) == num_top_stocks:
                    break
            elif ticker2 not in top_list:
                top_list.append(ticker2)
                if len(top_list) == num_top_stocks:
                    break
        # Get top stocks' correlation matrix
        top_rank_df = rank_df.loc[:, top_list]
        top_rank_corr_df = top_rank_df.corr()
        return top_rank_df, top_rank_corr_df
