"""
Fetch historical stock data from Yahoo Finance
"""
from __future__ import print_function
from typing import List
import yfinance as yf
import pickle
import requests
import pandas as pd

import os


class UniverseDownloader:
    """
    The class UniverseDownloader fetches a part of the SP500 symbols and
    returns the historic closing prices.
    """
    def __init__(self, cache=False, cachepath="./tmp"):
        """
        :param cache: (bool) Cache downloaded data as pickle (False by default)
        :param cache_path: (str) Path where to cache data (./tmp by default)
        """
        self.cache = cache
        self.cachepath = cachepath
    
    @staticmethod
    def fetch_sp500symbols() -> List[str]:
        """
        Fetches constituents symbols of the SP500 from GitHub repository and returns it as a list.
        :return: (List[str]) returns a list of SP500 symbols
        """
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents_symbols.txt"
        r = requests.get(url)
        symbols = r.text.split("\n")[:-1]
        return symbols

    def _fetch_historic_sp500_data(self, start="2015-01-01", end="2020-01-01") -> pd.DataFrame:
        """
        :param start: (str) start date ('2015-01-01' by default)
        :param end: (str) end date ('2020-01-01' default)
        :return: (DataFrame) returns dataframe of SP500 historic data (including open,close)
        """
        sp500symbols = self.fetch_sp500symbols()
        sp500prices = yf.download(" ".join(sp500symbols), start=start, end=end)
        return sp500prices

    def historic_sp500_prices(self, start="2015-01-01", end="2020-01-01") -> pd.DataFrame:
        """
        :param start: (str) start date ('2015-01-01' by default)
        :param end: (str) end date ('2020-01-01' default)
        :return: (List[str]) returns a list of SP500 symbols
        """
        if self.cache:
            cache_pickle_path = f"{self.cachepath}/sp500_{start}_{end}.p"
            try:
                return pd.read_pickle(cache_pickle_path)
            except FileNotFoundError:
                print("File could not be loaded from cache")
        sp500_prices = self._fetch_historic_sp500_data(
            start=start, end=end)
        sp500_closeprices = sp500_prices['Close'].dropna(how='all')
        if self.cache:
            if not os.path.exists(self.cachepath):
                os.makedirs(self.cachepath)
            sp500_closeprices.to_pickle(cache_pickle_path)
        return sp500_closeprices
