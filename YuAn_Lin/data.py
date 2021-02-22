#For crawling S&P500 Tickers
import os
import requests
from io import StringIO

#For getting and modifying data
import numpy as np
import pandas as pd
import datetime
import yfinance as yf


def get_sp500_tickers():
    """Crawling S&P500 Tickers

    Returns
    -------
    List contains S&P500 tickers
    """

    # Send a request to the link
    headers = { 'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36'}
    link = "https://www.slickcharts.com/sp500"
    res = requests.get(link, headers = headers)

    # Clean the text, and save as a dataframe 
    lines = res.text.replace("\r", "").split("\n")
    df = pd.read_html( StringIO("\n".join( lines[:] ) ), header = None )[0]
    tickers = df["Symbol"]
    tickers = tickers.apply(lambda s: s.replace(".", "-"))  # Modify tickers, so they can fit Yahoo Finance's ticker format

    return list(tickers.values)

def get_adj_close(tickers, start_y, start_m, start_d, end_y, end_m, end_d):
    """Get adjusted close data from Yahoo Finance

    Parameters
    ----------
    tickers : str, list of str
        List of tickers to download
    start_y, start_m, start_d : int
        start year, start month, start day
    end_y, end_m, end_d : int
        end year, end month, end day

    Returns
    -------
    Pandas DataFrame contains adjusted close
    """

    # Modify date format
    start = datetime.datetime(start_y, start_m, start_d).strftime("%Y-%m-%d")
    end = datetime.datetime(end_y, end_m, end_d).strftime("%Y-%m-%d")

    # Get data using yfinance module
    data = yf.download(tickers, start, end)
    data = data['Adj Close']
    data = data.round(decimals=2)

    return data