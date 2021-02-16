import pandas as pd
from pandas_datareader import data as wb

from bs4 import BeautifulSoup

import requests

from datetime import datetime

req_headers = {
                "Connection": "keep-alive",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0",
                "Accept-Language": "en"
              }


def get_index_ticker(index: str):
    """
    Function to scrape existing index members from CNN website.

    :param index: Name of index to scrape.
    :type index: str
    :return: Tickers contained within specified input index
    :rtype: list
    """

    exchange_dict = {
                    'r2000': 'https://money.cnn.com/data/markets/russell/?page=1',
                    'spx': 'https://money.cnn.com/quote/quote.html?symb=SPX&page=1',
                    'ndx': 'https://money.cnn.com/quote/quote.html?symb=NDX&page=1',
                    'dji': 'https://money.cnn.com/data/dow30/?page=1',
                    'djt': 'https://money.cnn.com/data/markets/dowtrans/?page=1',
                    'dju': 'https://money.cnn.com/quote/quote.html?symb=DJU&page=1',
                    'nya': 'https://money.cnn.com/data/markets/nyse/?page=1'
                    }
    base_url = exchange_dict[index]
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, features='html.parser')
    try:
        # Retriving the number of pages that we have to scrape using the pagination provided by the website.
        pages = soup.findAll('div', {'class': 'paging'})[0].text.split()[-1]
    except:
        # No pagination, likely that there is an error/base_url has changed.
        pages = 0

    tickers = []

    for page in range(int(pages)+1):
        url = base_url[:-1] + str(page)  # Slicing base_url to change page number
        response = requests.get(url)
        soup = BeautifulSoup(response.text, features='html.parser')
        list_of_tickers = soup.findAll('a', {'class': 'wsod_symbol'})
        for t in list_of_tickers:
            tickers.append(t.text)
    return tickers


def build_price_df(tickers: list, index_name: str, data_start_date: str):
    """
    Build price dataframe based on given tickers and interval.

    :param tickers: A list of tickers you would like to retrieve the price for.
    :type tickers: list
    :param index_name: Index name of where tickers were retrieved from.
    :type index_name: str
    :param data_start_date: Start date of the price dataframe.
    :type data_start_date: str
    """

    df = pd.DataFrame()
    for i in tickers:
        try:
            df[i] = wb.DataReader(i, data_source='yahoo', start=data_start_date, end=datetime.today())['Adj Close']
            print('Successfully retrieved the price data for:', i)
        except:
            print('Failed to retrieve price data for: ', i)
            continue
    df.to_csv('{}_price.csv'.format(index_name), index=True)

if __name__ == '__main__':
    index_name = input('Please input which index (SPX/DJI/NDX) you would like to retrieve the price data for: ').lower()
    data_start_date = input('Please input the start date of the price data. (YYYY-MM-DD format): ')
    tickers = get_index_ticker(index_name)
    build_price_df(tickers, index_name, data_start_date)
