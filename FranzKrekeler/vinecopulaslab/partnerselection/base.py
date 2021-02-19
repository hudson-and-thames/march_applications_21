# Author: Franz Krekeler 2021
from typing import List
import itertools
import numpy as np
import pandas as pd


class SelectionBase(object):
    """The base class for the partner selection framework.
    """
    def __init__(self):
        """Initialization
        """
        self.corr_returns_top_n = None

    @staticmethod
    def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate percentage based daily returns
        :param: prices (pd.DataFrame): The columns must the closing prices for the stocks
        :return: returns (pd.DataFrame)
        """
        return prices.pct_change(fill_method='ffill').dropna(how='all')

    @staticmethod
    def _ranked_correlation(returns: pd.DataFrame) -> pd.DataFrame:
        """Given a df of returns calculated it's Spearman correlation matrix
        :param: returns (pd.DataFrame): The input needs to be in percentage based returns
        :return: returns_correlateion (pd.DataFrame)
        """
        return returns.corr("spearman")

    @staticmethod
    def _rankings_pct(returns: pd.DataFrame):
        """Calculate the rank of a given dataframe and then convert it to percentage based
        :param: returns (pd.DataFrame)
        :return: returns_ranked_percentile  (pd.DataFrame)
        """
        return returns.rank(pct=True)

    @staticmethod
    def _top_n_correlations(corr_returns: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        """For correlation matrix return the top n correlations (default 50)
        :param: corr_returns (pd.DataFrame): correlation matrix
        :return: corr_returns_top_n pd.DataFrame shape is (n,n) 
        """
        # Filter self correlated and self correlated stocks
        corr_returns_unstacked = corr_returns[corr_returns < 1].unstack().sort_values(ascending=False)
        corr_returns_unstacked = corr_returns_unstacked.reset_index().dropna()
        corr_returns_unstacked.columns = "TARGET_STOCK", "STOCK_PAIR", "CORRELATION"
        # Note pandas was chosen here, but many ways lead to rome
        corr_returns_top_n = corr_returns_unstacked.groupby("TARGET_STOCK").head(top_n)
        return corr_returns_top_n.sort_values(['TARGET_STOCK', 'CORRELATION'])

    @staticmethod
    def _prepare_combinations_of_partners(stock_selection: List[str]) -> pd.DataFrame:
        """Helper function to calculate all combinations for a target stock and it's potential partners
        :param: stock_selection (pd.DataFrame): the target stock has to be the first element of the array
        :return: the possible combinations for the quadruples.Shape (19600,4) or
        if the target stock is left out (19600,3)
        """
        # We will convert the stock names into integers and then get a list of all combinations with a length of 3
        num_of_stocks = len(stock_selection)
        # We turn our partner stocks into numerical indices so we can use them directly for indexing
        partner_stocks_idx = np.arange(1, num_of_stocks)  # basically exclude the target stock
        partner_stocks_idx_combs = itertools.combinations(partner_stocks_idx, 3)
        return list(partner_stocks_idx_combs)
    
    def _find_partners(self, target_stocks: List[str] = []):
        """
        Helper functions where we apply the approach to each stock. Optional a subset of target stocks can be chosen.
        :param: return_target_stock (List[str]): the subset of target stocks to analyze (default [])
        :return: (pd.DataFrame)
        """
        assert self.corr_returns_top_n is not None
        corr_returns_top_n = self.corr_returns_top_n.copy()
        if len(target_stocks):
            sublist = corr_returns_top_n.TARGET_STOCK.isin(target_stocks)
            corr_returns_top_n = corr_returns_top_n[sublist]
        target_stocks_partners_quadruples = corr_returns_top_n.groupby('TARGET_STOCK').apply(
            self._partner_selection_approach)
        return target_stocks_partners_quadruples

