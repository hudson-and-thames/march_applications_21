# Author: Franz Krekeler 2021
from typing import List
import numpy as np
import pandas as pd
import itertools


class SelectionBase:
    """The base class for the partner selection framework.
    """

    def __init__(self):
        self.corr_returns_top_n = None

    @staticmethod
    def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate percentage based daily returns
        :param prices (pd.DataFrame): The columns must the closing prices of 
        """
        return prices.pct_change(fill_method='ffill').dropna(how='all')

    @staticmethod
    def _ranked_correlation(returns: pd.DataFrame) -> pd.DataFrame:
        """Given a df of returns it's Spearman correlation matrix
        :param: df_returns (pd.DataFrame): The input needs to be in percentage based returns
        :return: pd.DataFrame
        """
        return returns.corr("spearman")

    @staticmethod
    def _rankings_pct(returns: pd.DataFrame):
        """Calculate the rank of a given dataframe and then convert it to percentage based
        :param: df (pd.DataFrame)
        :return: pd.DataFrame
        """
        return returns.rank(pct=True)

    @staticmethod
    def _top_n_correlations(corr_returns: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        """For correlation matrix return the top 50 correlations
        :param: df_corr (pd.DataFrame): correlation matrix
        :return: pd.DataFrame
        """
        # Filter self correlated and self correlated stocks
        corr_returns_unstacked = corr_returns[corr_returns < 1].unstack().sort_values(ascending=False)
        corr_returns_unstacked = corr_returns_unstacked.reset_index().dropna()
        corr_returns_unstacked.columns = "TARGET_STOCK", "STOCK_PAIR", "CORRELATION"
        # Note pandas was chosen here, but many ways lead to rome
        corr_returns_top_n = corr_returns_unstacked.groupby("TARGET_STOCK").head(top_n)
        return corr_returns_top_n.sort_values(['TARGET_STOCK', 'CORRELATION'])


    def _find_partners_for_target_stock(self, group):
        """
        Helper function for df.groupby("TARTGET_STOCK").apply(...)
        :param group: (group) The group of 50 most correlated stocks
        :return: (List[str]) returns a list of highest correlated quadruple
        """
        pass

    def _get_quadruples(self, df_corr_top50: pd.DataFrame) -> pd.DataFrame:
        """
        :param df_corr_top50: ([pd.DataFrame])the top50 correlated stocks for a list of given target stocks
        :return: (pd.Series) returns list of partner quadruples for a target stock
        """
        qf = df_corr_top50.groupby('TARGET_STOCK').apply(self._find_partners_for_target_stock)
        return qf

    def _prepare_combinations_of_partners(self, stock_selection: List[str], return_target_stock=True) -> pd.DataFrame:
        # We will convert the stock names into integers and then get a list of all combinations with a length of 3
        num_of_stocks = len(stock_selection)
        # We turn our partner stocks into numerical indices so we can use them directly for indexing
        partner_stocks_idx = np.arange(1, num_of_stocks) # basically exclude the target stock
        partner_stocks_idx_combs = itertools.combinations(partner_stocks_idx, 3)
        if return_target_stock:
            return np.array(list((0,) + comb for comb in partner_stocks_idx_combs))
        return list(partner_stocks_idx_combs)

    def _find_partners(self, target_stocks: List[str] = []):
        """
        main function to return quadruples of correlated stock (spearman) method
        :return: (pd.DataFrame)
        """
        assert type(self.corr_returns_top_n) != type(None)
        corr_returns_top_n = self.corr_returns_top_n.copy()
        if len(target_stocks):
            sublist = corr_returns_top_n.TARGET_STOCK.isin(target_stocks)
            corr_returns_top_n = corr_returns_top_n[sublist]
        target_stocks_partners_quadruples = corr_returns_top_n.groupby('TARGET_STOCK').apply(self._partner_selection_approach)
        return target_stocks_partners_quadruples

    def find_partners(self, close: pd.DataFrame, target_stocks: List[str] = []):
        self._preprocess(close)
        return self._find_partners(target_stocks)




