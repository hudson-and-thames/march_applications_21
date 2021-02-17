import itertools
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.special
from typing import List
from statsmodels.distributions.empirical_distribution import ECDF
from .base import SelectionBase


class ExtendedSelection(SelectionBase):
    def _transform_df(self, df_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the ECDF for each stock
        """
        ecdf_df = df_returns.apply(lambda x: ECDF(x)(x), axis=0)
        return ecdf_df

    def _get_partner_for_target_stock(self, group) -> List[str]:
        """
        Helper function for df.groupby("TARTGET_STOCK").apply(...)
         :param group: (group) The group of 50 most correlated stocks
        :return: (List[str]) returns a list of highest correlated quadruple
        """
        target_stock = group.name
        partner_stocks = group.STOCK_PAIR.tolist()
        all_stocks = [target_stock] + partner_stocks
        # We create a subset of our ecdf dataframe to increase lookup speed.
        df_subset = self.ecdf_df[all_stocks].copy()
        num_of_stocks = len(all_stocks)
        # We turn our partner stocks into numerical indices so we can use them directly for indexing
        possible_pairs_for_target_stock = np.arange(num_of_stocks)[1:]  # exclude the target stock
        partner_combinations = itertools.combinations(possible_pairs_for_target_stock, 3)
        # Let's add the target stock
        combinations_quadruples = np.array(list((0,) + comb for comb in partner_combinations))
        #We can now use our list of possible quadruples as an index
        df_all_quadruples = df_subset.values[:, combinations_quadruples]
        # Now we can get closer to a vectorized calculation 
        n, _, d = df_all_quadruples.shape
        hd = (d + 1)/(2**d-d-1)
        ecdf_df_product = np.product(df_all_quadruples, axis=-1)
        est1 = hd * (-1 + (2**d / n) * (1-ecdf_df_product).sum(axis=0))
        est2 = hd * (-1 + (2**d / n) * ecdf_df_product.sum(axis=0))
        idx = np.array([(k, l) for l in range(0, d) for k in range(0, l)])
        est3 = -3 + (12 / (n*scipy.special.comb(n, 2, exact=True))) * ((1 - df_all_quadruples[:, :, idx[:, 0]])*(1-df_all_quadruples[:, :, idx[:, 1]])).sum(axis=(0, 2))
        quadruples_scores = (est1 + est2 + est3)/3
        #The quadruple scores have the shape of (19600,1) now
        max_index = np.argmax(quadruples_scores)
        return df_subset.columns[list(combinations_quadruples[max_index])].tolist()

    def _get_quadruples(self, df_corr_top50):
        qf = df_corr_top50.groupby('TARGET_STOCK').apply(self._get_partner_for_target_stock)
        return qf

    def select_pairs(self, df):
        df_returns = self._returns(df)
        self.df_corr = self._ranked_correlations(df)
        df_corr_top50 = self._top_50_correlations(self.df_corr)
        self.ecdf_df = self._transform_df(df_returns)
        return self._get_quadruples(df_corr_top50)
