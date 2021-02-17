import itertools
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.special
from statsmodels.distributions.empirical_distribution import ECDF
from .base import SelectionBase


class ExtendedSelection(SelectionBase):
    def _transform_df(self, df_returns):
        ecdf_df = df_returns.apply(lambda x: ECDF(x)(x), axis=0)
        return ecdf_df

    def _get_pair_with_highest_corr_sum(self, group):
        target_stock = group.name
        combinations = group.STOCK_PAIR.tolist()
        stock_selection = [target_stock] + combinations
        df_subset = self.ecdf_df[stock_selection].copy()
        num_of_stocks = len(stock_selection)
        possible_pairs_for_target_stock = np.arange(num_of_stocks)[1:]  # exclude the target stock
        all_possible_combinations = np.array(list((0,) + comb for comb in itertools.combinations(possible_pairs_for_target_stock, 3)))
        all_possible_df = df_subset.values[:, all_possible_combinations]
        n, _, d = all_possible_df.shape
        hd = (d + 1)/(2**d-d-1)
        ecdf_df_product = np.product(all_possible_df, axis=-1)
        est1 = hd * (-1 + (2**d / n) * (1-ecdf_df_product).sum(axis=0))
        est2 = hd * (-1 + (2**d / n) * ecdf_df_product.sum(axis=0))
        idx = np.array([(k, l) for l in range(0, d) for k in range(0, l)])
        est3 = -3 + (12 / (n*scipy.special.comb(n, 2, exact=True))) * ((1 - all_possible_df[:, :, idx[:, 0]])*(1-all_possible_df[:, :, idx[:, 1]])).sum(axis=(0, 2))
        res = (est1 + est2 + est3)/3
        max_index = np.argmax(res)
        return df_subset.columns[list(all_possible_combinations[max_index])].tolist()

    def _get_quadruples(self, df_corr_top50):
        qf = df_corr_top50.groupby('TARGET_STOCK').apply(self._get_pair_with_highest_corr_sum)
        return qf

    def select_pairs(self, df):
        df_returns = self._returns(df)
        self.df_corr = self._ranked_correlations(df)
        df_corr_top50 = self._top_50_correlations(self.df_corr)
        self.ecdf_df = self._transform_df(df_returns)
        return self._get_quadruples(df_corr_top50)
