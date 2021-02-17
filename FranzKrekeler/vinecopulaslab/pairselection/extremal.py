import itertools
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.special
from statsmodels.distributions.empirical_distribution import ECDF
from .base import SelectionBase


class ExtremalSelection(SelectionBase):
    def _get_pair_with_highest_corr_sum(self, group):
        target_stock = group.name
        combinations = group.STOCK_PAIR.tolist()
        stock_selection = [target_stock] + combinations
        df_subset = self.df_returns[stock_selection].copy()
        num_of_stocks = len(stock_selection)
        possible_pairs_for_target_stock = np.arange(num_of_stocks)[1:]  ##exclude the target stock
        all_possible_combinations = np.array(list((0,) + comb for  comb in itertools.combinations(possible_pairs_for_target_stock, 3)))
        all_possible_df = df_subset.values[:, all_possible_combinations]
        n, _, d =  all_possible_df.shape        
        matrix = np.array(list(itertools.product([-1, 1], repeat=n)), dtype=np.int8)
        matrix_inner_p = matrix @ matrix.T
        F = (n+matrix)/2
        D = (n-matrix)/2
        T_per = matrix
        rank_n = rank_df.div(n+1)
        #TODO:

    def _get_quadruples(self, df_corr_top50):
        qf = df_corr_top50.groupby('TARGET_STOCK').apply(self._get_pair_with_highest_corr_sum)
        return qf

    def select_pairs(self, df):
        self.df_returns = self._returns(df)
        self.df_corr = self._ranked_correlations(df)
        df_corr_top50 = self._top_50_correlations(self.df_corr)
        self.ecdf_df = self._transform_df(self.df_returns)
        return self._get_quadruples(df_corr_top50)
        