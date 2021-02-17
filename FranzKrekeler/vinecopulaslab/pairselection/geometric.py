from .base import SelectionBase
import itertools
import tensorflow as tf
import pandas as pd
import numpy as np


class GeometricSelection(SelectionBase):
    def _get_pair_with_highest_corr_sum(self, group):
        target_stock = group.name
        combinations = group.STOCK_PAIR.tolist()
        stock_selection = [target_stock] + combinations
        df_subset = self.ranked_df[stock_selection].copy()
        num_of_stocks = len(stock_selection)
        possible_pairs_for_target_stock = np.arange(num_of_stocks)[1:]  ## exclude the target stock
        all_possible_combinations = np.array(list((0,) + comb for comb in itertools.combinations(possible_pairs_for_target_stock, 3)))
        all_possible_df = df_subset.values[:,all_possible_combinations]
        # TO FIX
        all_possible_df = all_possible_df.reshape(len(all_possible_combinations), -1,4)
        line = np.ones(4)
        pp = np.dot(all_possible_df, line)/np.linalg.norm(line)
        pn = np.linalg.norm(all_possible_df, axis=-1)
        res = np.sqrt(pn**2 - pp**2).sum(axis=1)

        max_index = np.argmin(res)
        return df_subset.columns[list(all_possible_combinations[max_index])].tolist()


    def distance_to_line(self, line, pts):
        """
        original helper function
        """
        dp = np.dot(pts, line)
        pp = dp/np.linalg.norm(line)
        pn = np.linalg.norm(pts, axis=-1)
        return np.sqrt(pn**2 - pp**2)

    def _get_quadruples(self, df_corr_top50):
        qf = df_corr_top50.groupby('TARGET_STOCK').apply(self._get_pair_with_highest_corr_sum)
        return qf

    def convert_pairs_series_to_list(self, pairs):
        return np.concatenate((np.array([pairs.index.tolist()]).T, pairs.tolist()), axis=1)

    def select_pairs(self, df: pd.DataFrame):
        df_returns = self._returns(df)
        self.df_corr = self._ranked_correlations(df)
        self.ranked_df = df_returns.rank(pct=True)
        df_corr_top50 = self._top_50_correlations(self.df_corr)
        return self._get_quadruples(df_corr_top50)
