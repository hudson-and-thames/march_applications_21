from .base import SelectionBase
import itertools
import tensorflow as tf
import pandas as pd
import numpy as np


class GeometricSelection(SelectionBase):
    def _get_pair_with_highest_corr_sum(self, group):
        target_stock = group.name
        partner_stocks = group.STOCK_PAIR.tolist()
        all_stocks = [target_stock] + partner_stocks
        # We create a subset of our ecdf dataframe to increase lookup speed.
        df_subset = self.ranked_df[all_stocks].copy()
        num_of_stocks = len(all_stocks)
        # We turn our partner stocks into numerical indices so we can use them directly for indexing
        possible_pairs_for_target_stock = np.arange(num_of_stocks)[1:]  # exclude the target stock
        partner_combinations = itertools.combinations(possible_pairs_for_target_stock, 3)
        # Let's add the target stock
        combinations_quadruples = np.array(list((0,) + comb for comb in partner_combinations))
        #We can now use our list of possible quadruples as an index
        df_all_quadruples = df_subset.values[:, combinations_quadruples]
        line = np.ones(4)
        pp = (np.einsum("ijk,k->ji", df_all_quadruples, line)/np.linalg.norm(line))
        pn = np.sqrt(np.einsum('ijk,ijk->ji', df_all_quadruples, df_all_quadruples))
        res = np.sqrt(pn**2 - pp**2).sum(axis=1)
        min_index = np.argmin(res)
        partners = df_subset.columns[list(combinations_quadruples[min_index])].tolist()
        return partners


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

    def _select_pairs_slow(self, df: pd.DataFrame):
        df_returns = self._returns(df)
        self.df_corr = self._ranked_correlations(df)
        self.ranked_df = df_returns.rank(pct=True)
        df_corr_top50 = self._top_50_correlations(self.df_corr)
        for name, group in df_corr_top50.groupby('TARGET_STOCK'):
            return self._get_pair_with_highest_corr_slow(name,group)
        return 

    def _get_pair_with_highest_corr_slow(self, name,group):
        target_stock = name
        partner_stocks = group.STOCK_PAIR.tolist()
        all_stocks = [target_stock] + partner_stocks
        df_subset = self.ranked_df[all_stocks].copy()
        num_of_stocks = len(all_stocks)
        possible_pairs_for_target_stock = np.arange(num_of_stocks)[1:]  ## exclude the target stock
        partner_combinations = itertools.combinations(possible_pairs_for_target_stock, 3)
        # Let's add the target stock
        combinations_quadruples = np.array(list((0,) + comb for comb in partner_combinations))
        #We can now use our list of possible quadruples as an index      
        scores = []
        line = np.ones(4)
        for combination in combinations_quadruples:
            df_combination = df_subset.iloc[:,combination]
            pp = np.dot(df_combination, line)/np.linalg.norm(line)
            pn = np.linalg.norm(df_combination, axis=-1)
            res = np.sqrt(pn**2 - pp**2).sum()
            scores.append(res)
        min_index = np.argmin(scores)
        print(np.min(scores))
        return df_subset.columns[list(combinations_quadruples[min_index])].tolist()

    def select_pairs(self, df: pd.DataFrame):
        df_returns = self._returns(df)
        self.df_corr = self._ranked_correlations(df)
        self.ranked_df = df_returns.rank(pct=True)
        df_corr_top50 = self._top_50_correlations(self.df_corr)
        return self._get_quadruples(df_corr_top50)
