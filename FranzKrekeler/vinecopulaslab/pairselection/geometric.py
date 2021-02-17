from .base import SelectionBase
import itertools
import tensorflow as tf
import pandas as pd
import numpy as np


class GeometricSelection(SelectionBase):
    def _find_partners_for_target_stock(self, group):
        """
        Helper function for df.groupby("TARTGET_STOCK").apply(...)
         :param group: (group) The group of 50 most correlated stocks
        :return: (List[str]) returns a list of highest correlated quadruple
        """
        target_stock = group.name
        partner_stocks = group.STOCK_PAIR.tolist()
        all_stocks = [target_stock] + partner_stocks
        # We create a subset of our ecdf dataframe to increase lookup speed.
        df_subset = self.ranked_df[all_stocks].copy()
        num_of_stocks = len(all_stocks)
        # We turn our partner stocks into numerical indices so we can use them directly for indexing
        possible_partners_for_target_stock = np.arange(num_of_stocks)[1:]  # exclude the target stock
        partner_combinations = itertools.combinations(possible_partners_for_target_stock, 3)
        # Let's add the target stock
        combinations_quadruples = np.array(list((0,) + comb for comb in partner_combinations))
        # We can now use our list of possible quadruples as an index
        df_all_quadruples = df_subset.values[:, combinations_quadruples]
        line = np.ones(4)
        pp = (np.einsum("ijk,k->ji", df_all_quadruples, line)/np.linalg.norm(line))
        pn = np.sqrt(np.einsum('ijk,ijk->ji', df_all_quadruples, df_all_quadruples))
        res = np.sqrt(pn**2 - pp**2).sum(axis=1)
        min_index = np.argmin(res)
        partners = df_subset.columns[list(combinations_quadruples[min_index])].tolist()
        return partners

    @staticmethod
    def distance_to_line(line, pts):
        """
        original helper function
        """
        dp = np.dot(pts, line)
        pp = dp/np.linalg.norm(line)
        pn = np.linalg.norm(pts, axis=-1)
        return np.sqrt(pn**2 - pp**2)

    def find_partners(self, df: pd.DataFrame):
        """
        main function to return quadruples of correlated stock using a geometric approach for more read the paper that is linked Readme
        :return: (pd.DataFrame)
        """
        df_returns = self._returns(df)
        self.df_corr = self._ranked_correlations(df_returns)
        self.ranked_df = df_returns.rank(pct=True)
        df_corr_top50 = self._top_50_correlations(self.df_corr)
        return self._get_quadruples(df_corr_top50)
