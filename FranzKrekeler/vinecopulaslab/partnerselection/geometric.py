from .base import SelectionBase
import itertools
import tensorflow as tf
import pandas as pd
import numpy as np


class GeometricSelection(SelectionBase):
    """
    This class implements the geometric approach for partner selection. Mentioned section 3.1
    of the paper "Statistical arbitrage with vine copulas" https://www.econstor.eu/bitstream/10419/147450/1/870932616.pdf
    """
    def _find_partners_for_target_stock(self, group):
        """
        Helper function for df.groupby("TARTGET_STOCK").apply(...)
         :param group: (group) The group of 50 most correlated stocks
        :return: (List[str]) returns a list of highest scored quadruple
        """
        target_stock = group.name
        partner_stocks = group.STOCK_PAIR.tolist()
        all_stocks = [target_stock] + partner_stocks
        # We create a subset of our rank transformed dataframe to increase lookup speed.
        df_subset = self.ranked_df[all_stocks].copy()
        num_of_stocks = len(all_stocks)
        # We turn our partner stocks into numerical indices so we can use them directly for indexing
        possible_partners_for_target_stock = np.arange(num_of_stocks)[1:]  # exclude the target stock
        partner_combinations = itertools.combinations(possible_partners_for_target_stock, 3)
        # Let's add the target stock
        combinations_quadruples = np.array(list((0,) + comb for comb in partner_combinations))
        # We can now use our list of possible quadruples as an index
        df_all_quadruples = df_subset.values[:, combinations_quadruples]
        # Now we will create a diagonal for our distance calculation.
        # Please reffer to the paper
        n,d = self.ranked_df.shape
        line = np.ones(d)
        #this extends the distance method for all 19600 combinations
        pp = (np.einsum("ijk,k->ji", df_all_quadruples, line)/np.linalg.norm(line))
        pn = np.sqrt(np.einsum('ijk,ijk->ji', df_all_quadruples, df_all_quadruples))
        distance_scores = np.sqrt(pn**2 - pp**2).sum(axis=1)
        min_index = np.argmin(distance_scores)
        partners = df_subset.columns[list(combinations_quadruples[min_index])].tolist()
        return partners

    @staticmethod
    def distance_to_line(line, pts):
        """
        original helper function
        :param line: the line endpoint assuming it starts at point zero. For example np.array([1,1,1]) for a 3d line
        :param pts: the points to measure the distance to the line
        :return: float np.array with distances
        """
        dp = np.dot(pts,line)
        pp = dp/np.linalg.norm(line)
        pn = np.linalg.norm(pts, axis=1)
        print(np.sqrt((pn**2) - (pp**2)))
        return np.sqrt(pn**2 - pp**2)

    def find_partners(self, df: pd.DataFrame):
        """
        main function to return quadruples of correlated stock using the geometric approach of the paper "Statistical arbitrage with vine copulas" section 3.1
        :return: (pd.DataFrame)
        """
        df_returns = self._returns(df)
        self.ranked_correlation = self._ranked_correlations(df_returns)
        self.ranked_df = df_returns.rank(pct=True)
        df_corr_top50 = self._top_50_correlations(self.ranked_correlation)
        return self._get_quadruples(df_corr_top50)
