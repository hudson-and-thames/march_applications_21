from typing import List
import itertools
import logging
import pandas as pd
import numpy as np
import scipy.special
import scipy.linalg
from vinecopulaslab.partnerselection.base import SelectionBase


class ExtremalSelection(SelectionBase):
    """
    Class for partner selection based on "A multivariate linear rank test of independence
    based on a multiparametric copula with cubic sections"
    Mangold 2015
    """

    def __init__(self):
        """Initialization
        """
        super().__init__()
        self.corr_returns_top_n = None

    def _partner_selection_approach(self, group):
        """
        Approach function Partner selection based on "A multivariate linear rank test of independence based on
        a multiparametric copula with cubic sections"
        for df.groupby("TARGET_STOCK").apply(...)
        This has only been implemented for performance testing.
        References:
        https://www.researchgate.net/publication/309408947_A_multivariate_linear_rank_test_of_independence_based_on_a_multiparametric_copula_with_cubic_sections
        https://pypi.org/project/Independence-test/
        :param group: (group) The group of 50 most correlated stocks
        :return: (List[str]) returns a list of highest correlated quadruple
        """
        logging.warning("Extremal approach is still under construction")
        target_stock = group.name
        partner_stocks = group.STOCK_PAIR.tolist()
        stock_selection = [target_stock] + partner_stocks
        # We create a subset of our ranked returns dataframe to increase lookup speed.
        data_subset = self.ranked_returns[stock_selection].copy()
        # We turn our partner stocks into numerical indices so we can use them directly for indexing
        quadruples_combinations = self._prepare_combinations_of_partners(stock_selection)
        # We can now use our list of possible quadruples as an index
        quadruples_combinations_data = data_subset.values[:, quadruples_combinations]
        # Now we can get closer to a vectorized calculation
        # n is equal to the total number of returns d to the number of stocks
        # we use lodash because we don't need the 19600 dimension
        n, _, d = quadruples_combinations_data.shape
        # Here the math from the Mangold 2015 paper begins
        permut_mat = np.array(list(itertools.product([-1, 1], repeat=d)), dtype=np.int8)
        sub_mat = permut_mat @ permut_mat.T
        F = (d + sub_mat) / 2
        D = (d - sub_mat) / 2
        cov_mat = ((2 / 15) ** F) * ((1 / 30) ** D)
        cov_mat_inv = scipy.linalg.inv(cov_mat)
        rank_df_norm = quadruples_combinations_data / (n + 1)
        pos_rank_df = (rank_df_norm - 1) * (3 * rank_df_norm - 1)
        neg_rank_df = rank_df_norm * (2 - 3 * rank_df_norm)
        # performance here is still lagging
        # Proposition 3.3. from the paper
        pos_neg_combined = np.add(np.einsum('ijk,lmk->jmik', pos_rank_df, np.expand_dims(permut_mat > 0, axis=0)),
                                  np.einsum('ijk,lmk->jmik', neg_rank_df, np.expand_dims(permut_mat < 0, axis=0)))
        TNP = pos_neg_combined.prod(axis=-1).mean(-1)
        # Incomplete: Still not documented
        # performance here is also not optimal here
        T = ((np.expand_dims(TNP, axis=1) @ np.expand_dims(cov_mat_inv, axis=0)) @ TNP.T)
        T_results = np.diag(T[:, 0, :]) * n
        max_index = np.argmax(T_results)
        partners = data_subset.columns[list(quadruples_combinations[max_index])].tolist()
        # Please take this with a grain of salt, I was too obsessed with a proof of concept.
        return partners

    def _preprocess(self, close: pd.DataFrame) -> pd.DataFrame:
        """
        Helper function for preparing the data.
        :param close: (pd.DataFrame) the closing prices
        """
        close.sort_index(axis=1, inplace=True)
        self.close_returns = self.calculate_returns(close)
        self.ranked_correlation = self._ranked_correlation(self.close_returns)
        self.corr_returns_top_n = self._top_n_correlations(self.ranked_correlation)
        self.ranked_returns = self.close_returns.rank()

    def find_partners(self, close: pd.DataFrame, target_stocks: List[str] = []):
        """
        Find partners based on the extremal approach mentioned in section 3.1
        of the paper "Statistical arbitrage with vine copulas"
        https://www.econstor.eu/bitstream/10419/147450/1/870932616.pdf
        Based on the paper  Class for partner selection based on "A multivariate linear rank test of independence
        based on a multiparametric copula with cubic sections" Mangold 2015
        :param: close (pd.DataFrame) The close prices of the SP500
        :param: target_stocks (List[str]) A list of target stocks to analyze
        :return: (List[str]) returns a list of highest correlated quadruple
        """
        self._preprocess(close)
        # find_partners could be moved to the base class but then it wouldn't have the right docstring...
        # looking for best practice
        return self._find_partners(target_stocks)
