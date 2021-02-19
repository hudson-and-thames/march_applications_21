import itertools
import numpy as np
import scipy.special
import scipy.linalg
from vinecopulaslab.partnerselection.base import SelectionBase


class ExtremalSelection(SelectionBase):
    """
    Class for partner selection based on "A multivariate linear rank test of independence based on a multiparametric copula with cubic sections"
    Mangold 2015
    :return: (pd.DataFrame)
    """

    @staticmethod
    def indepence_test(rank_df):
        raise "Not implemented"
        n, d = rank_df.shape
        permut_mat = np.array(list(itertools.product([-1, 1], repeat=d)), dtype=np.int8)
        sub_mat = permut_mat @ permut_mat.T
        F = (d + sub_mat) / 2
        D = (d - sub_mat) / 2
        cov_mat = ((2 / 15) ** F) * ((1 / 30) ** D)
        cov_mat_inv = scipy.linalg.inv(cov_mat)
        rank_df_norm = rank_df.values / (n + 1)
        pos_rank_df = (rank_df_norm - 1) * (3 * rank_df_norm - 1)
        neg_rank_df = rank_df_norm * (2 - 3 * rank_df_norm)
        prodsum = np.expand_dims(pos_rank_df, axis=0) * np.expand_dims(permut_mat > 0, axis=1) + np.expand_dims(
            neg_rank_df, axis=0) * np.expand_dims(permut_mat < 0, axis=1)
        T = prodsum.prod(-1).mean(-1).reshape((-1, 1))
        return (T.T @ cov_mat_inv @ T)[0][0] * n

    def _find_partners_for_target_stock(self, group):
        """
        Helper function for df.groupby("TARTGET_STOCK").apply(...)
        :param group: (group) The group of 50 most correlated stocks
        :return: (List[str]) returns a list of highest correlated quadruple
        """
        raise "Not implemented"
        target_stock = group.name
        partner_stocks = group.STOCK_PAIR.tolist()
        all_stocks = [target_stock] + partner_stocks
        df_subset = self.df_rank[all_stocks].copy()
        num_of_stocks = len(all_stocks)
        # We turn our partner stocks into numerical indices so we can use them directly for indexing
        possible_partners_for_target_stock = np.arange(num_of_stocks)[1:]  # exclude the target stock
        partner_combinations = itertools.combinations(possible_partners_for_target_stock, 3)
        # Let's add the target stock
        combinations_quadruples = np.array(list((0,) + comb for comb in partner_combinations))
        # We can now use our list of possible quadruples as an index
        df_all_quadruples = df_subset.values[:, combinations_quadruples]
        n, _, d = df_all_quadruples.shape
        permut_mat = np.array(list(itertools.product([-1, 1], repeat=d)), dtype=np.int8)
        sub_mat = permut_mat @ permut_mat.T
        F = (d + sub_mat) / 2
        D = (d - sub_mat) / 2
        cov_mat = ((2 / 15) ** F) * ((1 / 30) ** D)
        cov_mat_inv = scipy.linalg.inv(cov_mat)
        rank_df_norm = df_all_quadruples / (n + 1)
        pos_rank_df = (rank_df_norm - 1) * (3 * rank_df_norm - 1)
        neg_rank_df = rank_df_norm * (2 - 3 * rank_df_norm)
        # Incomplete needs to be fixed
        prodsum = np.add(np.einsum('ijk,lmk->jmik', pos_rank_df, np.expand_dims(permut_mat > 0, axis=0)),
                         np.einsum('ijk,lmk->jmik', neg_rank_df, np.expand_dims(permut_mat < 0, axis=0)))
        T = prodsum.prod(-1).mean(-1)
        # Also incomplete
        T = ((np.expand_dims(T, axis=1) @ np.expand_dims(cov_mat_inv, axis=0)) @ T.T)
        T_results = np.diag(T[:, 0, :]) * n
        max_index = np.argmax(T_results)
        partners = df_subset.columns[list(combinations_quadruples[max_index])].tolist()
        return partners

    def find_partners(self, df):
        """
        main function to return quadruples of correlated stock based on "A multivariate linear rank test of independence based on a multiparametric copula with cubic sections"
        Mangold 2015
        :return: (pd.DataFrame)
        """
        self.df_returns = self._returns(df)
        self.ranked_correlation = self._ranked_correlations(self.df_returns)
        self.ranked_returns = self.df_returns.rank()
        df_corr_top50 = self._top_50_correlations(self.ranked_correlation)
        return self._get_quadruples(df_corr_top50)
