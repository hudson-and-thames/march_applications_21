"""
Module for selecting partners at Initialization Period

@author: Joohwan Ko
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF


class SelectPartner:
    """
    Implementation for section 3.1.1: Partner selection from the paper "Statistical arbitrage with vine copulas(2016,Stübinger)
    and this handles 4 different ways of selecting partner at Initialization Period
    1. Traditional Approach
    2. Extended Approach
    3. Geometric Approach
    4. Extremal Approach
    """

    def __init__(self, rank_df):
        self.rank_df = rank_df
        self.rank_corr_df = rank_df.corr()

    def get_partner(self, method='traditional', extended_rho='first'):
        """
        Select partners for copula by choosing a method
        :param method: (str) {'traditional', 'extended', 'geometric', 'extremal'}, default: 'traditional'
        :param extended_rho: (str) {'first', 'second', 'third'}, default: 'first'
        :return: (list) Output matrix quadruple Q
        """
        Q_matrix = []
        total_stocks = self.rank_df.columns
        for ticker in tqdm(total_stocks):
            # Set a variable which indicates sum of the highest pairwise correlations
            highest_sum = float('-inf')
            # Make a list of the best quadruple for a given target stock
            best_quadruple = []
            # Get a list without the target stock
            other_stocks = [x for x in total_stocks if x != ticker]
            # Using combinations, make 49 choose 3 pairs of triple candidates for a given target stock
            triple_candidates = combinations(other_stocks, 3)
            # This for loop calculate the sum of all pairwise correlations for all possible quadruples
            # and decide which one's the highest
            for stock1, stock2, stock3 in triple_candidates:
                # Set a candidate list of the quadruple with the target stock and other three stocks
                quadruple_candidate = [ticker, stock1, stock2, stock3]
                # Select method for partner selection
                if method == 'traditional':
                    pairwise_corr_sum = self._traditional(quadruple_candidate)
                elif method == 'extended':
                    pairwise_corr_sum = self._extended(quadruple_candidate, extended_rho)
                elif method == 'geometric':
                    pairwise_corr_sum = self._geometric(quadruple_candidate)
                elif method == 'extremal':
                    pairwise_corr_sum = self._extremal(quadruple_candidate)
                else:
                    raise KeyError("Invalid method for partner selection")
                # Compare the previous highest sum with a new pairwise_corr_sum
                if pairwise_corr_sum > highest_sum:
                    highest_sum = pairwise_corr_sum
                    best_quadruple = quadruple_candidate
            # Append the best quadruple to the output matrix Q
            Q_matrix.append(best_quadruple)
        return Q_matrix

    def _traditional(self, quadruple_candidate):
        """
        Helper function for calculating traditional approach
        :param quadruple_candidate: (list) four pairs of quadruple candidates
        :return: (float) sum of pairwise correlation
        """
        # Get the correlation matrix for given quadruple candidates
        quadruple_corr_df = self.rank_corr_df.loc[quadruple_candidate, quadruple_candidate]
        # Calculate the pairwise sum of the correlation for the stocks
        pairwise_corr_sum = np.sum(quadruple_corr_df.values)
        return pairwise_corr_sum

    def _extended(self, quadruple_candidate, extended_rho):
        """
        Helper function for calculating extended approach
        :param quadruple_candidate: (list) four pairs of quadruple candidates
        :param extended_rho: (str) different types of estimators
        :return: (float) sum of pairwise correlation
        """
        # According to the paper from Schmid and Schmidt (2007) below are the equations for multivariate
        # rank based measures of association
        h_d = (4 + 1) / (2 ** 4 - 4 - 1)
        # Get a dataframe consists of the quadruple candidates
        quad_df = self.rank_df.loc[:, quadruple_candidate]
        # Build empirical cumulative function for given quadruple_candidate
        ecdf_target = ECDF(quad_df.iloc[:, 0].values)
        ecdf_1 = ECDF(quad_df.iloc[:, 1].values)
        ecdf_2 = ECDF(quad_df.iloc[:, 2].values)
        ecdf_3 = ECDF(quad_df.iloc[:, 3].values)
        # Calculate the multivariate rho according to the method given
        if extended_rho == 'first':
            _rho = 0
            for j in range(len(quad_df)):
                _rho += (1 - ecdf_target(quad_df.iloc[j, 0])) * \
                        (1 - ecdf_1(quad_df.iloc[j, 0])) * \
                        (1 - ecdf_2(quad_df.iloc[j, 0])) * \
                        (1 - ecdf_3(quad_df.iloc[j, 0]))

            _rho = -1 + (16 / len(quad_df)) * _rho
            _rho = h_d * _rho
            return _rho
        elif extended_rho == 'second':
            _rho = 0
            for j in range(len(quad_df)):
                _rho += (ecdf_target(quad_df.iloc[j, 0])) * \
                        (ecdf_1(quad_df.iloc[j, 0])) * \
                        (ecdf_2(quad_df.iloc[j, 0])) * \
                        (ecdf_3(quad_df.iloc[j, 0]))

            _rho = -1 + (16 / len(quad_df)) * _rho
            _rho = h_d * _rho
            return _rho
        elif extended_rho == 'third':
            _rho = 0
            ecdf_dict = {1: ecdf_target, 2: ecdf_1, 3: ecdf_2, 4: ecdf_3}
            for k, l in combinations([1, 2, 3, 4], 2):
                for j in range(len(quad_df)):
                    _rho += (1 - ecdf_dict[k](quad_df.iloc[j, k - 1])) * \
                            (1 - ecdf_dict[l](quad_df.iloc[j, l - 1]))
            _rho += -3 + (12 / (len(quad_df) * 6)) * _rho
            return _rho
        else:
            raise KeyError("Select a proper estimator for the function: one, two or three")

    def _geometric(self, quadruple_candidate):
        """
        Helper function for calculating geometric approach
        :param quadruple_candidate: (list) four pairs of quadruple candidates
        :return: (float) sum of pairwise correlation
        """
        # For geometric method, as we have to fine the lowest value of the diagonal measure,
        # we will set the value as negative for the consistency
        quad_df = self.rank_df.loc[:, quadruple_candidate]
        relative_rank_df = quad_df / len(quad_df)
        # Initial sum to zero
        _sum = 0
        # For every combination of pairs, calculate the geometric correlation of those
        for a, b in combinations([0, 1, 2, 3], 2):
            tmp = relative_rank_df.sort_values(by=quadruple_candidate[a])
            _sum -= np.abs(tmp.iloc[:, a] - tmp.iloc[:, b]).sum()
        return _sum

    def _extremal(self, quadruple_candidate):
        """
        Helper function for calculating extremal approach
        :param quadruple_candidate: still in progress
        :return: still in progress
        """
        return None

    def plot_scatters(self,demo_Q_list):
        """
        Plot 6 scatter plots for a given set of quadruple(4 stocks)
        :param demo_Q_list: (list) four stocks in a row of quadruple matrix Q
        :return: (sns.scatterplot) six scatter plots
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Rank of Each Stock',fontsize=20)
        sns.scatterplot(ax=axes[0, 0], data=self.rank_df, x=demo_Q_list[0], y=demo_Q_list[1])
        sns.scatterplot(ax=axes[0, 1], data=self.rank_df, x=demo_Q_list[0], y=demo_Q_list[2])
        sns.scatterplot(ax=axes[0, 2], data=self.rank_df, x=demo_Q_list[0], y=demo_Q_list[3])
        sns.scatterplot(ax=axes[1, 0], data=self.rank_df, x=demo_Q_list[1], y=demo_Q_list[2])
        sns.scatterplot(ax=axes[1, 1], data=self.rank_df, x=demo_Q_list[1], y=demo_Q_list[3])
        sns.scatterplot(ax=axes[1, 2], data=self.rank_df, x=demo_Q_list[2], y=demo_Q_list[3])

