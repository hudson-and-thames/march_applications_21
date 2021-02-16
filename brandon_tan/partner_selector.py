import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from itertools import combinations

from statsmodels.distributions.empirical_distribution import ECDF

from math import comb

from tqdm import tqdm


class PartnerSelector:
    """
    Implementation of the Partner Selection Framework.
    Hudson and Thames Skilling Challenge, March 2021.
    Traditional Approach, extended approach and geometric approach are implemented.

    https://www.econstor.eu/bitstream/10419/147450/1/870932616.pdf
    """

    def __init__(self, price_data: pd.DataFrame, target_ticker: str, no_of_partners: int = 3, top_n_corr: int = 50):
        """
        Constructor

        Initialise the parter selector object.

        :param price_data: Consumes a price dataframe provided by the user. Index by datetime and columns are tickers. Implicitly, the columns will define the universe that the partners will be found in.
        :type price_data: pd.DataFrame
        :param target_ticker: The target ticker that the user would like to create the grouping around.
        :type price_data: str
        :param no_of_partners: The number of partners to be identified.
        :type price_data: int, optional
        :param top_n_corr: The number of top n correlated securities to be used for partner selection process.
        :type top_n_corr: int, optional
        """

        self.price_data = price_data
        self.target_ticker = target_ticker
        self.no_of_partners = no_of_partners
        self.top_n_corr = top_n_corr
        self.partner_data = None
        self.best_Q = None
        self._prepare_data()

    def select(self, method: str, estimator: int = 1) -> list:
        """
        The main step of the framework. Based on the method that the user would like, we identify the
        best partners based on the target security.

        :param method: Either traditional, extended or geometric method as outlined in the paper.
        :type method: str
        :param estimator: The extended method has 3 estimators. This parameter identifies which estimators to be used. Defaults to 1.
        :type estimator: int, optional
        :raises Exception: In the event an invalid method is chosen, error will be raised.
        :return: The best partners based on user chosen method will be returned in a list format.
        :rtype: list
        """

        method = method.lower()
        if method == 'traditional':
            best_Q = self._traditional_method()

        elif method == 'extended':
            best_Q = self._extended_method(estimator)

        elif method == 'geometric':
            best_Q = self._geometric_method()

        else:
            raise Exception("Unknown method. Either 'traditional' , 'extended' or 'geometric'.")

        return best_Q

    def _traditional_method(self) -> list:
        """
        The function loops through all possible partner combinations and calculates the sum of all pairwise correlations.
        The largest sum of pairwise correlation will be returned.

        :return: The best partners based on sum of pairwise correlation will be returned.
        :rtype: list
        """

        highest_corr = 0
        partner_data = {}
        for i in tqdm(range(len(self.all_possible_partners))):

            # Converting to list as itertools combination generates an itertools object.
            partners = list(self.all_possible_partners[i])

            # When generating the partner combinations, we excluded the target ticker
            # and hence append it back here.
            # Append (instead of insert) was chosen due to O(1) complexity.
            partners.append(self.target_ticker)

            # Finding the sum of the right upper triangle of the correlation matrix.
            sum_of_pairwise_corr = self._upper_tri_sum(self.ranked_returns_data_corr.loc[partners, partners].values)

            if sum_of_pairwise_corr > highest_corr:
                highest_corr = sum_of_pairwise_corr
                best_Q = partners

            partner_data[i] = {
                'Grouping': partners,
                'Spearman Bivariate Correlation Sum': sum_of_pairwise_corr
            }

        self.partner_data = partner_data
        self.best_Q = best_Q
        return best_Q

    def _upper_tri_sum(self,corr_matrix):
        """
        Helper function for traditional method. Given that
        we want the sum of pairwise correlation, it is essentially the
        upper triangular of the correlation matrix, without the
        diagonal. We use a mask to identify that triangle and sum it.

        :param corr_matrix: Spearman correlation matrix.
        :type corr_matrix: np.ndarray
        :return: Sum of upper triangular, less the diagonal.
        :rtype: float
        """
        m = corr_matrix.shape[0]
        r = np.arange(m)
        mask = r[:,None] < r

        return np.sum(corr_matrix[mask])

    def _extended_method(self, estimator: int) -> list:
        """
        The extended method identifies the best partners based on spearman multivariate correlation.
        This function serves as a helper, pointing to the specific implementation of the extended
        method that the user would like to use.

        :param estimator: (int): The type of estimator that you would like to use (p1,p2,p3)
        :raises Exception: In the event an invalid estimator is chosen, error will be raised.
        :return: The best partners based on multivariate spearman correlation will be returned.
        :rtype: list
        """
        # d, number of potential partners, inclusive of the target ticker.
        d = self.no_of_partners + 1

        # Term required for computation of estimators.
        h_d = (d + 1) / (2 ** d - d - 1)

        # Setting it to private attributes.
        self.__d = d
        self.__h_d = h_d

        if estimator == 1:
            best_Q = self._extended_method_estimator_1()
        elif estimator == 2:
            best_Q = self._extended_method_estimator_2()
        elif estimator == 3:
            best_Q = self._extended_method_estimator_3()
        else:
            raise Exception('Invalid estimator provided. See documentation for estimator 1, 2 and 3.')

        return best_Q

    def _extended_method_estimator_1(self) -> list:
        """
        This function loops through all potential partners, using estimator 1
        provided by the paper to estimate multivariate correlation.

        :return: The best partners based on multivariate spearman correlation (estimator 1) will be returned.
        :rtype: list

        https://econpapers.repec.org/article/eeestapro/v_3a77_3ay_3a2007_3ai_3a4_3ap_3a407-416.htm
        """

        highest_p_1 = 0

        # Initialise the dictionary to collect data on all
        # possible groupings for descriptive statistics.
        partner_data = {}

        for i in tqdm(range(len(self.all_possible_partners))):

            # Converting to list as itertools combination generates an itertools object.
            partners = list(self.all_possible_partners[i])

            # When generating the partner combinations, we excluded the target ticker
            # and hence append it back here.
            # Append (instead of insert) was chosen due to O(1) complexity.
            partners.append(self.target_ticker)

            # Converting to numpy array
            ECDF_dataframe = 1 - np.array(self.ECDF_dataframe[partners].values, dtype=np.float64)

            # n , number of data points.
            n = ECDF_dataframe.shape[0]

            curr_p_1 = self.__h_d * (-1 + ((2 ** self.__d) / n) * np.sum(np.prod(ECDF_dataframe, axis=1)))
            
            if curr_p_1 > highest_p_1:
                highest_p_1 = curr_p_1
                best_Q = partners

            partner_data[i] = {
                    'Grouping': partners,
                    'Spearman Multivariate Correlation (Estimator 1)': curr_p_1
                }

        self.partner_data = partner_data
        self.best_Q = best_Q

        return best_Q

    def _extended_method_estimator_2(self) -> list:
        """
        This function loops through all potential partners, using estimator 2
        provided by the paper to estimate multivariate correlation.

        :return: The best partners based on multivariate spearman correlation (estimator 2) will be returned.
        :rtype: list

        https://econpapers.repec.org/article/eeestapro/v_3a77_3ay_3a2007_3ai_3a4_3ap_3a407-416.htm
        """

        highest_p_2 = 0
        partner_data = {}
        for i in tqdm(range(len(self.all_possible_partners))):

            # Converting to list as itertools combination generates an itertools object.
            partners = list(self.all_possible_partners[i])

            # When generating the partner combinations, we excluded the target ticker
            # and hence append it back here.
            # Append (instead of insert) was chosen due to O(1) complexity.
            partners.append(self.target_ticker)

            # Converting to numpy array
            ECDF_dataframe = np.array(self.ECDF_dataframe[partners].values, dtype=np.float64)

            # n , number of data points.
            n = ECDF_dataframe.shape[0]

            curr_p_2 = self.__h_d * (-1 + ((2 ** self.__d) / n) * np.sum(np.prod(ECDF_dataframe, axis=1)))

            if curr_p_2 > highest_p_2:
                highest_p_2 = curr_p_2
                best_Q = partners

            partner_data[i] = {
                'Grouping': partners,
                'Spearman Multivariate Correlation (Estimator 2)': curr_p_2
            }

        self.partner_data = partner_data
        self.best_Q = best_Q

        return best_Q

    def _extended_method_estimator_3(self) -> list:
        """
        This function loops through all potential partners, using estimator 3
        provided by the paper to estimate multivariate correlation.

        :return: The best partners based on multivariate spearman correlation (estimator 3) will be returned.
        :rtype: list

        https://econpapers.repec.org/article/eeestapro/v_3a77_3ay_3a2007_3ai_3a4_3ap_3a407-416.htm
        """

        highest_p_3 = 0
        partner_data = {}
        for i in tqdm(range(len(self.all_possible_partners))):

            # Converting to list as itertools combination generates an itertools object.
            partners = list(self.all_possible_partners[i])

            # When generating the partner combinations, we excluded the target ticker
            # and hence append it back here.
            # Append (instead of insert) was chosen due to O(1) complexity.
            partners.append(self.target_ticker)

            # Converting to numpy array
            ECDF_dataframe = 1 - np.array(self.ECDF_dataframe[partners].values, dtype=np.float64)

            # n , number of data points.
            n = ECDF_dataframe.shape[0]

            # Double summation from the formula
            summation = 0
            for k in range(0, self.__d-1):
                for l in range(i+1, self.__d):
                    summation += np.dot(ECDF_dataframe[:, k], ECDF_dataframe[:, l])

            curr_p_3 = -3 + (12 / (n * comb(self.__d, 2))) * summation

            if curr_p_3 > highest_p_3:
                highest_p_3 = curr_p_3
                best_Q = partners

            partner_data[i] = {
                'Grouping': partners,
                'Spearman Multivariate Correlation (Estimator 3)': curr_p_3
            }

        self.partner_data = partner_data
        self.best_Q = best_Q

        return best_Q

    def _geometric_method(self) -> list:
        """
        The geometric method measures the total euclidean distance between the hyper diagonal in R^n space
        and given ranked data matrix.
        The partners with the lowest euclidean distance (least error from the hyper diagonal) will be selected.
        R^n, where n refers to number of partners + 1 (target ticker).

        :return: The best partners based on euclidean distance between ranked returns data and hyper diagonal in the space.
        :rtype: list
        """

        lowest_euclidean_distance = float('inf')
        partner_data = {}
        for i in tqdm(range(len(self.all_possible_partners))):
            # Converting to list as itertools combination generates an itertools object.
            partners = list(self.all_possible_partners[i])

            # When generating the partner combinations, we excluded the target ticker
            # and hence append it back here.
            # Append (instead of insert) was chosen due to O(1) complexity.
            partners.append(self.target_ticker)

            partners_ranked_values = self.ranked_returns_data[partners].values

            # Calling the private function _diagonal_distance to compute
            # distance between point and the hyper diagonal.
            total_euclidean_distance = np.sum(np.apply_along_axis(self._diagonal_distance, 1, partners_ranked_values))

            if total_euclidean_distance < lowest_euclidean_distance:
                lowest_euclidean_distance = total_euclidean_distance
                best_Q = partners

            partner_data[i] = {
                'Grouping': partners,
                'Total Euclidean Distance': total_euclidean_distance
            }

        self.partner_data = partner_data
        self.best_Q = best_Q

        return best_Q

    def _diagonal_distance(self, vector: np.ndarray):
        """
        Helper private function to calculate the euclidean distance between a given point
        and the hyper diagonal of the space.

        The distance is computed by taking a perpendicular point on the hyper diagonal
        as v1 = (k,k,k,k). Our point of interest is v2 = (w,y,x,z). Solving for the first derivative
        of the distance d(v1,v2), we obtain a formula to compute the distance.

        :param vector: The coordinates of the point.
        :type vector: np.ndarray
        :return: Euclidean distance between point and hyper diagonal of space.
        :rtype: np.float
        """
        # Using a loop so that if the user chooses higher dimensions,
        # this will still work, rather than a hardcoded formula.
        euclidean_distance = 0
        for i in vector:
            # Math can be found in docstring.
            mask = vector != i
            other_terms_sum = sum(vector[mask])
            euclidean_distance += ((3 * i - other_terms_sum) / 4) ** 2
        return euclidean_distance

    def print_info(self) -> pd.DataFrame:
        """
        Prints descriptive statistics of the selection process.
        This is inspired by pd.describe().

        :raises Exception: If selection process has not been done, an error will be raised.
        :return: Dataframe containing descriptive statistics of the selection process.
        :rtype: pd.DataFrame
        """
        if self.partner_data is None:
            raise Exception("Potential parters have not been identified yet. Please run select() before this method.")

        info_df = pd.DataFrame.from_dict(self.partner_data).transpose()
        info_df = info_df[info_df.columns[1]]
        info_df = info_df.astype(float)
        info_df = pd.DataFrame(info_df.describe())
        return info_df

    def plot_selected_partners(self):
        """
        Plots the historical log prices of the best partners that was identified
        by the method of choice by the user.

        :raises Exception: If selection has not been done, error will be raised.
        """
        if self.best_Q is None:
            raise Exception("The best parters have not been identified yet. Please run select() before this method.")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

        for security in self.best_Q:
            rets = np.log(self.price_data[security]).pct_change()
            rets = np.cumprod(1+rets)
            ax.plot(rets, label=security)
        plt.title('Log Prices of Best Partners')
        plt.legend(loc=0)
        plt.show()

    def _prepare_data(self):
        """
        Private function to trigger all the necessary functions to prepare the data
        for the partner selection process.
        """
        self._input_data_format_checker()
        self._discrete_returns_transform()
        self._rank_transformation()
        self._generate_corr_dataframe()
        self._generate_potential_partners()
        self._generate_ECDF_dataframe()

    def _input_data_format_checker(self):
        """
        Private function to check if input data by user has the correct index.

        :raises Exception: If the index cannot be converted to datetime, an error will be raised.
        """
        try:
            self.price_data.index = pd.to_datetime(self.price_data.index)
        except:
            raise Exception("Invalid input price data provided. Please ensure that the dataframe is indexed by datetime.")

    def _discrete_returns_transform(self):
        """
        Private function to obtain daily discrete returns based on input price data.
        """
        temp = self.price_data.copy()
        temp = temp.astype(float)
        temp = temp.pct_change()
        temp = temp.iloc[1:, :]
        temp = temp.dropna(axis=1)
        self.ddr_price_data = temp

    def _rank_transformation(self):
        """
        Private function to perform rank transformation on daily discrete returns.
        """
        self.ranked_returns_data = self.ddr_price_data.rank(method='average')

    def _generate_corr_dataframe(self):
        """
        Private function to generate spearman correlation dataframe from ranked daily returns.
        """
        ranked_returns_data_corr = self.ranked_returns_data.corr(method='spearman')
        self.ranked_returns_data_corr = ranked_returns_data_corr

    def _generate_potential_partners(self):
        """
        Private function to take the target ticker provided and construct all possible
        partner of the target ticker.
        """
        self.potential_partners = self.ranked_returns_data_corr[self.target_ticker].sort_values(ascending=False)[1:self.top_n_corr+1]
        self.top_tickers = self.potential_partners.index
        all_possible_partners = list(combinations(self.top_tickers, self.no_of_partners))
        self.all_possible_partners = all_possible_partners

    def _generate_ECDF_dataframe(self):
        """
        Private function to generate dataframe with ranked returns data
        converted to ECDF.
        The ECDF dataframe is limited to the top tickers identified in
        _generate_potential_partners().
        """
        df = self.ranked_returns_data[list(self.top_tickers) + [self.target_ticker]]
        df_values = np.array(df.values, dtype=np.float64)
        for i in range(len(df.columns)):
            temp_ECDF = ECDF(df_values[:, i])
            df_values[:, i] = temp_ECDF(df_values[:, i])

        self.ECDF_dataframe = pd.DataFrame(df_values, index=df.index, columns=df.columns)
