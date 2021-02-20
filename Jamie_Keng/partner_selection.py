import heapq
import numpy as np
from itertools import permutations


class PairSelection:

    def __init__(self, return_df, partner_num):
        self.return_df = return_df
        self.partner_num = partner_num
        self.length = (len(self.return_df) + 1)
        self.rank_df = self.return_df.rank() / self.length

    def __partner_selection(self):
        """ Generate potential partner stocks."""

        # Obtain pairwise Spearman rho in thw dataset.
        df_corr = self.return_df.corr(method="spearman")
        # An empty dictionary
        partner_dict = {target: None for target in range(len(df_corr))}
        for target_stock in range(len(df_corr)):
            lst = []
            for stock in range(len(df_corr)):
                lst.append(df_corr.iloc[target_stock][stock])
                top_partner = heapq.nlargest(self.partner_num, range(len(lst)),
                                             key=lst.__getitem__)
            partner_dict[target_stock] = top_partner

        return df_corr, partner_dict

    @staticmethod
    def __potential_quadruples(partner_dict):
        """Form quadruples out of the potential partner stocks pool.

        :param partner_dict: (pandas series): potential partner stocks for each stock in the dataset.
        """
        # Potential quadruples formation
        pairs_dict = {}
        for target_stock in partner_dict:
            pairs_dict[target_stock] = list(permutations(
                partner_dict[target_stock], 4))
        # Remove quadruples without target stocks.
        selected_pairs_dict = {}
        for target_stock in range(len(pairs_dict)):
            selected_pairs_dict[target_stock] = [item for item in pairs_dict[target_stock]
                                                 if item[0] == target_stock or
                                                 item[1] == target_stock or
                                                 item[2] == target_stock or
                                                 item[3] == target_stock]
        # Remove duplicated quadruples.
        final_pair_dict = {}

        for target_stock in range(len(selected_pairs_dict)):
            final_pair_dict[target_stock] = set(tuple(sorted(x))
                                                for x in selected_pairs_dict[target_stock])

        return final_pair_dict

    @staticmethod
    def __euclidean_distance(quadruple_point):
        """Calculate the Euclidean distance.

        :param quadruple: (np.array): points representing ranks of quadruples.
        """
        distance = np.linalg.norm(quadruple_point -
                                  quadruple_point @ np.array([1, 1, 1, 1]) / 4 * np.array([1, 1, 1, 1]))

        return distance

    @staticmethod
    def __final_quadruple(quadruple_dict):
        """Select the final quadruple when users choose the traditional approach, the extended approach,
           the extremal approach.

        :param quadruple_dict: (dict): quadruple candidates
        """
        result_pair_dict = {}
        for pair in quadruple_dict:
            # Get the key(the quadruple) that yield the maximum value of the desired measure.
            keymax = max(quadruple_dict[pair], key=quadruple_dict[pair].get)
            result_pair_dict[pair] = keymax

        return result_pair_dict

    @staticmethod
    def __geometric_final_quadruple(quadruple_dict):
        """Select the final quadruple when the geometric approach is in use.

        :param quadruple_dict: (dict): quadruple candidates
        """
        result_pair_dict = {}
        for pair in quadruple_dict:
            # Get the key(the quadruple) that yield the minimum Euclidean distance.
            keymax = min(quadruple_dict[pair], key=quadruple_dict[pair].get)
            result_pair_dict[pair] = keymax

        return result_pair_dict

    def __solution(self, final_dict, approach):

        result = {}
        for target_stock in final_dict:
            result[target_stock] = {}
            for pair_num in range(0, len(final_dict[target_stock])):
                # Indexes of target stock + 3 partner stocks on the dataframe, sp500
                indexes = list(list(final_dict[target_stock])[pair_num])
                # Calculate Multivariate Spearman's rho
                # Normalized ranks of each stocks
                target = self.rank_df.iloc[:, indexes[0]]
                part_1 = self.rank_df.iloc[:, indexes[1]]
                part_2 = self.rank_df.iloc[:, indexes[2]]
                part_3 = self.rank_df.iloc[:, indexes[3]]

                if approach == "extended":
                    h = (4 + 1) / (2 ** 4 - 4 - 1) * 2 ** 4 / len(self.return_df)
                    sum_ranks_prod = np.multiply(np.multiply(np.multiply(target, part_1), part_2), part_3).sum()
                    mulspearman = h * sum_ranks_prod - 1
                    result[target_stock][tuple(indexes)] = mulspearman

                if approach == "geometric":
                    # A created list to store distance values
                    lst_dist = []
                    for j in range(0, len(self.return_df)):
                        point = np.array([target[j], part_1[j], part_2[j], part_3[j]])
                        distance = self._PairSelection__euclidean_distance(point)
                        lst_dist.append(distance)
                    # Sum of distance of a quadruple
                    sum_dist = sum(lst_dist)
                    # Store sum of distance to pair dictionary
                    result[target_stock][tuple(indexes)] = sum_dist

                if approach == "extremal":
                    a = target
                    b = part_1
                    c = part_2
                    d = part_3
                    # A created list to store distance values
                    # Implementation of Proposition 3 on page 17
                    ans_lst = []
                    for j in range(0, len(self.return_df)):
                        ans = 1 - 2 * (a[j] + b[j] + c[j] + d[j]) + \
                              4 * (a[j] * b[j] + a[j] * c[j] + a[j] * d[j] + b[j] *
                                   c[j] + b[j] * d[j] + c[j] * d[j]) - \
                              8 * (a[j] * b[j] * c[j] + a[j] * b[j] * d[j] + a[j] *
                                   c[j] * d[j] + b[j] * c[j] * d[j]) + \
                              16 * (a[j] * b[j] * c[j] * d[j])
                        ans_lst.append(ans)
                    # Mean of the values of density derivative
                    statistic = sum(ans_lst) / len(self.return_df)
                    # Store sum of distance to pair dictionary
                    result[target_stock][tuple(indexes)] = statistic

        if approach == "geometric":
            final_quadruples = self._PairSelection__geometric_final_quadruple(result)
            return final_quadruples

        else:
            final_quadruples = self._PairSelection__final_quadruple(result)
            return final_quadruples


class Traditional(PairSelection):
    """Generate the final quadruple using the traditional approach. """

    def solve(self):
        df_corr, partner_dict = self._PairSelection__partner_selection()
        final_pair_dict = self._PairSelection__potential_quadruples(partner_dict)

        result = {}
        for target_stock in final_pair_dict:
            result[target_stock] = {}
            for pair_num in range(0, len(final_pair_dict[target_stock])):
                # Indexes of target stock + 3 partner stocks on the correlation matrix, df_corr
                indexes = list(list(final_pair_dict[target_stock])[pair_num])
                # Sum up all the correlation values and subtract 4 and divided by 2
                sum_corr = (df_corr.iloc[indexes, indexes].sum().sum() - 4) / 2
                result[target_stock][tuple(indexes)] = sum_corr
        final_quadruples = self._PairSelection__final_quadruple(result)

        return final_quadruples


class Extended(PairSelection):
    """Generate the final quadruple using the extended approach. """

    def solve(self):
        df_corr, partner_dict = self._PairSelection__partner_selection()
        final_pair_dict = self._PairSelection__potential_quadruples(partner_dict)
        final_quadruples = self._PairSelection__solution(final_pair_dict, approach="extended")

        return final_quadruples


class Geometric(PairSelection):
    """Generate the final quadruple using the geometric approach. """

    def solve(self):
        df_corr, partner_dict = self._PairSelection__partner_selection()
        final_pair_dict = self._PairSelection__potential_quadruples(partner_dict)
        final_quadruples = self._PairSelection__solution(final_pair_dict, approach="geometric")
        return final_quadruples


class Extremal(PairSelection):
    """Generate the final quadruple using the extremal approach. """

    def solve(self):
        df_corr, partner_dict = self._PairSelection__partner_selection()
        final_pair_dict = self._PairSelection__potential_quadruples(partner_dict)
        final_quadruples = self._PairSelection__solution(final_pair_dict, approach="extremal")
        return final_quadruples
