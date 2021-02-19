from typing import List
import numpy as np
import pandas as pd
import scipy.special
from statsmodels.distributions.empirical_distribution import ECDF
from vinecopulaslab.partnerselection.base import SelectionBase


class ExtendedSelection(SelectionBase):
    """
    This class implements the extended approach for partner selection. Mentioned section 3.1
    of the paper "Statistical arbitrage with vine copulas"
    https://www.econstor.eu/bitstream/10419/147450/1/870932616.pdf
    It is an extension to the spearman correlation
    """
    def __init__(self):
        """Initialization
        """
        super().__init__()
        self.corr_returns_top_n = None

    def _partner_selection_approach(self, group) -> List[str]:
        """
        Find the partners stocks for the groupby group of the data df.groupby("TARGET_STOCK").apply(...)
        :param: group (pd.group) The group of n most correlated stocks
        :return: (List[str]) returns a list of highest correlated quadruple
        """
        target_stock = group.name
        partner_stocks = group.STOCK_PAIR.tolist()
        stock_selection = [target_stock] + partner_stocks
        # We create a subset of our ecdf dataframe to increase lookup speed.
        data_subset = self.ecdf_df[stock_selection].copy()
        # We turn our partner stocks into numerical indices so we can use them directly for indexing
        quadruples_combinations = self._prepare_combinations_of_partners(stock_selection)
        # We can now use our list of possible quadruples as an index
        quadruples_combinations_data = data_subset.values[:, quadruples_combinations]
        # Now we can get closer to a vectorized calculation
        # n is equal to the total number of returns d to the number of stocks
        # we use lodash because we don't need the 19600 dimension
        n, _, d = quadruples_combinations_data.shape
        # We split up the given formula
        # For reference:
        # https://github.com/hudson-and-thames/march_applications_21/blob/main/Guide%20for%20the%20Extended%20Approach.pdf
        hd = (d + 1) / (2 ** d - d - 1)
        ecdf_df_product = np.product(quadruples_combinations_data, axis=-1)
        est1 = hd * (-1 + (2 ** d / n) * (1 - ecdf_df_product).sum(axis=0))
        est2 = hd * (-1 + (2 ** d / n) * ecdf_df_product.sum(axis=0))
        # here we create the index as we will use it on specific dimensions
        idx = np.array([(k, l) for l in range(0, d) for k in range(0, l)])
        est3 = -3 + (12 / (n * scipy.special.comb(n, 2, exact=True))) * (
                (1 - quadruples_combinations_data[:, :, idx[:, 0]]) * (
                1 - quadruples_combinations_data[:, :, idx[:, 1]])).sum(axis=(0, 2))
        quadruples_scores = (est1 + est2 + est3) / 3
        # The quadruple scores have the shape of (19600,1) now
        max_index = np.argmax(quadruples_scores)
        return data_subset.columns[list(quadruples_combinations[max_index])].tolist()

    def _preprocess(self, close: pd.DataFrame) -> pd.DataFrame:
        """
        Helper function for preparing the data. Here we already prepare the ecdf
        :param close: (pd.DataFrame) the closing prices
        """
        close.sort_index(axis=1, inplace=True)
        self.close_returns = self.calculate_returns(close)
        self.ranked_correlation = self._ranked_correlation(self.close_returns)
        self.corr_returns_top_n = self._top_n_correlations(self.ranked_correlation)
        self.ecdf_df = self.close_returns.apply(lambda x: ECDF(x)(x), axis=0)

    def find_partners(self, close: pd.DataFrame, target_stocks: List[str] = []):
        """
        Find partners based on an extension of the Spearmann correlation. Mentioned in section 3.1
        of the paper "Statistical arbitrage with vine copulas"
        https://www.econstor.eu/bitstream/10419/147450/1/870932616.pdf
        :param: close (pd.DataFrame) The close prices of the SP500
        :param: target_stocks (List[str]) A list of target stocks to analyze
        :return: (List[str]) returns a list of highest correlated quadruple
        """
        self._preprocess(close)
        # find_partners could be moved to the base class but then it woudln't have the right docstring... looking for best practice
        return self._find_partners(target_stocks)
