from typing import List
import numpy as np
import pandas as pd
from vinecopulaslab.partnerselection.base import SelectionBase


class GeometricSelection(SelectionBase):
    """
    This class implements the geometric approach for partner selection. Mentioned section 3.1
    of the paper "Statistical arbitrage with vine copulas"
    https://www.econstor.eu/bitstream/10419/147450/1/870932616.pdf
    """
    def __init__(self):
        """Initialization
        """
        super().__init__()
        self.corr_returns_top_n = None


    def _partner_selection_approach(self, group):
        """
        Find the partners stocks for the groupby group of the data df.groupby("TARGET_STOCK").apply(...)
        :param: group (pd.group) The group of n most correlated stocks
        :return: (List[str]) returns a list of highest correlated quadruple
        """
        target_stock = group.name
        partner_stocks = group.STOCK_PAIR.tolist()
        stock_selection = [target_stock] + partner_stocks
        # We create a subset of our rank transformed dataframe to increase lookup speed.
        data_subset = self.ranked_returns_pct[stock_selection].copy()
        combinations_quadruples = self._prepare_combinations_of_partners(stock_selection)
        # We can now use our list of possible quadruples as an index
        quadruples_combinations_data = data_subset.values[:, combinations_quadruples]
        # n is equal to the total number of returns d to the number of stocks
        # we use lodash because we don't need the 19600 dimension
        n, _, d = quadruples_combinations_data.shape
        # Now we will create a diagonal for our distance calculation.
        # Please refer to the paper
        line = np.ones(d)
        # Einsum is great for specifying which dimension to multiply together
        # this extends the distance method for all 19600 combinations
        pp = (np.einsum("ijk,k->ji", quadruples_combinations_data, line) / np.linalg.norm(line))
        pn = np.sqrt(np.einsum('ijk,ijk->ji', quadruples_combinations_data, quadruples_combinations_data))
        distance_scores = np.sqrt(pn ** 2 - pp ** 2).sum(axis=1)
        min_index = np.argmin(distance_scores)
        partners = data_subset.columns[list(combinations_quadruples[min_index])].tolist()
        return partners

    @staticmethod
    def distance_to_line(line, pts):
        """
        original helper function
        :param line: the line endpoint assuming it starts at point zero. For example np.array([1,1,1]) for a 3d line
        :param pts: the points to measure the distance to the line
        :return: float np.array with distances
        """
        dp = np.dot(pts, line)
        pp = dp / np.linalg.norm(line)
        pn = np.linalg.norm(pts, axis=1)
        return np.sqrt(pn ** 2 - pp ** 2)

    def _preprocess(self, close: pd.DataFrame) -> pd.DataFrame:
        """
        Helper function for preparing the data.
        :param close: (pd.DataFrame) the closing prices
        """
        self.close_returns = self.calculate_returns(close)
        self.ranked_correlation = self._ranked_correlation(self.close_returns)
        self.corr_returns_top_n = self._top_n_correlations(self.ranked_correlation)
        self.ranked_returns_pct = self._rankings_pct(self.close_returns)

    def find_partners(self, close: pd.DataFrame, target_stocks: List[str] = []):
        """
        Find partners based on the geometric mentioned in section 3.1
        of the paper "Statistical arbitrage with vine copulas" https://www.econstor.eu/bitstream/10419/147450/1/870932616.pdf
        :param: close (pd.DataFrame) The close prices of the SP500
        :param: target_stocks (List[str]) A list of target stocks to analyze
        :return: (List[str]) returns a list of highest correlated quadruple
        """
        self._preprocess(close)
        # find_partners could be moved to the base class but then it woudln't have the right docstring... looking for best practice
        return self._find_partners(target_stocks)
