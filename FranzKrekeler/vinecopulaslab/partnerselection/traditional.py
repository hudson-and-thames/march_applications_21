from typing import List
import pandas as pd
import numpy as np
from vinecopulaslab.partnerselection.base import SelectionBase


class TraditionalSelection(SelectionBase):
    """
    This class implements the traditional approach for partner selection. Mentioned section 3.1
    of the paper "Statistical arbitrage with vine copulas"
    https://www.econstor.eu/bitstream/10419/147450/1/870932616.pdf
    """
    def __init__(self):
        """Initialization
        """
        self.corr_returns_top_n = None

    def _preprocess(self, close: pd.DataFrame) -> pd.DataFrame:
        """
        Helper function for preparing the data.
        :param close: (pd.DataFrame) the closing prices
        """
        self.close_returns = self.calculate_returns(close)
        self.ranked_correlation = self._ranked_correlation(self.close_returns)
        self.corr_returns_top_n = self._top_n_correlations(self.ranked_correlation)

    def _partner_selection_approach(self, group: pd.DataFrame):
        """
        Find the partners stocks for the groupby group of the data df.groupby("TARGET_STOCK").apply(...)
        :param: group (pd.group) The group of n most correlated stocks
        :return: (List[str]) returns a list of highest correlated quadruple
        """
        target_stock = group.name
        potential_partners = group.STOCK_PAIR.tolist()
        stock_selection = [target_stock] + potential_partners
        # we convert our stocks symbols into indices and then return the combinations of the indices
        # For example A,AAPL,... -> 0,1,...
        # these are the quadruples we are going to use for our calculation
        all_possible_combinations = self._prepare_combinations_of_partners(stock_selection)
        df_subset = self.ranked_correlation.loc[stock_selection, stock_selection].copy()
        # Here the magic happens:
        # We use the combinations as an index
        corr_matrix_a = df_subset.values[:,all_possible_combinations]
        # corr_matrix_a has now the shape of (51, 19600, 4)
        # We now use take along axis to get the shape (4,19600,4), then we can sum the first and the last dimension
        corr_sums = np.sum(np.take_along_axis(corr_matrix_a, all_possible_combinations.T[..., np.newaxis], axis=0),axis=(0,2))  
        # this returns the shape of
        # (19600,1)
        # Afterwards we return the maximum index for the sums
        max_index = np.argmax(
            corr_sums)
        # Finally convert the index to the list of stocks and return the column names
        return [target_stock] + df_subset.columns[list(all_possible_combinations[max_index])].tolist()

    def find_partners(self, close: pd.DataFrame, target_stocks: List[str] = []):
        """
        Find partners based on the traditional apprach mentioned in section 3.1.
        Returns quadruples of highest scoring sum of correlated stock (spearman) method 
        of the paper "Statistical arbitrage with vine copulas"
        https://www.econstor.eu/bitstream/10419/147450/1/870932616.pdf
        :param: close (pd.DataFrame) The close prices of the SP500
        :param: target_stocks (List[str]) A list of target stocks to analyze
        :return: (List[str]) returns a list of highest correlated quadruple
        """
        self._preprocess(close)
        # find_partners could be moved to the base class but then it woudln't have the right docstring... looking for best practice
        return self._find_partners(target_stocks)
