from typing import List
import pandas as pd
import numpy as np
import tensorflow as tf
from .base import SelectionBase


class TraditionalSelection(SelectionBase):
    """
    This class implements the traditional approach for partner selection. Mentioned section 3.1
    of the paper "Statistical arbitrage with vine copulas"
    https://www.econstor.eu/bitstream/10419/147450/1/870932616.pdf
    """

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
        num_of_stocks = len(stock_selection)
        all_possible_combinations = self._prepare_combinations_of_partners(stock_selection, False)
        # Here we one hot encode the array of combinations so we can perform matrix multiplication
        # Currently tensforflow is used because of it's functionality to of it's one hot api.
        # Could be done in pure numpy
        df_subset = self.ranked_correlation.loc[stock_selection, stock_selection].copy()
        one_hot = tf.one_hot(all_possible_combinations, depth=num_of_stocks).numpy()
        # Let's add our target stock one hot encoded
        one_hot_target = np.zeros(num_of_stocks, dtype=bool)
        one_hot_target[0] = True
        one_hot = (one_hot.sum(axis=1) + one_hot_target).astype(bool)
        # it's important to use dtype bool to save memory
        # Here the magic happens:
        # We have encoded our combinations to an array with the shape
        # (19600,51)
        # Now we have to multiply with our correlations matrix which has the shape of
        # (51,51)
        # To do that we broadcast the dimensions to
        # (19600,51,1) * (1,51,51) * (19600,1,51)
        # and then take the sum
        # Afterwards we return the maximum index for the sums
        max_index = np.argmax(
            (np.expand_dims(one_hot, axis=2) * np.expand_dims(df_subset, axis=0) * np.expand_dims(one_hot, axis=1)).sum(
                axis=(1, 2)))
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
        return self._find_partners(close, target_stocks)
