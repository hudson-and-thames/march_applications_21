from .base import SelectionBase
import itertools
import tensorflow as tf
import pandas as pd
import numpy as np


class TraditionalSelection(SelectionBase):
    """
    This class implements the traditional approach for partner selection. Mentioned section 3.1
    of the paper "Statistical arbitrage with vine copulas" https://www.econstor.eu/bitstream/10419/147450/1/870932616.pdf
    """
    def _get_highest_corr_partners(self, group: pd.DataFrame):
        """
        internal helper function for apply in Pandas
        :param group: (pd.DataFrame) this the result of .groupBy...
        """
        target_stock = group.name
        combinations = group.STOCK_PAIR.tolist()
        # Create a subset dataframe of all top 50 correlated stocks + target stock.
        # This increases the lookup speed.
        stock_selection = [target_stock] + combinations
        df_subset = self.ranked_correlation.loc[stock_selection, stock_selection].copy()
        # Next we will convert the stock names into integers and then get a list of all combinations with a length of 3
        num_of_stocks = len(stock_selection)
        # exclude the target stock
        possible_partners_for_target_stock = np.arange(num_of_stocks)[1:]
        all_possible_combinations = list(itertools.combinations(possible_partners_for_target_stock, 3))
        # Here we one hot encode the array of combinations so we can perform matrix multiplication
        # Currently tensforflow is used because of it's functionality to of it's one hot api.
        # Could be done in pure numpy
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
        max_index = np.argmax((np.expand_dims(one_hot, axis=2) * np.expand_dims(df_subset, axis=0) * np.expand_dims(one_hot, axis=1)).sum(axis=(1, 2)))
        # Finally convert the index to the list of stocks and return the column names
        return [target_stock] + df_subset.columns[list(all_possible_combinations[max_index])].tolist()

    def find_partners(self, df: pd.DataFrame):
        """
        main function to return quadruples of correlated stock (spearman) method
        :return: (pd.DataFrame)
        """
        df_returns = self._returns(df)
        self.ranked_correlation = self._ranked_correlations(df_returns)
        df_corr_top50 = self._top_50_correlations(self.ranked_correlation)
        quadruples = df_corr_top50.groupby('TARGET_STOCK').apply(self._get_highest_corr_partners)
        return quadruples
