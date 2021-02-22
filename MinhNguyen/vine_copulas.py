import pandas as pd
import numpy as np

from itertools import combinations
from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF

from stat_utils import SpearmanExt


class PartnerSelector():
    """An implementation of the Partner Selection methods as described in 
    Chapter 3.1.1 of Statistical arbitrage with vine copulas by Stubinger, J., Mangold, B., Krauss, C. (2016)

    https://www.econstor.eu/bitstream/10419/147450/1/870932616.pdf

    Attributes
    ----------
        data (pd.DataFrame): original price data input
        data_returns (pd.DataFrame): discrete returns calculated from input data
        data_rank (pd.DataFrame): relative ranks of `data_returns`, this is transposed
        data_corr (pd.DataFrame): correlation matrix from relative ranks
        dimensions (int): number of tickers to put in each combination including the target ticker
    """

    class Method():
        """Constants for the different partner selection prodedures"""
        TRADITIONAL = 'traditional'
        EXTENDED = 'extended'
        GEOMETRIC = 'geometric'


    def __init__(self, data_prices, dimensions=4, threshold=50):
        """Upon initialization of this object, the pre-selection algorithm will be applied to compute
        the combinations of the top correlated symbols relative to each target ticker

        :param data_prices: (pd.DataFrame): input data with tickers along the column axis
        :param dimensions: (int): number of tickers to output in each combination
        :param threshold: (int): the number of top correlated tickers to construct combinations from,
                                used to reduce computational load
        """
        self.data = data_prices
        data_returns = data_prices.pct_change().iloc[1:]
        # Transpose to set index as ticker symbols for convenience
        self.data_returns = data_returns.transpose()
        self.data_rank = self.data_returns.rank(axis=1)
        # Calculate correlation matrix via spearman
        self.data_corr = data_returns.corr(method='spearman')
        self.dimensions = dimensions
        self._threshold = threshold

        # Internal data structure, keys are target tickers and values are DataFrames
        # containing potential combinations along with various association metrics
        self._partner_data = {ticker:self._get_combination_df(ticker, self._threshold) for ticker in tqdm(self.data_corr.index, desc="Calculating preselections")}

    def get_partners(self, method=Method.TRADITIONAL, targets=None):
        """Get the partner combinations. The first ticker in each combination is considerd the 'target' ticker.

        :param method: (str): the method by which to determine partner associations, see `PartnerSelector.Method`
        :param targets: (list): if supplied, only retrieve partners for tickers in this list

        :return (array): NxM array where N is the number of target tickers and M is `self.dimensions`
        """
        target_data = self._filter_targets(targets)

        if method == PartnerSelector.Method.TRADITIONAL:
            self._calculate_traditional(target_data)
            partners = [self._get_best_traditional(df) for df in target_data.values()]
        elif method == PartnerSelector.Method.EXTENDED:
            self._calculate_extended(target_data)
            partners = [self._get_best_extended(df) for df in target_data.values()]
        elif method == PartnerSelector.Method.GEOMETRIC:
            self._calculate_geometric(target_data)
            partners = [self._get_best_geometric(df) for df in target_data.values()]
        else:
            raise ValueError('Unsupported partner selection method')

        return partners
    
    def get_partner_data(self, target):
        return self._partner_data[target]

    def _filter_targets(self, tickers):
        """Gets a subset of the pre-selected partner combination data

        :param tickers: (list): list of symbol strings to filter by

        :return (dict): A subset of `self._partner_data` where keys are found in `tickers`
        """
        result = self._partner_data
        if tickers is not None:
            result = {k:v for k, v in self._partner_data.items() if k in tickers}
        return result

    def _get_best_extended(self, df):
        """Get the best combination according to the extended approach
        
        :param df: (pd.DataFrame): data for a target ticker as found in `self._partner_data`
        :return (array): 1D array containing combination of ticker symbols"""
        idx = df[[PartnerSelector.Method.EXTENDED]].idxmax()[0]
        return df.loc[idx]['tickers']

    def _get_best_traditional(self, df):
        """Get the best combination according to the traditional approach
        
        :param df: (pd.DataFrame): data for a target ticker as found in `self._partner_data`
        :return (array): 1D array containing combination of ticker symbols"""
        idx = df[[PartnerSelector.Method.TRADITIONAL]].idxmax()[0]
        return df.loc[idx]['tickers']

    def _get_best_geometric(self, df):
        """Get the best combination according to the geometric approach
        
        :param df: (pd.DataFrame): data for a target ticker as found in `self._partner_data`
        :return (array): 1D array containing combination of ticker symbols"""
        idx = df[[PartnerSelector.Method.GEOMETRIC]].idxmin()[0]
        return df.loc[idx]['tickers']

    def _calculate_geometric(self, target_data):
        """Calculate associations via geometric approach for each combination.
        
        Results are written back into the input data frames
        
        :param target_data: (dict): target ticker and combination data of the same format as `self._partner_data` 
        """
        d = np.full(self.dimensions, 1) # diagonal vector
        d_dist = np.linalg.norm(d)
        d_norm = (d / d_dist) # normalized diagonal vector
        d_norm_r = d_norm.reshape(self.dimensions, 1)
        for df in tqdm(target_data.values(), desc="Calculating geometric associations"):
            results = []
            for row in df.itertuples():
                ticker_idx = row.tickers_idx
                s_data = self.data_rank.iloc[ticker_idx].to_numpy()
                dist_sqr = self._calculate_geometric_impl(s_data, d_norm, d_norm_r)
                results.append(dist_sqr.sum())
            df[PartnerSelector.Method.GEOMETRIC] = results

    def get_geometric_distance(self, tickers):
        """Gets the geometric distance for the combination of tickers
        
        :param tickers: (list): combination of ticker symbols to retrieve geometric distance data"""
        ticker_idx = self._get_ticker_idx(tickers)
        n = len(tickers)
        d = np.full(n, 1) # diagonal vector
        d_dist = np.linalg.norm(d)
        d_norm = (d / d_dist) # normalized diagonal vector
        d_norm_r = d_norm.reshape(n, 1)
        s_data = self.data_rank.iloc[ticker_idx].to_numpy()
        return self._calculate_geometric_impl(s_data, d_norm, d_norm_r)

    def _calculate_geometric_impl(self, data, diagonal, diagonal_r):
        """Helper method to calculate the Euclidean distance between each sample and the diagonal
        
        :param data: (np.array): 2D array of our multivariate data
        :param diagonal: (np.array): the normalized diagonal vector
        :param diagonal_r: (np.array): reshaped diagonal vector
        :return (np.array): the squared distance from the diagonal for each sample
        """
        # Point projected along the diagonal. 
        # This is the intersection of the diagonal and the line perpendicular running through the point
        p = (diagonal @ data) * diagonal_r
        p_d = p - data
        # Don't bother taking sqrt since we're only interested in relative values
        dist_sqr = (p_d**2).sum(axis=0)
        return dist_sqr

    def _calculate_extended(self, target_data):
        """Calculate associations via extended approach for each combination.
        
        Results are written back into the input data frames
        
        :param target_data: (dict): target ticker and combination data of the same format as `self._partner_data` 
        """
        # ECDF quantile data based on rank observations
        self.ecdf = ecdf = self.data_rank.apply(lambda x: ECDF(x)(x), axis=1, result_type='broadcast')
        def calc_spearman_ext(ticker_idx):
            ecdf_data = ecdf.iloc[ticker_idx].to_numpy()
            s_ext = SpearmanExt(ecdf_data)
            return (s_ext.r1, s_ext.r2, s_ext.r3)

        for df in tqdm(target_data.values(), desc="Calculating extended Spearman associations"):
            results = np.array([calc_spearman_ext(row.tickers_idx) for row in df.itertuples()])
            df[PartnerSelector.Method.EXTENDED] = np.average(results, axis=1)

    def _calculate_traditional(self, target_data):
        """Calculate associations via traditional approach for each combination.
        
        Results are written back into the input data frames
        
        :param target_data: (dict): target ticker and combination data of the same format as `self._partner_data` 
        """
        def calc_pairwise_corr_sum(tickers_idx):
            pairs = combinations(tickers_idx,2)
            corr_sum = 0
            for p in pairs:
                corr_sum += self.data_corr.iloc[p[0],p[1]]
            return corr_sum

        for df in tqdm(target_data.values(), desc="Calculating traditional Spearman associations"):
            df[PartnerSelector.Method.TRADITIONAL] = pd.Series(calc_pairwise_corr_sum(row.tickers_idx) for row in df.itertuples())

    def _get_combination_df(self, target, count):
        """Builds a DataFrame with a column containing all the ticker combinations
        of the top correlated tickers relative to the target ticker
        """  

        top_corr = self._get_top_correlated(target, count)
        top_combination = combinations(top_corr.index, self.dimensions-1)
        target_idx = self.data_returns.index.get_loc(target)
        # Get the ticker names as well as index
        combined_tickers = [([target, *t], [target_idx, *self._get_ticker_idx(t)]) for t in top_combination]
        names, idxs = zip(*combined_tickers)
        return pd.DataFrame(data={'tickers':names, 'tickers_idx':idxs})

    def _get_ticker_idx(self, tickers):
        """Get integer index of tickers in `self.data_returns`
        
        :param tickers: (list): list of tickers to get indices for
        :return (list): integer indices for each ticker
        """
        return [self.data_returns.index.get_loc(t) for t in tickers]


    def _get_top_correlated(self, target, count):
        """Returns the top correlated tickers for the given target ticker

        :param target: (str): target ticker
        :param count: (int): number of top correlated tickers to retrieve
        :return (list): top correlated ticker symbols relative to target ticker
        """
        top_corr = self.data_corr.loc[target].sort_values(ascending=False)
        # We assume the first element is the target ticker
        if count > 0:
            top_corr = top_corr[1:count+1]
        else:
            top_corr = top_corr[1:]
        return top_corr