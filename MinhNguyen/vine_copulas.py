import pandas as pd
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

from stat_utils import SpearmanExt
from itertools import combinations
from tqdm import tqdm


class PartnerSelector():
    """
    """
    def __init__(self, data_returns, dimensions=4, pre_threshold=50):
        # Transpose to set index as ticker symbols
        self.data = data_returns.transpose()
        self.data_rank = self.data.rank()
        # Calculate correlation matrix via spearman
        self.data_corr = data_returns.corr(method='spearman')
        self.dimensions = dimensions
        self._pre_threshold = pre_threshold
        self._partner_data = {ticker:self._get_combination_df(ticker, self._pre_threshold) for ticker in tqdm(self.data_corr.index, desc="Calculating preselections")}

    def get_partners(self, method='traditional', targets=None):
        target_data = self._filter_targets(targets)

        if method == 'traditional':
            self._calculate_traditional(target_data)
            partners = [self._get_best_traditional(df) for df in target_data.values()]
        elif method == 'extended':
            self._calculate_extended(target_data)
            partners = [self._get_best_extended(df) for df in target_data.values()]
        else:
            # todo: exception
            partners = []

        return partners

    def _filter_targets(self, tickers):
        result = self._partner_data
        if tickers is not None:
            result = {k:v for k, v in self._partner_data.items() if k in tickers}
        return result

    def _get_best_extended(self, df):
        idx = df[['r1']].idxmax()[0]
        return df.loc[idx]['tickers']

    def _get_best_traditional(self, df):
        idx = df[['traditional']].idxmax()[0]
        return df.loc[idx]['tickers']

    def _calculate_extended(self, target_data):
        # ECDF quantile data based on rank observations
        self.ecdf = ecdf = self.data_rank.apply(lambda x: ECDF(x)(x), axis=1, result_type='broadcast')
        for df in tqdm(target_data.values(), desc="Calculating extended Spearman associations"):
            r1 = []
            r2 = []
            r3 = []
            for row in df.itertuples():
                ticker_idx = row.tickers_idx
                # Grab quantile data for this combination of tickers
                s_data = ecdf.iloc[ticker_idx].to_numpy()
                s_ext = SpearmanExt(s_data)
                r1.append(s_ext.r1)
                r2.append(s_ext.r2)
                r3.append(s_ext.r3)       
            df['r1'] = r1
            df['r2'] = r2
            df['r3'] = r3

    def _calculate_traditional(self, target_data):
        for df in tqdm(target_data.values(), desc="Calculating traditional Spearman associations"):
            df['traditional'] = pd.Series(self._get_pairwise_corr_sum(row.tickers_idx) for row in df.itertuples())

    def _get_pairwise_corr_sum(self, tickers_idx):
        pairs = combinations(tickers_idx,2)
        corr_sum = 0
        for p in pairs:
            corr_sum += self.data_corr.iloc[p[0],p[1]]
        return corr_sum

    def _get_combination_df(self, target, count):
        """Builds a DataFrame with a column containing all the ticker combinations
        of the top correlated tickers relative to the target ticker
        """  

        top_corr = self._get_top_correlated(target, count)
        top_combination = combinations(top_corr.index, self.dimensions-1)
        target_idx = self.data.index.get_loc(target)
        # Get the ticker names as well as index
        combined_tickers = [([target, *t], [target_idx, *self._get_ticker_idx(t)]) for t in top_combination]
        names, idxs = zip(*combined_tickers)
        return pd.DataFrame(data={'tickers':names, 'tickers_idx':idxs})

    def _get_ticker_idx(self, tickers):
        return [self.data.index.get_loc(t) for t in tickers]


    def _get_top_correlated(self, target, count):
        """Returns the top correlated tickers for the given target ticker
        """
        top_corr = self.data_corr.loc[target].sort_values(ascending=False)
        # We assume the first element is the target ticker
        if count > 0:
            top_corr = top_corr[1:count+1]
        else:
            top_corr = top_corr[1:]
        return top_corr