import pandas as pd
from itertools import combinations

class PartnerSelector():
    """
    """
    def __init__(self, data_returns, dimensions=4, pre_threshold=50):
        self.data = data_returns
        # Calculate correlation matrix via spearman
        self.data_corr = self.data.corr(method='spearman')
        self.dimensions = dimensions
        self._pre_threshold = pre_threshold
        self._partner_data = {ticker:self._get_combination_df(ticker, self._pre_threshold) for ticker in self.data_corr.index}

    def get_partners(self, method='traditional'):
        self._calculate_traditional()

        partners = [self._get_best_traditional(df) for df in self._partner_data.values()]

        return partners

    def _get_best_traditional(self, df):
        idx = df[['traditional']].idxmax()[0]
        return df.loc[idx]['tickers']

    def _calculate_traditional(self):
        for t,df in self._partner_data.items():
            if 'traditional' not in df:
                df['traditional'] = pd.Series(self._get_pairwise_corr_sum(row.tickers) for row in df.itertuples())

    def _get_pairwise_corr_sum(self, tickers):
        pairs = combinations(tickers,2)
        corr_sum = 0
        for p in pairs:
            corr_sum += self.data_corr[p[0]][p[1]]
        return corr_sum

    def _get_combination_df(self, target, count):
        """Builds a DataFrame with a column containing all the ticker combinations
        of the top correlated tickers relative to the target ticker
        """
        top_corr = self._get_top_correlated(target, count)
        # todo: parameterize the dimension
        top_combination = combinations(top_corr.index, self.dimensions-1)
        combined_tickers = [(target, *t) for t in top_combination]
        return pd.DataFrame(data={'tickers':combined_tickers})


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