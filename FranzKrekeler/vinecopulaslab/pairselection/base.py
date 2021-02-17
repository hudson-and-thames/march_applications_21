# Author: Franz Krekeler 2021

class SelectionBase:
    def _returns(self, df):
        return df.pct_change(fill_method='ffill').dropna(how='all')

    def _ranked_correlations(self, df):
        df_returns = self._returns(df)
        df_corr = df_returns.corr("spearman")
        return df_corr

    def _ranked_pct(self, df, pct=False):
        df_returns = self._returns(df).rank(pct=True)
        return df_returns

    def _top_50_correlations(self, df_corr):
        # Filter self correlated and too highly correlated stocks
        df_corr_unstacked = df_corr[df_corr < 1].unstack().sort_values(ascending=False)
        df_corr_unstacked = df_corr_unstacked.reset_index().dropna()
        df_corr_unstacked.columns = "TARGET_STOCK", "STOCK_PAIR", "CORRELATION"
        df_corr_top50 = df_corr_unstacked.groupby("TARGET_STOCK").head(50)
        return df_corr_top50.sort_values(['TARGET_STOCK', 'CORRELATION'])
