# Author: Franz Krekeler 2021
import pandas as pd


class SelectionBase:
    """The base class for the partner selection framework.
    """
    def _returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """calculate percentage based daily returns
        :param df (pd.DataFrame): The columns must the closing prices of the wanted stocks
        """
        return df.pct_change(fill_method='ffill').dropna(how='all')

    def _find_partners_for_target_stock(self, group):
        """
        Helper function for df.groupby("TARTGET_STOCK").apply(...)
        :param group: (group) The group of 50 most correlated stocks
        :return: (List[str]) returns a list of highest correlated quadruple
        """
        pass

    def _ranked_correlations(self, df_returns: pd.DataFrame) -> pd.DataFrame:
        """Given a df return it's Spearman correlation matrix
        :param: df_returns (pd.DataFrame): The input needs to be in percentage based returns
        :return: pd.DataFrame
        """
        df_corr = df_returns.corr("spearman")
        return df_corr

    def _ranked_pct(self, df: pd.DataFrame):
        """Calculate the rank of a given dataframe and then convert it to percentage based
        :param: df (pd.DataFrame)
        :return: pd.DataFrame
        """
        df_returns = self._returns(df).rank(pct=True)
        return df_returns

    def _top_50_correlations(self, df_corr: pd.DataFrame) -> pd.DataFrame:
        """For correlation matrix return the top 50 correlations
        :param: df_corr (pd.DataFrame): correlation matrix
        :return: pd.DataFrame
        """
        # Filter self correlated and self correlated stocks
        df_corr_unstacked = df_corr[df_corr < 1].unstack().sort_values(ascending=False)
        df_corr_unstacked = df_corr_unstacked.reset_index().dropna()
        df_corr_unstacked.columns = "TARGET_STOCK", "STOCK_PAIR", "CORRELATION"
        # Note pandas was chosen here, but many ways lead to tome
        df_corr_top50 = df_corr_unstacked.groupby("TARGET_STOCK").head(50)
        return df_corr_top50.sort_values(['TARGET_STOCK', 'CORRELATION'])

    def _get_quadruples(self, df_corr_top50: pd.DataFrame) -> pd.DataFrame:
        """
        :param df_corr_top50: ([pd.DataFrame])the top50 correlated stocks for a list of given target stocks
        :return: (pd.Series) returns list of partner quadruples for a target stock
        """
        qf = df_corr_top50.groupby('TARGET_STOCK').apply(self._find_partners_for_target_stock)
        return qf
