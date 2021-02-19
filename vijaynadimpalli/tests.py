import pandas as pd
import unittest

from statsmodels.distributions.empirical_distribution import ECDF
from partner_selection import PartnerSelection
from ps_utils import get_sum_correlations, multivariate_rho, diagonal_measure, extremal_measure, get_co_variance_matrix
from utils_multiprocess import run_traditional_correlation_calcs, run_extended_correlation_calcs, \
                                run_diagonal_measure_calcs, run_extremal_measure_calcs


class PartnerSelectionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.quadruple = ['A', 'AAL', 'AAP', 'AAPL']
        df = pd.read_csv('./data/data.csv', parse_dates=True, index_col='Date').dropna()[cls.quadruple]
        df = df['2016']
        cls.ps = PartnerSelection(df)

        cls.u = cls.ps.returns.copy()
        for column in cls.ps.returns.columns:
            ecdf = ECDF(cls.ps.returns.loc[:, column])
            cls.u[column] = ecdf(cls.ps.returns.loc[:, column])

        cls.co_variance_matrix = get_co_variance_matrix()

    def test_sum_correlations(self):
        self.assertEqual(round(get_sum_correlations(self.ps.correlation_matrix, self.quadruple), 4), 1.9678)

    def test_multivariate_rho(self):
        self.assertEqual(round(multivariate_rho(self.u[self.quadruple]), 4), 0.3114)

    def test_diagonal_measure(self):
        self.assertEqual(round(diagonal_measure(self.ps.ranked_returns[self.quadruple]), 4), 91.9374)

    def test_extremal_measure(self):
        self.assertEqual(round(extremal_measure(self.ps.ranked_returns[self.quadruple], self.co_variance_matrix), 4), 108.5128)

    def test_run_traditional_correlation_calcs(self):
        self.assertEqual(round(run_traditional_correlation_calcs(
            self.ps.correlation_matrix, [self.quadruple, self.quadruple], num_threads=1)['result'], 4), 1.9678)

    def test_run_extended_correlation_calcs(self):
        self.assertEqual(round(run_extended_correlation_calcs(
            self.u, [self.quadruple, self.quadruple], num_threads=1)['result'], 4), 0.3114)

    def test_run_diagonal_measure_calcs(self):
        self.assertEqual(round(run_diagonal_measure_calcs(
            self.ps.ranked_returns, [self.quadruple, self.quadruple], num_threads=1)['result'], 4), 91.9374)

    def test_run_extremal_measure_calcs(self):
        self.assertEqual(round(run_extremal_measure_calcs(
            self.ps.ranked_returns, [self.quadruple, self.quadruple], self.co_variance_matrix, num_threads=1)['result'], 4), 108.5128)


if __name__ == '__main__':
    unittest.main()
