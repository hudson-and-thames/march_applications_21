import unittest
import pandas as pd
from data_handler import ImportData, DataPreprocess
from partner_selector import SelectPartner


class CustomTests(unittest.TestCase):
    def setUp(self):
        self.sp500_list = ImportData().get_list_sp500()
        self.sp500_data = pd.read_csv("data/stocks.csv").set_index('Date')
        self.rank_df = DataPreprocess().get_rank(self.sp500_data)
        self.top_rank_df, self.top_rank_corr_df = DataPreprocess().most_correlated_stocks(self.rank_df, 10)
        self.SP = SelectPartner(self.top_rank_df)

    def test_data_handler_sp500(self):
        self.assertEqual(len(self.sp500_list), 505)

    def test_get_top_rank(self):
        self.assertEqual(self.top_rank_df.shape[1], self.top_rank_corr_df.shape[1])

    def test_traditional_approach(self):
        self.assertEqual(self.SP._traditional(self.top_rank_df.columns[:4]), 10.628642678680519)

    def test_extended_approach(self):
        self.assertEqual(self.SP._extended(self.top_rank_df.columns[:4], 'first'), 0.985608885910356)

    def test_geometric_approach(self):
        self.assertEqual(self.SP._geometric(self.top_rank_df.columns[:4]), -274.83333333333337)


if __name__ == '__main__':
    unittest.main()
