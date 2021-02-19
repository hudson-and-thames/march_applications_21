import unittest
from data_handler import ImportData


class CustomTests(unittest.TestCase):

    def test_data_handler_sp500(self):
        self.assertEqual(len(ImportData.get_list_sp500()),505)

    def test_data_handler_download(self):
        self.assertEqual(len(ImportData.download_data('GOOG','2020-01-01','2020-01-10')),7)


if __name__ == '__main__':
    unittest.main()