import heapq
import numpy as np
import pandas as pd
from itertools import permutations

"""
Class basic_processing provides a function to filter out potential highly correlated stocks 
for each stock in the dataset. Users have the option to set the number of potential partner stocks they want. 
"""

class basic_processing:
    
    def partner_selection(self, ret_df, partner_num):
        """
        Parameters
        ----------
        ret_df : pandas dataframe
            The return dataset generated in the preliminary step.
        
        partner_num : int
            The number of potential partner stocks that a user want for a target stock.
            (The larger the number is, the more computaionally expensive it is.)
        
        Returns
        -------
        partner_dict : dictionary   
            A dictionary containing all possible partner-stock candidates for each target stock.
        
        Reference
        ---------
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
        """
        # Obtain pairwise Spearsman's rho in thw dataset. 
        df_corr = ret_df.corr(method = "spearman")

        # Build an empty dictionary to store a number of top partners for each target stock.
        partner_dict = {target: None for target in range(len(df_corr))}

        for target_stock in range(len(df_corr)):
            lst = []
            for stock in range(len(df_corr)):
                lst.append(df_corr.iloc[target_stock][stock])
                # Find the list of the largest 50 stock's indexes
                top_partner = heapq.nlargest(partner_num, range(len(lst)), key=lst.__getitem__)

            partner_dict[target_stock]= top_partner

        return df_corr, partner_dict    

    def potential_quadruples(self, partner_dict):
        """
        Parameters
        ----------
        partner_dict : dictionary
            A dictionary containing potential partner stocks for each stock in the dataset.
            
            keys: target stocks.
            values: potential partner stocks.    

        Returns
        -------
        final_pair_dict : A dictionary of dictionaries
            A dictionary including all filtered potential quadruples for each stock in the dataset.
            
            keys: target stocks
            values: potential quadruples
        """
        
        # Potential quadruples formation
        pairs_dict= {}
        for target_stock in partner_dict: 
            pairs_dict[target_stock] = list(permutations(partner_dict[target_stock], 4))
        
        # Remove quadruples without target stocks.
        selected_pairs_dict = {}
        for target_stock in range(len(pairs_dict)):
            selected_pairs_dict[target_stock] = [item for item in pairs_dict[target_stock] if item[0] == target_stock or \
                                                item[1] == target_stock or \
                                                item[2] == target_stock or \
                                                item[3] == target_stock] 
            
        # Remove duplicated quadruples.
        final_pair_dict ={}

        for target_stock in range(len(selected_pairs_dict)):
            final_pair_dict[target_stock] = set(tuple(sorted(x)) for x in selected_pairs_dict[target_stock])    
       
        
        return final_pair_dict
    
