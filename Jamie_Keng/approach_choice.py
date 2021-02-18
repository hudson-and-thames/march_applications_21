import pandas as pd
import numpy as np
"""
Class approach_choice offers four different approaches for users to choose from in the partner selection phase.
"""
class approach_choice:
    
    def __init__(self, return_df):
        """
        Setting up class attribute.
        """
        self.return_df = return_df
        
    def euclidean_distance(self, quadruple):
        """
        Calculate the Euclidean distance between a point representing ranks of stocks and 
        the diagonal line starting from point (0,0,0,0) to point (1,1,1,1) in four-dimensional space.

        Parameters 
        ----------
        array : numpy array

        Returns
        -------
        distance : float
            The calculated sum of distance from each points to the four-dimensional diagonal line.   
        """
        distance = np.linalg.norm( quadruple - quadruple@np.array([1,1,1,1])/4 * np.array([1,1,1,1]) )
        return distance
    
    def traditional(self, final_pair_dict, df_corr):
        """
        Parameters
        ---------
        final_pair_dict : dictionary
            The dictionary with quadruple candidates.(generated from basic_processing phase.)
        
        df_corr : pandas dataframe
            The Spearman correlation matrix computed by a built-in pandas method.
        
        Returns
        -------
        pair_corr_val_dict : dictionary
            
            keys : target stocks 
            values : calculated sum of pairwise Spearman's rho for each quadruple candidate.
        """
        pair_corr_val_dict = {}

        for target_stock in final_pair_dict:
            pair_corr_val_dict[target_stock] = {}

            for pair_num in range(0,len(final_pair_dict[target_stock])):

                # Indexes of target stock + 3 partner stocks on the correlation matrix, df_corr
                indexes = list(list(final_pair_dict[target_stock])[pair_num])

                # Sum up all the correlation values and subtract 4 and divided by 2 
                sum_corr = (df_corr.iloc[indexes,indexes].sum().sum() -4) / 2

                pair_corr_val_dict[target_stock][tuple(indexes)] = sum_corr
    
        return pair_corr_val_dict
              
    def extended(self, final_pair_dict, df_corr):
        """
        Parameters
        ----------
        final_pair_dict : dictionary
            The dictionary with quadruple candidates.(generated from basic_processing phase.)
        
        df_corr : pandas dataframe
            The spearman correlation matrix computed by a built-in pandas method.
         
        Returns
        -------
        pair_rho_val_dict: dictionary
            A dictionary containing multivariate version of Spearman's rho for each quadruple. 
            (The algorithm is based on the empirical expression for multivariate Spearman's correlation.)
            
            keys : target stocks 
            values : multivariate Spearman's rho for each quadruple candidate.
        """
        pair_rho_val_dict = {}

        for target_stock in final_pair_dict:
            pair_rho_val_dict[target_stock] = {}

            for pair_num in range(0,len(final_pair_dict[target_stock])):

                # Indexes of target stock + 3 partner stocks on the dataframe, sp500
                indexes = list(list(final_pair_dict[target_stock])[pair_num])

                ### Calculate Multivariate Spearman's rho
                # Normalized ranks of each stocks
                length = (len(self.return_df)+1)
                first = self.return_df.iloc[:,indexes[0]].rank()/ length 
                sec = self.return_df.iloc[:,indexes[1]].rank()/ length 
                third = self.return_df.iloc[:,indexes[2]].rank()/ length 
                fourth = self.return_df.iloc[:,indexes[3]].rank()/ length 

                h = (4 + 1)/(2**4 - 4-1) * 2**4/ len(self.return_df)
                sum_ranks_prod = np.multiply(np.multiply(np.multiply( first, sec), third), fourth).sum() 

                mulSpearman = h* sum_ranks_prod -1

                pair_rho_val_dict[target_stock][tuple(indexes)] = mulSpearman
    
        return pair_rho_val_dict 
    
    def geometric(self, final_pair_dict, df_corr):
        """
        Parameters
        ----------
        final_pair_dict : dictionary
            The dictionary with quadruple candidates.(generated from basic_processing phase.)
        
        df_corr : pandas dataframe
            The spearman correlation matrix computed by a built-in pandas method.
        
        Returns
        -------
        pair_dist_val_dict : dictionary
            A dictionary containing sum of Euclidean distance between points representing ranks of stocks and 
            the diagonal line starting from point (0,0,0,0) to point (1,1,1,1) in four-dimensional space.
            
            keys : target stocks
            values : Euclidean distance summation of each quadruple candidates. 
            
        """  
        pair_dist_val_dict = {}

        for target_stock in final_pair_dict:
            pair_dist_val_dict[target_stock] = {}

            for pair_num in range(0,len(final_pair_dict[target_stock])):

                # Indexes of target stock + 3 partner stocks on the dataset, sp500
                indexes = list(list(final_pair_dict[target_stock])[pair_num])


                # Normalized ranks of each stocks
                length = (len(self.return_df)+1)
                a = self.return_df.iloc[:,indexes[0]].rank()/ length 
                b = self.return_df.iloc[:,indexes[1]].rank()/ length 
                c = self.return_df.iloc[:,indexes[2]].rank()/ length 
                d = self.return_df.iloc[:,indexes[3]].rank()/ length 

                # A created list to store distance values
                lst_dist=[]
                for j in range(0, len(self.return_df)):
                    array = np.array([a[j], b[j], c[j], d[j]])

                    distance = self.euclidean_distance(array)
                    lst_dist.append(distance)

                # Sum of distance of a quadruple
                sum_dist = sum(lst_dist)

                # Store sum of distance to pair dictionary
                pair_dist_val_dict[target_stock][tuple(indexes)] = sum_dist
           
        return pair_dist_val_dict
    
    def extremal(self, final_pair_dict, df_corr):
        """
        Parameters
        ----------
        final_pair_dict : dictionary
            The dictionary with quadruple candidates.(generated from basic_processing phase.)
        
        df_corr : pandas dataframe
            The spearman correlation matrix computed by a built-in pandas method.
         
        Returns
        -------
        pair_derivative_val_dict: dictionary
            A dictionary including chi square statistic based on algorithm derived from polynomial copula function.
            
            keys : target stocks
            values : chi square statistic for each quadruple candidate.
        """
        
        pair_derivative_val_dict = {}

        for target_stock in final_pair_dict:
            pair_derivative_val_dict[target_stock] = {}

            for pair_num in range(0,len(final_pair_dict[target_stock])):

                # Indexes of target stock + 3 partner stocks on the dataset, sp500
                indexes = list(list(final_pair_dict[target_stock])[pair_num])

                # Normalized ranks of each stocks
                length = (len(self.return_df)+1)
                a = self.return_df.iloc[:,indexes[0]].rank()/ length 
                b = self.return_df.iloc[:,indexes[1]].rank()/ length 
                c = self.return_df.iloc[:,indexes[2]].rank()/ length 
                d = self.return_df.iloc[:,indexes[3]].rank()/ length 

                # A created list to store distance values
                ### Implementation of Proposition 3 on page 17
                density_derivative= []
                for j in range(0, len(self.return_df)):

                    ans= 1 - 2*(a[j]+b[j]+c[j]+d[j]) + 4*(a[j]*b[j]+a[j]*c[j]+a[j]*d[j]+b[j]*c[j]+b[j]*d[j]+c[j]*d[j])- \
                      8*(a[j]*b[j]*c[j] + a[j]*b[j]*d[j] + a[j]*c[j]*d[j] + b[j]*c[j]*d[j]) + \
                      16*(a[j]*b[j]*c[j]*d[j])

                    density_derivative.append(ans)

                # Mean of the values of density derivative
                statistic= sum(density_derivative)/len(self.return_df)

                # Store sum of distance to pair dictionary
                pair_derivative_val_dict[target_stock][tuple(indexes)] = statistic

        return pair_derivative_val_dict
         
    def final_quadruple(self, quadruple_dict, approach):
        """
        Parameters
        ----------
        quadruple_dict : dictionary
        approach : string
            The name of the apporach a user chooses to conduct partner selection. 
            (valid inputs: "traditional", "extended", "geometric", "extremal")
            
        Returns
        -------
        result_pair_dict : dictionary
            A dictionary with one final quadruple for each target stock.
            
            keys : target stocks
            values : The one and only quadruple selected by any approach's mechanism for each target stock.
        """
        result_pair_dict = {}
        
        if (approach == "traditional")|(approach == "extended")|(approach == "extremal"):
            for pair in quadruple_dict:
            
                # Get the key(the quadruple) that yield the maximum value of the desired measure. 
                keymax = max(quadruple_dict[pair], key= quadruple_dict[pair].get)

                result_pair_dict[pair] = keymax 
            
        if (approach == "geometric"):
            for pair in quadruple_dict:
                
                # Get the key(the quadruple) that yield the minimum Euclidean distance.
                keymax = min(quadruple_dict[pair], key= quadruple_dict[pair].get)

                result_pair_dict[pair] = keymax 
     
        return result_pair_dict
        
