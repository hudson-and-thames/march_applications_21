import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.decomposition import PCA
import numpy as np

class visualization:  
    def __init__(self, return_df):
        
        self.return_df = return_df
     
    def scatterplot_matrix(self, approach_result, target_stock_index):
        """
        Parameters
        ----------        
        approach_result : dictionary
            The result dictionary of any approach.
            (In the code demonstration, the argument could be result_trad, result_extended, result_geometric, result_extremal.)
        
        target_stock_index : int
            The corresponding index of a target stock.
                    
        Reference
        ---------
        http://rasbt.github.io/mlxtend/user_guide/plotting/scatterplotmatrix/

        """
        sns.set_theme(style="darkgrid")

        df = self.return_df.iloc[:,list(approach_result[target_stock_index])]
        sns.pairplot(df)
    
    def countfrequency(self, my_list): 

        # An empty dictionary  
        freq = {} 
        for items in my_list: 
            freq[items] = my_list.count(items) -1 

        return freq
    
    def freq_partner_stocks(self, approach_result, stock_ind_dict, num =10):
        """
        Parameters
        ----------
        approach_result : dictionary
            The result dictionary of any approach.
        
        stock_ind_dict : dictionary
            A dictionary that has each stock's corresponding index in the dataframe.
            
                keys : indexes 
                values : stock tickers
        
        num : int 
            The number of the top most commonly picked partner stocks.   
        """
        
        # Transform each value's type from a tuple to a list.
        for key in approach_result:
            approach_result[key] = list(approach_result[key])

        # Combine all the values in the result dictionary
        comb_result = sum(list(approach_result.values()), [])
        
        combined_dict = self.CountFrequency(comb_result)
        
        # Eliminate stocks that only appear once(being a target stock in a quadruple.)
        for key in list(combined_dict):
            if combined_dict[key] ==0:
                combined_dict.pop(key, None)
        
        # Get the most commonly picked partner stocks.
        d =Counter(combined_dict)
        freq_partner =d.most_common(num)
        
        # Transform freq_parnter's type back to a dictionary
        dict_ = {}
        for k, v in freq_partner:
            dict_[k] = v
        
        # Switch the stock indexes with stock tickers, so that user get to know exactly what the most commonly picked 
        # partner stocks are.
        new_dict= {}
        for key in list(dict_):
            new_dict[stock_ind_dict[key]] = dict_[key]
        
        # Plot the freqency histogram
        plt.figure(figsize=(8,5))
        data = new_dict

        plt.bar(data.keys(), data.values(), color ='orange')
        plt.xlabel("most commonly picked partner stocks")
        plt.ylabel("frequency")
        plt.show()
        
    def pca_plot(self, approach_result, target_stock_index, choice):
        """
        Plot the first and second most important primary components. 
        The notion here is that if stocks in a quadruple are closely correlated,
        then the values of their first and second primary components should be close to one another 
        on a 2-D scatter plot as well.
        
        Parameters
        ----------
        approach_result : dictionary
            The result dictionary of any approach.
        
        target_stock_index : int
            The index of the target stock in a quadruple the user want to conduct PCA. 
        
        choice: string
            A user can choose either show the PCA result of all quadruples or show the PCA result of a single quadruple.
            (valid inputs : "single", "all") 
        """     
        pca = PCA(n_components =2)

        # Need dataframe.transpose() to fit into PCA method.
        return_PCA = pca.fit_transform(self.return_df.transpose())
        
        # Assign a color(RGB) to each final quadruple
        colors = {quadruple: np.random.rand(3) for quadruple in approach_result}
        
        plt.figure(dpi=100)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        
        if choice == "all":
            for target_stock in list(approach_result):
                for j in range(0,4):
                    indexes = list(approach_result[target_stock])[j]

                    plt.scatter(return_PCA[indexes, 0], return_PCA[indexes, 1], s= 8, color = colors[target_stock])

            plt.xlabel("PCA_1")        
            plt.ylabel("PCA_2")

            plt.show()
        
        if choice == "single":

            target_stock = target_stock_index

            for j in range(0,4):
                indexes = list(approach_result[target_stock])[j]

                plt.scatter(return_PCA[indexes, 0], return_PCA[indexes, 1], s =20, color = colors[target_stock])

            plt.xlabel("PCA_1")        
            plt.ylabel("PCA_2")

            plt.show()
        
