import utility

import numpy as np
import pandas as pd

from itertools import combinations
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.special import comb
from sympy.utilities.lambdify import lambdify

from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import CheckboxGroup, CustomJS, ColumnDataSource, Panel, Tabs
from bokeh.models.widgets import CheckboxGroup
from bokeh.layouts import row, column
from bokeh.palettes import Spectral10

class PartnerSelection:
    """Implementation of 4 partner selection methods described in Stübinger et al. (2016) section 3.1.1.

    Parameters
    ----------
    daily_return : Pandas DataFrame
        DataFrame contains adjusted close with tickers as column names and datetime as indexes
    num_of_partner : int
        Number of partner the target stock needed. if num_of_partner = n, total stocks as a cohort will be n + 1
    num_take_account : int
        Number of potential partner stocks take into account.

    Returns
    -------
    PartnerSelection object
    """

    def __init__(self, daily_return, num_of_partner, num_take_account):
        self.daily_return = daily_return
        self.ticker_list = list(daily_return.columns)
        self.num_of_partner = num_of_partner
        self.num_take_account = num_take_account

        self.potential_partner_dict = {}
        self.potential_partner_combinations_dict = {}
        self.approach_result_dict = {}


        self.d = num_of_partner + 1  # Total stocks as a cohort
        self.h_d = (self.d + 1)/(2**(self.d) - self.d - 1)  # Scalar needed for implementation of extended_approach approach
        
        self.ranked_daily_return = daily_return.rank(axis = 0, pct = True, ascending = True, na_option = 'keep')  #NaN will be keep without affecting the ranking
        self.pairwise_spearman_corr = self.ranked_daily_return.corr()

    def add_target_stock(self, target_stock):
        """Before calculating the partners of the target stock, you need to add it to this object, and determine it's potential partner combinations.

        Parameters
        ----------
        target_stock : str
            Ticker of target stock

        Returns
        -------
        None
        """

        if target_stock not in self.ticker_list:
            raise KeyError("Target stock dose not exist in the object's data.")

        # The paper didn't clearly describe how to determine the top N most highly correlated stocks.
        # Here I will take into account top N stocks with the highest Spearman's ρ
        # First element of the below series will always be the target stock, so we need to skeep it.
        potential_partner_stocks = self.pairwise_spearman_corr[target_stock].sort_values(ascending = False)[1:self.num_take_account + 1]
        potential_partner_stocks = list(potential_partner_stocks.index)
        potential_partner_combinations = combinations(potential_partner_stocks, self.num_of_partner)
        potential_partner_combinations = list(potential_partner_combinations)

        # Save result to this object's dictionary
        self.potential_partner_dict[target_stock] = potential_partner_stocks
        self.potential_partner_combinations_dict[target_stock] = potential_partner_combinations

    def target_stock_added(self):
        """Return added stocks' tickers """

        return list(self.potential_partner_dict.keys())

    def traditional_approach(self, target_stock):
        """Implementation of traditional approach described in Stübinger et al. (2016) section 3.1.1.

        Method:
        1. Calculate the sum of all pairwise correlations for all possible combinations.
        2. The combination with the largest sum of pairwise correlations is considered as the result.

        Parameters
        ----------
        target_stock : str
            Ticker of target stock

        Returns
        -------
        List of selected stocks' tickers. The last element of the list will be the target stock's ticker.
        """

        if target_stock not in self.potential_partner_combinations_dict:
            raise KeyError("Please add the target stock to the object first.")
        else:
            potential_partner_combinations = self.potential_partner_combinations_dict[target_stock]

        max_sum = float("-inf")
        max_sum_combination = []
        
        # Calculate sums of all pairwise correlations one by one, and store the combination with the largest sum
        for c in potential_partner_combinations:
            combination = list(c)
            combination.append(target_stock)
            
            sum_of_corr = self.pairwise_spearman_corr[combination].loc[combination].sum().sum()
            if sum_of_corr > max_sum:
                max_sum = sum_of_corr
                max_sum_combination = combination

        # Save result to this object's dictionary
        if target_stock in self.approach_result_dict:
            self.approach_result_dict[target_stock]["traditional approach"] = max_sum_combination
        else:
            self.approach_result_dict[target_stock] = {"traditional approach" : max_sum_combination}

        return max_sum_combination

    def extended_approach(self, target_stock, type_of_estimator = 3):
        """Implementation of extended approach described in Stübinger et al. (2016) section 3.1.1.

        Method:
        1. Calculate multivariate conditional versions of Spearman’s rho (3 kinds of type) described in Schmid and Schmidt (2007) for all possible combinations.
        2. The combination with the largest rho is considered as the result.

        Parameters
        ----------
        target_stock : str
            Ticker of target stock
        type_of_estimator : {1, 2, 3}, default 3
            Type of Spearman’s rho

        Returns
        -------
        List of selected stocks' tickers. The last element of the list will be the target stock's ticker.
        """

        if target_stock not in self.potential_partner_combinations_dict:
            raise KeyError("Please add the target stock to the object first.")
        else:
            potential_partner_combinations = self.potential_partner_combinations_dict[target_stock]

        # functions of different types of rho
        def rho_func1(combination):
            values  = 1 - self.ranked_daily_return[combination].dropna()  #Drop out NaN to let all retrun series have same number of valid values
            return self.h_d*(-1 + 2**(self.d)/len(values)*(values.prod(axis = 1).sum()))

        def rho_func2(combination):
            values  = self.ranked_daily_return[combination].dropna()
            return self.h_d*(-1 + 2**(self.d)/len(values)*(values.prod(axis = 1).sum()))

        def rho_func3(combination):
            values  = 1 - self.ranked_daily_return[combination].dropna().values.T
            double_sum = (values.cumsum(axis = 0) - values).ravel().dot(values.ravel())  #Calculate double sum in the rho equation using a O(n) runtime algorithm
            return -3 + 12/(len(values.T)*comb(self.d, 2))*double_sum

        # Choose function
        if type_of_estimator == 1:
            rho_func = rho_func1

        elif type_of_estimator == 2:
            rho_func = rho_func2

        elif type_of_estimator == 3:
            rho_func = rho_func3

        else:
            raise TypeError("Type of estimator need to be 1, 2 or 3.")


        max_rho = float("-inf")
        max_rho_combination = []
        
        # Calculate rhos one by one, and store the combination with the largest rho
        for c in potential_partner_combinations:
            combination = list(c)
            combination.append(target_stock)

            rho = rho_func(combination)
            if rho > max_rho:
                max_rho = rho
                max_rho_combination = combination

        # Save result to this object's dictionary
        if target_stock in self.approach_result_dict:
            self.approach_result_dict[target_stock]["extended approach"] = max_rho_combination
        else:
            self.approach_result_dict[target_stock] = {"extended approach" : max_rho_combination}

        return max_rho_combination

    def geometric_approach(self, target_stock):
        """Implementation of geometric approach described in Stübinger et al. (2016) section 3.1.1.

        Method:
        1. Calculate the sum of all perpendicular distances from each row instance of the data to the diagonal vector for all possible combinations.
        2. The combination with the smallest sum is considered as the result.

        Parameters
        ----------
        target_stock : str
            Ticker of target stock

        Returns
        -------
        List of selected stocks' tickers. The last element of the list will be the target stock's ticker.
        """

        if target_stock not in self.potential_partner_combinations_dict:
            raise KeyError("Please add the target stock to the object first.")
        else:
            potential_partner_combinations = self.potential_partner_combinations_dict[target_stock]

        min_sum = float("inf")
        min_sum_combination = []
        
        # Calculate sums one by one, and store the combination with the smallest sum
        for c in potential_partner_combinations:
            combination = list(c)
            combination.append(target_stock)

            values  = self.ranked_daily_return[combination].dropna().values
            sum_of_dis = utility.perpendicular_distance_to_diagonal(values).sum()
            if sum_of_dis < min_sum:
                min_sum = sum_of_dis
                min_sum_combination = combination
                
        # Save result to this object's dictionary
        if target_stock in self.approach_result_dict:
            self.approach_result_dict[target_stock]["geometric approach"] = min_sum_combination
        else:
            self.approach_result_dict[target_stock] = {"geometric approach" : min_sum_combination}

        return min_sum_combination

    def extremal_approach(self, target_stock):
        """Implementation of extremal approach described in Stübinger et al. (2016) section 3.1.1.

        Method:
        1. Calculate the chi square type test statistic described in Mangold (2015) section 3 for all possible combinations.
        2. The combination with the largest test statistic is considered as the result.

        Parameters
        ----------
        target_stock : str
            Ticker of target stock

        Returns
        -------
        List of selected stocks' tickers. The last element of the list will be the target stock's ticker.

        Notes
        -------
        This approach will take more time than others, especially the first time you use it.
        """

        if target_stock not in self.potential_partner_combinations_dict:
            raise KeyError("Please add the target stock to the object first.")
        else:
            potential_partner_combinations = self.potential_partner_combinations_dict[target_stock]

        # Construct symbolic formulas and inverse of covariance matrix needed for implementation
        # if self.d >= 4, the construction will take more than 5 minutes
        # This function runs only when the approach is executed for the first time
        def construct_required_items(self):
            self.c_dot_theta_u, self.u_vector = utility.generate_formula(self.d)
            self.func = lambdify(self.u_vector, self.c_dot_theta_u, 'numpy')
            self.cov_inv = utility.generate_cov_inv(self.c_dot_theta_u, self.u_vector)

        if hasattr(self, 'c_dot_theta_u'):
            pass
        else:
            construct_required_items(self)

        max_T = float("-inf")
        max_T_combination = []
        
        # Calculate test statistics one by one, and store the combination with the largest test statistic
        for c in potential_partner_combinations:
            combination = list(c)
            combination.append(target_stock)

            values  = self.ranked_daily_return[combination].dropna()
            values = values.values*len(values)/(len(values) + 1) 

            Tn = np.array(self.func(*values.T))
            Tn = np.mean(Tn, axis = 1)
            T = Tn.T.dot(self.cov_inv).dot(Tn)*len(values)
            if T > max_T:
                max_T = T
                max_T_combination = combination
                
        # Save result to this object's dictionary
        if target_stock in self.approach_result_dict:
            self.approach_result_dict[target_stock]["extremal approach"] = max_T_combination
        else:
            self.approach_result_dict[target_stock] = {"extremal approach" : max_T_combination}

        return max_T_combination

    def get_result(self):
        """Return a dictionary containing all calculated results."""

        return self.approach_result_dict

    def visualization(self, target_stock):
        """Simple visualization of the result using Bokeh."""

        if target_stock not in self.approach_result_dict:
            raise KeyError("No result for the target_stock")

        else:
            result = self.approach_result_dict[target_stock]

        tabs = []
        for i in result.keys():
            label_name = i
            tickers = result[i]
            cumprod = (self.daily_return[tickers] + 1).cumprod()
            source = ColumnDataSource(data = cumprod)
            
            p = figure(x_axis_type="datetime", title="Data", plot_height=350, plot_width=800)
            p.xgrid.grid_line_color=None
            p.ygrid.grid_line_alpha=0.5
            p.xaxis.axis_label = 'Time'
            p.yaxis.axis_label = 'Total Return'
            lines = []
            for i in range(len(cumprod.columns)):
                lines.append(p.line("Date", cumprod.columns[i], source=source, line_width=2, line_alpha=0.6, line_color = Spectral10[i%10], legend_label = cumprod.columns[i]))
            
            p.legend.location = "top_left"
            
            LABELS = list(cumprod.columns)
            checkbox_group = CheckboxGroup(labels=LABELS)
            checkbox_group.active = list(range(len(LABELS)))
            
            code = """ for (var i = 0; i < lines.length; i++) {
                            lines[i].visible = false;
                            if (cb_obj.active.includes(i)){lines[i].visible = true;}
                        }
                   """
            callback = CustomJS(code = code, args = {'lines': lines})
            checkbox_group.js_on_click(callback)

            layout = row(p, checkbox_group)
            tabs.append(Panel(child = layout, title = label_name))
        

        tabs = Tabs(tabs = tabs)
        show(tabs)