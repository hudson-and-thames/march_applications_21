# Pair Selection
In this repo, I implemented 4 different approaches for pair selection which were introduced in the paper(Statistical Arbitrage with Vine Copulas, 2016).
For the usage of the module, please see the jupyter notebook

### Files 
- data : (folder) It contains csv file for examples in the jupyter notebook
- data_handler.py : It contains a module for importing data and preprocessing it
- partner_selector.py : It contains a module for selecting partners in 4 different ways
- Notebook Submission.ipynb : This jupyter notebook contains some examples of using the modules I created

### Requirements  
- pandas
- yfinance
- numpy
- seaborn
- matplotlib
- itertools
- statsmodels
- tqdm(not really neccessary)


### 1. Way of Work
First, I tried to read through the paper(Statistical Arbitrage with Vine Copulas, 2016) and get general ideas from it. 
Then, for the section 3.1.1: Partner Selection, I read all papers related to all of the four procedures and designed the right class for implementation.
For the first three of the approaches, it was very clear on implementation so I followed the equations from the paper to implement it. Then although I could have made a better improvement in speed efficiency, I tried other ways to speed up the code while still making it straight forward.  
However, I couldn't finish working on the last part of the partner selection which is extremal approach as it was my first time seeing those concepts from paper. Although I couldn't fully implement the procedure, I tried to understand the concept of it through reading the paper(A multivariate linear rank test of independence based on a multi parametric copula with cubic sections, 2015).

### 2. Design Choices
I thought I should split the module into two different files. One is for handling data which I named it after and the other is for partner selection which I also named it after. 
Besides the data_handler.py, in partner_selector.py, I created a single class for end-user to use because I thought it would be simpler than calling different modules for different partner selection approaches. In the class, as rank dataframe and rank correlation dataframe are frequently used, I assigned them as attributes in __init__ method. Also, for each of the approaches, I made separate hidden methods for each one of them.
For users, I made a single method get_partner which takes input of method for partner selection. Therefore a user can call this method only to use different types of partner selection.

### 3. Learnings
As I was not very aware of the concept of copula and extremal approach in the section 3.1.1, this was a great opportunity for me to dive into those topics as well. I not only learned just implementing the equations from the paper but also optimizing and simplifying the code for other people to use it in the future. 
