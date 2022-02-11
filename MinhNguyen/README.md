# Hudson & Thames - March 2021 Application

This document provides additional context for the skillset submission that implements the Partner Selection methods described in Statistical Arbitrage with Vine Copulas. by Stübinger, Johannes; Mangold, Benedikt; Krauss, Christopher.

## Way of Work

My first step was to read through the paper and get a general overview of the strategy. This also entailed reviewing any concepts I was not familiar with or needed a refresher on.

After getting a general sense of the strategy, I then focused on the chapter regarding *Partner Selection* as that is the primary objective of the challenge. 

The approach I used to implement the required functionality loosely follows:

1. Decompose into smaller problems
2. For each sub problem, experiment and iterate on implementation in Jupyter Notebook
3. Once satisfied, migrate solution into external classes/modules
4. Refactor and/or optimize as necessary once functionality has been integrated
5. Repeat steps 2 - 4 until finished 

## Design Choices

- Data transformations and pre-selection procedure is performed in `PartnerSelector.__init__()` since the results are used across selection methods
- Arbitrary number of dimensions is supported
- Measures of associations calculated for each combination are cached internally so that it can be made available for analysis
- `PartnerSelector.get_partners()` is the main access point and supports specifying selection method as well as a filter to specify a subset of the data to compute partners for
- Vectorized operations were used where possible for efficiency
- The extended Spearman's approach was extracted into a separate class. I wasn't clear on how to use the results of the estimators so for this implementation, I took the average of all 3 estimators as the final measure.
- A strategy pattern to implement the various selection methods could have been used if we wanted to make this system more extensible. It didn't seem necessary at this time.

## Learnings

- Using vectorized operations and being cognizant of unnecessary memory allocations was critical to performance
- Exposure to new (or forgotten) math concepts