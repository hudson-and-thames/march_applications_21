from statsmodels.distributions.empirical_distribution import ECDF
import scipy.special
import numpy as np

class SpearmanExt():
    """
    This is an implementation of a generalized Spearman's rho for higher dimensionality as described in 
    Schmid, F., Schmidt, R., 2007. Multivariate extensions of Spearman’s rho and related statistics.
    """
    def __init__(self, rank_data):
        self.data = rank_data
        self.d = d = len(rank_data)
        self.n = len(rank_data[0])
        self._hd = (d+1) / (2**d - d - 1)
        self.u = np.array([np.array(ECDF(data)(data)) for data in rank_data])

    @property
    def r1(self):
        inner = 1 - self.u
        inner_prod = np.prod(inner, axis=0)
        outer = inner_prod.sum() * ((2**4)/self.n) - 1
        return self._hd * outer

    @property
    def r2(self):
        inner_prod = np.prod(self.u, axis=0)
        outer = inner_prod.sum() * ((2**4)/self.n) - 1
        return self._hd * outer

    @property
    def r3(self):
        right = 0
        u = self.u
        for l in range(1, self.d):
            for k in range(l):
                a = 1 - u[k]
                b = 1 - u[l]
                right += np.dot(a,b)
        m = 12 / (self.n * scipy.special.comb(self.d, 2, exact=True))
        return (m * right) - 3
