import numpy as np
import scipy.special

class SpearmanExt():
    """This is an implementation of a generalized Spearman's rho for higher dimensionality as described in 
    Schmid, F., Schmidt, R., 2007. Multivariate extensions of Spearman’s rho and related statistics.
    """
    def __init__(self, data):
        """Constructor

        :param data: (pd.DataFrame): quantile data of shape (i,j) where i is the number of dimensions and j is number of samples
        """
        # Pre calculate values used across estimators
        self.d = d = len(data)
        self.n = len(data[0])
        self._hd = (d+1) / (2**d - d - 1)
        self._mult = (2**4)/self.n
        self.u = data

    @property
    def r1(self):
        """Calculates and returns the first estimator"""
        inner = 1 - self.u
        inner_prod = np.prod(inner, axis=0)
        outer = inner_prod.sum() * self._mult - 1
        return self._hd * outer

    @property
    def r2(self):
        """Calculates and returns the second estimator"""
        inner_prod = np.prod(self.u, axis=0)
        outer = inner_prod.sum() * self._mult - 1
        return self._hd * outer

    @property
    def r3(self):
        """Calculates and returns the third estimator"""
        right = 0
        u = self.u
        for l in range(1, self.d):
            for k in range(l):
                a = 1 - u[k]
                b = 1 - u[l]
                right += np.dot(a,b)
        m = 12 / (self.n * scipy.special.comb(self.d, 2, exact=True))
        return (m * right) - 3
