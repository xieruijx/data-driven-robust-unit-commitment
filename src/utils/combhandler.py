import numpy as np
import scipy.special
import math

class CombHandler(object):
    """
    CombHandler class to calculate some combinatorial numbers
    """

    @staticmethod
    def binomial_term(n, k, a):
        """
        Calculate the binomial term $C_n^k a^k (1 - a)^{n-k}$
        """
        binomial_coeff = scipy.special.comb(n, k, exact=False)  # Use exact=False for floating point output
        term = binomial_coeff * (a ** k) * ((1 - a) ** (n - k))
        return term
    
    @staticmethod
    def binomial_term_gamma(n, k, a):
        """
        An alternative method to calculate the binomial term $C_n^k a^k (1 - a)^{n-k}$
        """
        return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(a) + (n - k) * math.log(1 - a))
    
    @staticmethod
    def get_rank(n, epsilon, delta):
        """
        Get rank
        n: numbers
        epsilon: chance constraint parameter
        delta: probability guarantee parameter
        """
        sum = 1
        for i in range(n):
            sum -= CombHandler().binomial_term(n, n - i, 1 - epsilon)
            if sum < 1 - delta:
                return n - i
        return 0
    
    @staticmethod
    def get_rank_total(n, epsilon):
        """
        Get probability guarantee parameters
        n: numbers
        epsilon: chance constraint parameter
        """
        output = np.ones((n + 1,))
        for i in range(n):
            output[n - i - 1] = output[n - i] - CombHandler().binomial_term(n, n - i, 1 - epsilon)
        return output
