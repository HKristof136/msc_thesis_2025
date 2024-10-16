import numpy as np

# TODO: Implement MC class
class MonteCarlo:
    def __init__(self, underlier_price, expiry, interest_rate, volatility, n, m, path_dependent=False):
        self.underlier = underlier_price
        self.term = expiry
        self.r = interest_rate
        self.sigma = volatility
        self.n = n
        self.m = m

        self.path_dependent = path_dependent