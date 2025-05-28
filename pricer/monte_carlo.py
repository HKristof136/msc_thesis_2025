import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from tqdm import tqdm
from pricer.config_base import MonteCarloConfig

class MonteCarloHeston:
    payoff_function = lambda self, x: x

    def __init__(self, config: MonteCarloConfig):
        self.config = config
        self.n = config.n
        self.m = config.m
        self.interest_rate = config.interest_rate
        self.corr = config.corr
        self.kappa = config.kappa
        self.variance_theta = config.variance_theta
        self.sigma = config.sigma
        self.barrier = config.barrier
        self.strike = config.strike

    def price(self, S0, V0, T):
        pass

    def generate_paths(self, s, v, t):
        dt = t / self.m
        paths = np.zeros((2, self.n, self.m + 1))
        paths[0, :, 0] = v
        paths[1, :, 0] = s

        z1 = np.random.normal(size=(self.n, self.m + 1))
        z2 = np.random.normal(size=(self.n, self.m + 1))
        z2 = self.corr * z1 + np.sqrt(1 - self.corr**2) * z2
        for i in tqdm(range(1, self.m + 1), disable=True):
            paths[0, :, i] = paths[0, :, i - 1] + self.kappa * (self.variance_theta - paths[0, :, i - 1]) * dt + self.sigma * np.sqrt(paths[0, :, i - 1]) * np.sqrt(dt) * z1[:, i] + 0.25 * self.sigma ** 2 * dt * (z1[:, i] ** 2 - 1)
            paths[0, :, i] = np.maximum(0.0, paths[0, :, i])

            paths[1, :, i] = paths[1, :, i - 1] * np.exp((self.interest_rate - 0.5 * paths[0, :, i - 1]) * dt + np.sqrt(paths[0, :, i - 1] * dt) * z2[:, i])
        return paths


class MonteCarloHestonCall(MonteCarloHeston):
    payoff_function = lambda self, x, strike: np.maximum(0.0, x - strike)

    def price(self, S0, V0, T):
        paths = self.generate_paths(S0, V0, T)
        return np.mean(self.payoff_function(paths[1, :, -1], self.strike) * np.exp(-self.interest_rate * T))

class MonteCarloHestonBarrierUpAndOutCall(MonteCarloHeston):
    payoff_function = lambda self, x, strike: np.maximum(0.0, x - strike)

    def price(self, S0, V0, T):
        paths = self.generate_paths(S0, V0, T)
        barrier_breached = np.any(paths[1, :, :] >= self.barrier, axis=1)
        paths[1, :, -1] = np.where(~barrier_breached, self.payoff_function(paths[1, :, -1], self.strike), 0.0)
        return np.mean(paths[1, :, -1]) * np.exp(-self.interest_rate * T)

if __name__ == "__main__":
    config = MonteCarloConfig(
        n=5000,
        m=1000,
        interest_rate=0.05,
        corr=-0.5,
        kappa=2.0,
        variance_theta=0.04,
        sigma=0.3,
        barrier=1.2
    )

    mc_heston = MonteCarloHestonCall(config)
    for s in [1.0, 1.1, 1.19]:
        price = mc_heston.price(S0=s, V0=0.04, T=1)
        print(f"Option Price: {price}")