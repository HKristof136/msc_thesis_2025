from scipy.stats import norm
import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Iterable


class BlackScholes(ABC):
    def __init__(self, underlier_price, strike, expiry, interest_rate, volatility):
        self.underlier = underlier_price
        self.strike = strike
        self.term = expiry
        self.r = interest_rate
        self.sigma = volatility

    @abstractmethod
    def price(self):
        pass


class BlackScholesCall(BlackScholes):
    def price(self):
        d1 = (
            np.log(self.underlier / self.strike)
            + (self.r + 0.5 * self.sigma**2) * self.term
        ) / (self.sigma * np.sqrt(self.term))
        d2 = d1 - self.sigma * np.sqrt(self.term)
        return self.underlier * norm.cdf(d1) - self.strike * np.exp(
            -self.r * self.term
        ) * norm.cdf(d2)
    def delta(self):
        d1 = (
            np.log(self.underlier / self.strike)
            + (self.r + 0.5 * self.sigma**2) * self.term
        ) / (self.sigma * np.sqrt(self.term))
        return norm.cdf(d1)
    def gamma(self):
        d1 = (
            np.log(self.underlier / self.strike)
            + (self.r + 0.5 * self.sigma**2) * self.term
        ) / (self.sigma * np.sqrt(self.term))
        return norm.pdf(d1) / (self.underlier * self.sigma * np.sqrt(self.term))
    def vega(self):
        d1 = (
            np.log(self.underlier / self.strike)
            + (self.r + 0.5 * self.sigma**2) * self.term
        ) / (self.sigma * np.sqrt(self.term))
        return self.underlier * norm.pdf(d1) * np.sqrt(self.term)
    def theta(self):
        d1 = (
            np.log(self.underlier / self.strike)
            + (self.r + 0.5 * self.sigma**2) * self.term
        ) / (self.sigma * np.sqrt(self.term))
        d2 = d1 - self.sigma * np.sqrt(self.term)
        return (-self.underlier * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.term)) - self.r * self.strike * np.exp(-self.r * self.term) * norm.cdf(d2))
    def rho(self):
        d2 = (
            np.log(self.underlier / self.strike)
            + (self.r - 0.5 * self.sigma**2) * self.term
        ) / (self.sigma * np.sqrt(self.term))
        return self.strike * self.term * np.exp(-self.r * self.term) * norm.cdf(d2)


class BlackScholesPut(BlackScholes):
    def price(self):
        d1 = (
            np.log(self.underlier / self.strike)
            + (self.r + 0.5 * self.sigma**2) * self.term
        ) / (self.sigma * np.sqrt(self.term))
        d2 = d1 - self.sigma * np.sqrt(self.term)
        return self.strike * np.exp(-self.r * self.term) * norm.cdf(
            -d2
        ) - self.underlier * norm.cdf(-d1)
    def delta(self):
        d1 = (
            np.log(self.underlier / self.strike)
            + (self.r + 0.5 * self.sigma**2) * self.term
        ) / (self.sigma * np.sqrt(self.term))
        return norm.cdf(d1) - 1
    def gamma(self):
        d1 = (
            np.log(self.underlier / self.strike)
            + (self.r + 0.5 * self.sigma**2) * self.term
        ) / (self.sigma * np.sqrt(self.term))
        return norm.pdf(d1) / (self.underlier * self.sigma * np.sqrt(self.term))
    def vega(self):
        d1 = (
            np.log(self.underlier / self.strike)
            + (self.r + 0.5 * self.sigma**2) * self.term
        ) / (self.sigma * np.sqrt(self.term))
        return self.underlier * norm.pdf(d1) * np.sqrt(self.term) / 100
    def theta(self):
        d1 = (
            np.log(self.underlier / self.strike)
            + (self.r + 0.5 * self.sigma**2) * self.term
        ) / (self.sigma * np.sqrt(self.term))
        d2 = d1 - self.sigma * np.sqrt(self.term)
        return (-self.underlier * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.term)) + self.r * self.strike * np.exp(-self.r * self.term) * norm.cdf(-d2)) / 365
    def rho(self):
        d2 = (
            np.log(self.underlier / self.strike)
            + (self.r - 0.5 * self.sigma**2) * self.term
        ) / (self.sigma * np.sqrt(self.term))
        return -self.strike * self.term * np.exp(-self.r * self.term) * norm.cdf(-d2) / 100


class Barrier(BlackScholes):
    def __init__(
        self,
        underlier_price,
        strike,
        expiry,
        interest_rate,
        volatility,
        barrier,
        rebate=0,
    ):
        super().__init__(underlier_price, strike, expiry, interest_rate, volatility)
        self.barrier = barrier
        self.rebate = rebate

    def _compute_parameters(self):
        mu = self.r - 0.5 * self.sigma**2
        lambd = 1 + mu / self.sigma**2

        d1 = (
            np.log(self.underlier / self.barrier) / (self.sigma * self.term**0.5)
            + lambd * self.sigma * self.term**0.5
        )
        d2 = (
            np.log(self.barrier**2 / (self.underlier * self.strike))
            / (self.sigma * self.term**0.5)
            + lambd * self.sigma * self.term**0.5
        )
        d3 = (
            np.log(self.barrier / self.underlier) / (self.sigma * self.term**0.5)
            + lambd * self.sigma * self.term**0.5
        )
        d4 = (
            np.log(self.underlier / self.strike) / (self.sigma * self.term**0.5)
            + lambd * self.sigma * self.term**0.5
        )

        b1 = mu / self.sigma**2
        b2 = (mu**2 + 2 * self.r * self.sigma**2) ** 0.5 / self.sigma**2
        d5 = (
            np.log(self.barrier / self.underlier) / (self.sigma * self.term**0.5)
            + b2 * self.sigma * self.term**0.5
        )

        return lambd, d1, d2, d3, d4, b1, b2, d5

    def _compute_pricing_components(self, index=None):
        if index is None:
            index = []

        component_map = {}

        lambd, d1, d2, d3, d4, b1, b2, d5 = self._compute_parameters()

        if 1 in index:
            a1 = self.call_fl * self.underlier * norm.cdf(
                self.call_fl * d4
            ) - self.call_fl * self.strike * np.exp(-self.r * self.term) * norm.cdf(
                self.call_fl * d4 - self.call_fl * self.sigma * self.term**0.5
            )
            component_map["a1"] = a1
        if 2 in index:
            a2 = self.call_fl * self.underlier * norm.cdf(
                self.call_fl * d1
            ) - self.call_fl * self.strike * np.exp(-self.r * self.term) * norm.cdf(
                self.call_fl * d1 - self.call_fl * self.sigma * self.term**0.5
            )
            component_map["a2"] = a2
        if 3 in index:
            a3 = self.call_fl * self.underlier * (self.barrier / self.underlier) ** (
                2 * lambd
            ) * norm.cdf(self.down_fl * d2) - self.call_fl * self.strike * np.exp(
                -self.r * self.term
            ) * (
                self.barrier / self.underlier
            ) ** (
                2 * lambd - 2
            ) * norm.cdf(
                self.down_fl * d2 - self.down_fl * self.sigma * self.term**0.5
            )
            component_map["a3"] = a3
        if 4 in index:
            a4 = self.call_fl * self.underlier * (self.barrier / self.underlier) ** (
                2 * lambd
            ) * norm.cdf(self.down_fl * d3) - self.call_fl * self.strike * np.exp(
                -self.r * self.term
            ) * (
                self.barrier / self.underlier
            ) ** (
                2 * lambd - 2
            ) * norm.cdf(
                self.down_fl * d3 - self.down_fl * self.sigma * self.term**0.5
            )
            component_map["a4"] = a4
        if 5 in index:
            a5 = self.rebate * (
                norm.cdf(self.down_fl * d1 - self.down_fl * self.sigma * self.term**0.5)
                - (self.barrier / self.underlier) ** (2 * lambd - 2)
                * norm.cdf(
                    self.down_fl * d3 - self.down_fl * self.sigma * self.term**0.5
                )
            )
            component_map["a5"] = a5
        if 6 in index:
            a6 = self.rebate * (
                (self.barrier / self.underlier) ** (b1 + b2)
                * norm.cdf(self.down_fl * d5)
                + (self.barrier / self.underlier) ** (b1 - b2)
                * norm.cdf(
                    self.down_fl * d5
                    - 2 * self.down_fl * b2 * self.sigma * self.term**0.5
                )
            )
            component_map["a6"] = a6

        return component_map

# TODO: Implement vectorized return values for put options

class BarrierUpAndOutCall(Barrier):
    def __init__(
        self,
        underlier_price,
        strike,
        expiry,
        interest_rate,
        volatility,
        barrier,
        rebate=0,
    ):
        super().__init__(
            underlier_price, strike, expiry, interest_rate, volatility, barrier, rebate
        )
        self.call_fl = 1
        self.down_fl = -1

    def price(self):
        if np.all(self.strike <= self.barrier):
            component_map = self._compute_pricing_components([1, 2, 3, 4, 6])
            price_vector = (
                component_map["a1"]
                - component_map["a2"]
                + component_map["a3"]
                - component_map["a4"]
                + component_map["a6"]
            )
        else:
            component_map = self._compute_pricing_components([6])
            price_vector = component_map["a6"]

        if isinstance(price_vector, Iterable):
            price_vector[self.underlier >= self.barrier] = 0
            price_vector = np.nan_to_num(price_vector, nan=0)
            return price_vector
        return price_vector if self.underlier < self.barrier else 0

    def delta(self, precision=0.01):
        s_up = self.underlier + precision
        s_down = self.underlier - precision
        
        price_up = BarrierUpAndOutCall(
            s_up, self.strike, self.term, self.r, self.sigma, self.barrier
        ).price()
        price_down = BarrierUpAndOutCall(
            s_down, self.strike, self.term, self.r, self.sigma, self.barrier
        ).price()
        return (price_up - price_down) / (2 * precision)


class BarrierUpAndOutPut(Barrier):
    def __init__(
        self,
        underlier_price,
        strike,
        expiry,
        interest_rate,
        volatility,
        barrier,
        rebate=0,
    ):
        super().__init__(
            underlier_price, strike, expiry, interest_rate, volatility, barrier, rebate
        )
        self.call_fl = -1
        self.down_fl = -1

    def price(self):
        if np.all(self.strike <= self.barrier):
            component_map = self._compute_pricing_components([1, 3, 6])
            return component_map["a1"] - component_map["a3"] + component_map["a6"]

        component_map = self._compute_pricing_components([2, 4, 6])
        return component_map["a2"] - component_map["a4"] + component_map["a6"]


class BarrierUpAndInCall(Barrier):
    def __init__(
        self,
        underlier_price,
        strike,
        expiry,
        interest_rate,
        volatility,
        barrier,
        rebate=0,
    ):
        super().__init__(
            underlier_price, strike, expiry, interest_rate, volatility, barrier, rebate
        )
        self.call_fl = 1
        self.down_fl = -1

    def price(self):
        if np.all(self.strike <= self.barrier):
            component_map = self._compute_pricing_components([2, 3, 4, 5])
            price_vector = (
                component_map["a2"]
                - component_map["a3"]
                + component_map["a4"]
                + component_map["a5"]
            )
        else:
            component_map = self._compute_pricing_components([1, 5])
            price_vector = component_map["a1"] + component_map["a5"]

        if isinstance(price_vector, Iterable):
            price_vector[self.underlier >= self.barrier] = BlackScholesCall(
                self.underlier[self.underlier >= self.barrier],
                self.strike,
                self.term,
                self.r,
                self.sigma,
            ).price()
            price_vector = np.nan_to_num(price_vector, nan=0)
            return price_vector
        return price_vector if self.underlier < self.barrier else BlackScholesCall(
            self.underlier, self.strike, self.term, self.r, self.sigma
        ).price()


class BarrierUpAndInPut(Barrier):
    def __init__(
        self,
        underlier_price,
        strike,
        expiry,
        interest_rate,
        volatility,
        barrier,
        rebate=0,
    ):
        super().__init__(
            underlier_price, strike, expiry, interest_rate, volatility, barrier, rebate
        )
        self.call_fl = -1
        self.down_fl = -1

    def price(self):
        if np.all(self.strike <= self.barrier):
            component_map = self._compute_pricing_components([3, 5])
            return component_map["a3"] + component_map["a5"]

        component_map = self._compute_pricing_components([1, 2, 4, 5])
        return (
            component_map["a1"]
            - component_map["a2"]
            + component_map["a4"]
            + component_map["a5"]
        )


class BarrierDownAndInCall(Barrier):
    def __init__(
        self,
        underlier_price,
        strike,
        expiry,
        interest_rate,
        volatility,
        barrier,
        rebate=0,
    ):
        super().__init__(
            underlier_price, strike, expiry, interest_rate, volatility, barrier, rebate
        )
        self.call_fl = 1
        self.down_fl = 1

    def price(self):
        if np.all(self.strike <= self.barrier):
            component_map = self._compute_pricing_components([1, 2, 4, 5])
            price_vector = (
                component_map["a1"]
                - component_map["a2"]
                + component_map["a4"]
                + component_map["a5"]
            )
        else:
            component_map = self._compute_pricing_components([3, 5])
            price_vector = component_map["a3"] + component_map["a5"]

        if isinstance(price_vector, Iterable):
            price_vector[self.underlier <= self.barrier] = BlackScholesCall(
                self.underlier[self.underlier <= self.barrier],
                self.strike,
                self.term,
                self.r,
                self.sigma,
            ).price()
            price_vector = np.nan_to_num(price_vector, nan=0)
            return price_vector
        return (
            price_vector
            if self.underlier > self.barrier
            else BlackScholesCall(
                self.underlier, self.strike, self.term, self.r, self.sigma
            ).price()
        )


class BarrierDownAndInPut(Barrier):
    def __init__(
        self,
        underlier_price,
        strike,
        expiry,
        interest_rate,
        volatility,
        barrier,
        rebate=0,
    ):
        super().__init__(
            underlier_price, strike, expiry, interest_rate, volatility, barrier, rebate
        )
        self.call_fl = -1
        self.down_fl = 1

    def price(self):
        if np.all(self.strike <= self.barrier):
            component_map = self._compute_pricing_components([1, 5])
            return component_map["a1"] + component_map["a5"]

        component_map = self._compute_pricing_components([2, 3, 4, 5])
        return (
            component_map["a2"]
            - component_map["a3"]
            + component_map["a4"]
            + component_map["a5"]
        )


class BarrierDownAndOutCall(Barrier):
    def __init__(
        self,
        underlier_price,
        strike,
        expiry,
        interest_rate,
        volatility,
        barrier,
        rebate=0,
    ):
        super().__init__(
            underlier_price, strike, expiry, interest_rate, volatility, barrier, rebate
        )
        self.call_fl = 1
        self.down_fl = 1

    def price(self):
        if np.all(self.strike <= self.barrier):
            component_map = self._compute_pricing_components([2, 4, 6])
            price_vector = component_map["a2"] - component_map["a4"] + component_map["a6"]
        else:
            component_map = self._compute_pricing_components([1, 3, 6])
            price_vector = component_map["a1"] - component_map["a3"] + component_map["a6"]

        if isinstance(price_vector, Iterable):
            price_vector[self.underlier <= self.barrier] = 0
            price_vector = np.nan_to_num(price_vector, nan=0)
            return price_vector
        return price_vector if self.underlier > self.barrier else 0


class BarrierDownAndOutPut(Barrier):
    def __init__(
        self,
        underlier_price,
        strike,
        expiry,
        interest_rate,
        volatility,
        barrier,
        rebate=0,
    ):
        super().__init__(
            underlier_price, strike, expiry, interest_rate, volatility, barrier, rebate
        )
        self.call_fl = -1
        self.down_fl = 1

    def price(self):
        if np.all(self.strike <= self.barrier):
            component_map = self._compute_pricing_components([6])
            return component_map["a6"]

        component_map = self._compute_pricing_components([1, 2, 3, 4, 6])
        return (
            component_map["a1"]
            - component_map["a2"]
            + component_map["a3"]
            - component_map["a4"]
            + component_map["a6"]
        )
