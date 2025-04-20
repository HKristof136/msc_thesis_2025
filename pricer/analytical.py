from scipy.integrate import quad
from scipy.stats import norm
import numpy as np
from abc import ABC, abstractmethod
from pricer.config_base import BlackScholesConfig, HestonConfig, DefaultConfig


class BlackScholes(ABC):
    type = "analytical"
    input_names = ["underlier_price", "strike", "expiry", "interest_rate", "volatility"]

    def __init__(self, config: BlackScholesConfig = DefaultConfig.black_scholes):
        self.config = config

        self.underlier = config.underlier_price
        self.strike = config.strike
        self.term = config.expiry
        self.r = config.interest_rate
        self.sigma = config.volatility
        self.barrier = config.barrier

    @abstractmethod
    def price(self):
        pass

    def _calc_greek(self, greek, order, precision):
        up_config = BlackScholesConfig(
            underlier_price=(
                self.underlier + precision if greek == "delta" else self.underlier
            ),
            strike=self.strike,
            expiry=self.term + precision if greek == "theta" else self.term,
            interest_rate=self.r + precision if greek == "rho" else self.r,
            volatility=self.sigma + precision if greek == "vega" else self.sigma,
            barrier=self.barrier,
        )
        price_up = self.__class__(up_config).price()

        down_config = BlackScholesConfig(
            underlier_price=(
                self.underlier - precision if greek == "delta" else self.underlier
            ),
            strike=self.strike,
            expiry=self.term - precision if greek == "theta" else self.term,
            interest_rate=self.r - precision if greek == "rho" else self.r,
            volatility=self.sigma - precision if greek == "vega" else self.sigma,
            barrier=self.barrier,
        )
        price_down = self.__class__(down_config).price()
        if order == "first":
            return (price_up - price_down) / (2 * precision)
        else:
            return (price_up - 2 * self.price() + price_down) / precision**2

    def delta(self, precision=0.01):
        return self._calc_greek("delta", "first", precision)

    def gamma(self, precision=0.01):
        return self._calc_greek("delta", "second", precision)

    def vega(self, precision=0.001):
        return self._calc_greek("vega", "first", precision)

    def theta(self, precision=0.001):
        return self._calc_greek("theta", "first", precision)

    def rho(self, precision=0.001):
        return self._calc_greek("rho", "first", precision)


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

    def delta(self, precision=0.0):
        d1 = (
            np.log(self.underlier / self.strike)
            + (self.r + 0.5 * self.sigma**2) * self.term
        ) / (self.sigma * np.sqrt(self.term))
        return norm.cdf(d1)

    def gamma(self, precision=0.0):
        d1 = (
            np.log(self.underlier / self.strike)
            + (self.r + 0.5 * self.sigma**2) * self.term
        ) / (self.sigma * np.sqrt(self.term))
        return norm.pdf(d1) / (self.underlier * self.sigma * np.sqrt(self.term))

    def vega(self, precision=0.0):
        d1 = (
            np.log(self.underlier / self.strike)
            + (self.r + 0.5 * self.sigma**2) * self.term
        ) / (self.sigma * np.sqrt(self.term))
        return self.underlier * norm.pdf(d1) * np.sqrt(self.term)

    def theta(self, precision=0.0):
        d1 = (
            np.log(self.underlier / self.strike)
            + (self.r + 0.5 * self.sigma**2) * self.term
        ) / (self.sigma * np.sqrt(self.term))
        d2 = d1 - self.sigma * np.sqrt(self.term)
        return (-1) * (
            -self.underlier * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.term))
            - self.r * self.strike * np.exp(-self.r * self.term) * norm.cdf(d2)
        )

    def rho(self, precision=0.0):
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

    def delta(self, precision=0.0):
        d1 = (
            np.log(self.underlier / self.strike)
            + (self.r + 0.5 * self.sigma**2) * self.term
        ) / (self.sigma * np.sqrt(self.term))
        return norm.cdf(d1) - 1

    def gamma(self, precision=0.0):
        d1 = (
            np.log(self.underlier / self.strike)
            + (self.r + 0.5 * self.sigma**2) * self.term
        ) / (self.sigma * np.sqrt(self.term))
        return norm.pdf(d1) / (self.underlier * self.sigma * np.sqrt(self.term))

    def vega(self, precision=0.0):
        d1 = (
            np.log(self.underlier / self.strike)
            + (self.r + 0.5 * self.sigma**2) * self.term
        ) / (self.sigma * np.sqrt(self.term))
        return self.underlier * norm.pdf(d1) * np.sqrt(self.term)

    def theta(self, precision=0.0):
        # d1 = (
        #     np.log(self.underlier / self.strike)
        #     + (self.r + 0.5 * self.sigma**2) * self.term
        # ) / (self.sigma * np.sqrt(self.term))
        # d2 = d1 - self.sigma * np.sqrt(self.term)
        # return (-1) * (
        #     -self.underlier * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.term))
        #     + self.r * self.strike * np.exp(-self.r * self.term) * norm.cdf(d2)
        # )
        return super().theta(precision=0.0001)

    def rho(self, precision=0.0):
        d2 = (
            np.log(self.underlier / self.strike)
            + (self.r - 0.5 * self.sigma**2) * self.term
        ) / (self.sigma * np.sqrt(self.term))
        return -self.strike * self.term * np.exp(-self.r * self.term) * norm.cdf(-d2)


class Barrier(BlackScholes):
    input_names = [
        "underlier_price",
        "strike",
        "expiry",
        "interest_rate",
        "volatility",
        "barrier",
    ]

    def __init__(
        self,
        config: BlackScholesConfig = DefaultConfig.black_scholes_barrier_call,
    ):
        super().__init__(config)
        self.barrier = config.barrier
        self.call_fl = None
        self.down_fl = None

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
        return lambd, d1, d2, d3, d4

    def _compute_pricing_components(self, index=None):
        if index is None:
            index = []

        component_map = {}

        lambd, d1, d2, d3, d4 = self._compute_parameters()

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
        return component_map

    def price(self):
        pass


class BarrierUpAndOutCall(Barrier):
    def __init__(
        self,
        config: BlackScholesConfig = DefaultConfig.black_scholes_barrier_call,
    ):
        super().__init__(config)
        self.call_fl = 1
        self.down_fl = -1

    def price(self):
        component_map = self._compute_pricing_components([1, 2, 3, 4])
        price_vector = (
            component_map["a1"]
            - component_map["a2"]
            + component_map["a3"]
            - component_map["a4"]
        )
        if isinstance(price_vector, np.ndarray):
            price_vector[self.underlier >= self.barrier] = 0.0
            price_vector = np.nan_to_num(price_vector, nan=0.0)
            return price_vector
        else:
            if self.underlier >= self.barrier:
                return 0.0
            else:
                return price_vector


class BarrierUpAndOutPut(Barrier):
    def __init__(
        self,
        config: BlackScholesConfig = DefaultConfig.black_scholes_barrier_put,
    ):
        super().__init__(config)
        self.call_fl = -1
        self.down_fl = -1

    def price(self):
        component_map = self._compute_pricing_components([1, 2, 3, 4])
        price_vector = component_map["a1"] - component_map["a3"]

        mask = self.strike > self.barrier
        price_vector[mask] = component_map["a2"][mask] - component_map["a4"][mask]

        if isinstance(price_vector, np.ndarray):
            price_vector[self.underlier >= self.barrier] = 0.0
            price_vector = np.nan_to_num(price_vector, nan=0.0)
            return price_vector
        return 0.0 if self.underlier >= self.barrier else price_vector


class BarrierUpAndInCall(Barrier):
    def __init__(
        self,
        config: BlackScholesConfig = DefaultConfig.black_scholes_barrier_call,
    ):
        super().__init__(config)

    def price(self):
        ko_price = BarrierUpAndOutCall(self.config).price()
        call_price = BlackScholesCall(self.config).price()
        return call_price - ko_price


class BarrierUpAndInPut(Barrier):
    def __init__(
        self,
        config: BlackScholesConfig = DefaultConfig.black_scholes_barrier_put,
    ):
        super().__init__(config)

    def price(self):
        ko_price = BarrierUpAndOutPut(self.config).price()
        put_price = BlackScholesPut(self.config).price()
        return put_price - ko_price


class BarrierDownAndOutCall(Barrier):
    def __init__(
        self,
        config: BlackScholesConfig = DefaultConfig.black_scholes_barrier_call,
    ):
        super().__init__(config)
        self.call_fl = 1
        self.down_fl = 1

    def price(self):
        component_map = self._compute_pricing_components([1, 2, 3, 4])
        price_vector = component_map["a2"] - component_map["a4"]

        mask = self.strike > self.barrier
        price_vector[mask] = component_map["a1"][mask] - component_map["a3"][mask]

        if isinstance(price_vector, np.ndarray):
            price_vector[self.underlier <= self.barrier] = 0.0
            price_vector = np.nan_to_num(price_vector, nan=0.0)
            return price_vector
        return price_vector if self.underlier > self.barrier else 0.0


class BarrierDownAndOutPut(Barrier):
    def __init__(
        self,
        config: BlackScholesConfig = DefaultConfig.black_scholes_barrier_put,
    ):
        super().__init__(config)
        self.call_fl = -1
        self.down_fl = 1

    def price(self):
        component_map = self._compute_pricing_components([1, 2, 3, 4])
        price_vector = (
            component_map["a1"]
            - component_map["a2"]
            + component_map["a3"]
            - component_map["a4"]
        )

        mask = self.strike <= self.barrier
        price_vector[mask] = 0.0

        if isinstance(price_vector, np.ndarray):
            price_vector[self.underlier <= self.barrier] = 0.0
            price_vector = np.nan_to_num(price_vector, nan=0.0)
            return price_vector
        return price_vector if self.underlier > self.barrier else 0.0


class BarrierDownAndInCall(Barrier):
    def __init__(
        self,
        config: BlackScholesConfig = DefaultConfig.black_scholes_barrier_call,
    ):
        super().__init__(config)
        self.call_fl = 1
        self.down_fl = 1

    def price(self):
        ko_price = BarrierDownAndOutCall(self.config).price()
        call_price = BlackScholesCall(self.config).price()

        return call_price - ko_price


class BarrierDownAndInPut(Barrier):
    def __init__(
        self,
        config: BlackScholesConfig = DefaultConfig.black_scholes_barrier_put,
    ):
        super().__init__(config)
        self.call_fl = -1
        self.down_fl = 1

    def price(self):
        ko_price = BarrierDownAndOutPut(self.config).price()
        put_price = BlackScholesPut(self.config).price()

        return put_price - ko_price


class ImpliedVol:
    type = "implied_vol"
    input_names = ["underlier_price", "strike", "expiry", "interest_rate", "price"]


if __name__ == "__main__":
    pricer_cls = HestonCall()
    price = pricer_cls.price()
    print("success")
