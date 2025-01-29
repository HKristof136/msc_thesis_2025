import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from pricer.config_base import BlackScholesConfig, PDESolverConfig, DefaultConfig
from pricer.analytical import BlackScholesCall, BlackScholesPut
from pricer.pde_solver_base import CrankNicolsonPDESolver, BarrierUpPDE, BarrierDownPDE, AmericanBarrierUpPDE, \
    AmericanBarrierDownPDE, AmericanOptionPDE


class BlackScholesCallPDE(CrankNicolsonPDESolver):
    underlier_lower_boundary = lambda self: 0.0
    underlier_upper_boundary = lambda self: np.maximum(
        self.x[-1, :]
        - np.exp(-self.r * (np.max(self.max_t) - self.t[-1, :])) * self.strike,
        0.0,
    )
    payoff = lambda self: np.maximum(self.x[:, -1] - self.strike, 0.0)


class BlackScholesPutPDE(CrankNicolsonPDESolver):
    underlier_lower_boundary = (
        lambda self: np.exp(-self.r * (np.max(self.max_t) - self.t[-1, :]))
        * self.strike
    )
    underlier_upper_boundary = lambda self: 0.0
    payoff = lambda self: np.maximum(self.strike - self.x[:, -1], 0.0)


class AmericanBlackScholesPutPDE(AmericanOptionPDE):
    underlier_lower_boundary = lambda self: self.strike
    underlier_upper_boundary = lambda self: 0.0
    payoff = lambda self: np.maximum(self.strike - self.x[:, -1], 0.0)


class BarrierUpAndOutCallPDE(BarrierUpPDE):
    underlier_lower_boundary = lambda self: 0
    underlier_upper_boundary = lambda self: 0
    payoff = lambda self: np.maximum(self.x[:, -1] - self.strike, 0) * (
        self.x[:, -1] < self.barrier
    )

    def __init__(self, config: PDESolverConfig = DefaultConfig.pde_solver_barrier_call):
        super().__init__(config)

class BarrierUpAndOutPutPDE(BarrierUpPDE):
    underlier_lower_boundary = (
        lambda self: np.exp(-self.r * (np.max(self.max_t) - self.t[-1, :]))
        * self.strike
    )
    underlier_upper_boundary = lambda self: 0
    payoff = lambda self: np.maximum(self.strike - self.x[:, -1], 0) * (
        self.x[:, -1] < self.barrier
    )

    def __init__(self, config: PDESolverConfig = DefaultConfig.pde_solver_barrier_put):
        super().__init__(config)

class BarrierUpAndInCallPDE(BarrierUpPDE):
    def __init__(self, config: PDESolverConfig = DefaultConfig.pde_solver_barrier_call):
        super().__init__(config)

    def solve(self):
        if self.solved:
            print("Already solved")
            return None

        call_config = BlackScholesConfig(
            underlier_price=self.x,
            strike=self.strike,
            expiry=self.max_t - self.t,
            interest_rate=self.r,
            volatility=self.sigma,
        )
        call_grid = BlackScholesCall(call_config).price()

        ko_grid_cls = BarrierUpAndOutCallPDE(self.config)
        ko_grid_cls.solve()

        self.grid = call_grid - ko_grid_cls.grid
        self.solved = True


class BarrierUpAndInPutPDE(BarrierUpPDE):
    def __init__(self, config: PDESolverConfig = DefaultConfig.pde_solver_barrier_put):
        super().__init__(config)

    def solve(self):
        if self.solved:
            print("Already solved")
            return None

        put_config = BlackScholesConfig(
            underlier_price=self.x,
            strike=self.strike,
            expiry=self.max_t - self.t,
            interest_rate=self.r,
            volatility=self.sigma,
        )
        put_grid = BlackScholesPut(put_config).price()

        ko_grid_cls = BarrierUpAndOutPutPDE(self.config)
        ko_grid_cls.solve()

        self.grid = put_grid - ko_grid_cls.grid
        self.solved = True


class BarrierDownAndOutCallPDE(BarrierDownPDE):
    underlier_lower_boundary = lambda self: 0
    underlier_upper_boundary = lambda self: np.maximum(
        self.x[-1, :]
        - np.exp(-self.r * (np.max(self.max_t) - self.t[-1, :])) * self.strike,
        0,
    )
    payoff = lambda self: np.maximum(self.x[:, -1] - self.strike, 0) * (
        self.x[:, -1] > self.barrier
    )

    def __init__(self, config: PDESolverConfig = DefaultConfig.pde_solver_barrier_call):
        super().__init__(config)


class BarrierDownAndOutPutPDE(BarrierDownPDE):
    underlier_lower_boundary = lambda self: 0
    underlier_upper_boundary = lambda self: 0
    payoff = lambda self: np.maximum(self.strike - self.x[:, -1], 0) * (
            self.x[:, -1] > self.barrier
    )

    def __init__(self, config: PDESolverConfig = DefaultConfig.pde_solver_barrier_put):
        super().__init__(config)


class BarrierDownAndInCallPDE(BarrierDownPDE):
    def __init__(self, config: PDESolverConfig = DefaultConfig.pde_solver_barrier_call):
        super().__init__(config)

    def solve(self):
        if self.solved:
            print("Already solved")
            return None

        call_config = BlackScholesConfig(
            underlier_price=self.x,
            strike=self.strike,
            expiry=self.max_t - self.t,
            interest_rate=self.r,
            volatility=self.sigma,
        )
        call_grid = BlackScholesCall(call_config).price()

        ko_grid_cls = BarrierDownAndOutCallPDE(self.config)
        ko_grid_cls.solve()

        self.grid = call_grid - ko_grid_cls.grid
        self.solved = True


class BarrierDownAndInPutPDE(BarrierDownPDE):
    def __init__(self, config: PDESolverConfig = DefaultConfig.pde_solver_barrier_put):
        super().__init__(config)

    def solve(self):
        if self.solved:
            print("Already solved")
            return None

        put_config = BlackScholesConfig(
            underlier_price=self.x,
            strike=self.strike,
            expiry=self.max_t - self.t,
            interest_rate=self.r,
            volatility=self.sigma,
        )
        put_grid = BlackScholesPut(put_config).price()

        ko_grid_cls = BarrierDownAndOutPutPDE(self.config)
        ko_grid_cls.solve()

        self.grid = put_grid - ko_grid_cls.grid
        self.solved = True


class AmericanBarrierUpAndOutCallPDE(AmericanBarrierUpPDE):
    underlier_lower_boundary = lambda self: 0.0
    underlier_upper_boundary = lambda self: 0.0
    payoff = lambda self: np.maximum(self.x[:, -1] - self.strike, 0.0) * (
        self.x[:, -1] < self.barrier
    )

class AmericanBarrierUpAndOutPutPDE(AmericanBarrierUpPDE):
    underlier_lower_boundary = lambda self: self.strike
    underlier_upper_boundary = lambda self: 0.0
    payoff = lambda self: np.maximum(self.strike - self.x[:, -1], 0.0) * (
        self.x[:, -1] < self.barrier
    )


class AmericanBarrierUpAndInCallPDE(AmericanBarrierUpPDE):
    underlier_lower_boundary = lambda self: 0
    underlier_upper_boundary = lambda self: np.maximum(self.x[-1, :] - self.strike, 0)
    payoff = lambda self: np.maximum(self.x[:, -1] - self.strike, 0) * (
        self.x[:, -1] > self.barrier
    )

    def __init__(self, config: PDESolverConfig = DefaultConfig.pde_solver_barrier_call):
        super().__init__(config)
        reduced_j = np.min(np.nonzero(self.config.underlier_price_grid >= self.barrier))

        call_config = BlackScholesConfig(
            underlier_price=self.x[reduced_j:, :],
            strike=self.strike,
            expiry=self.max_t - self.t[reduced_j:, :],
            interest_rate=self.r,
            volatility=self.sigma,
        )
        self.grid[reduced_j:, :] = BlackScholesCall(call_config).price()

        call_config = BlackScholesConfig(
            underlier_price=self.barrier,
            strike=self.strike,
            expiry=self.max_t - self.t[0, :],
            interest_rate=self.r,
            volatility=self.sigma,
        )
        self.f = BlackScholesCall(call_config).price()


class AmericanBarrierUpAndInPutPDE(AmericanBarrierUpPDE):
    underlier_lower_boundary = lambda self: 0.0
    underlier_upper_boundary = lambda self: 0.0
    payoff = lambda self: np.maximum(self.strike - self.x[:, -1], 0.0) * (
        self.x[:, -1] > self.barrier
    )

    def __init__(self, config: PDESolverConfig = DefaultConfig.pde_solver_barrier_put):
        super().__init__(config)
        reduced_j = np.min(np.nonzero(self.config.underlier_price_grid >= self.barrier))

        put_grid_cls = AmericanBlackScholesPutPDE(self.config)
        put_grid_cls.solve()
        self.grid[reduced_j:, :] = put_grid_cls.grid[reduced_j:, :]
        self.f = put_grid_cls.price([[self.barrier, self.max_t - t] for t in self.t[0, :]])


class AmericanBarrierDownAndOutCallPDE(AmericanBarrierDownPDE):
    underlier_lower_boundary = lambda self: 0.0
    underlier_upper_boundary = lambda self: np.maximum(self.x[-1, :] - self.strike, 0)
    payoff = lambda self: np.maximum(self.x[:, -1] - self.strike, 0.0) * (
            self.x[:, -1] < self.barrier
    )

class AmericanBarrierDownAndOutPutPDE(AmericanBarrierDownPDE):
    underlier_lower_boundary = lambda self: 0.0
    underlier_upper_boundary = lambda self: 0.0
    payoff = lambda self: np.maximum(self.strike - self.x[:, -1], 0.0) * (
            self.x[:, -1] < self.barrier
    )

class AmericanBarrierDownAndInCallPDE(AmericanBarrierDownPDE):
    underlier_lower_boundary = lambda self: 0.0
    underlier_upper_boundary = lambda self: 0.0
    payoff = lambda self: np.maximum(self.x[:, -1] - self.strike, 0.0) * (
        self.x[:, -1] < self.barrier
    )
    def __init__(self, config: PDESolverConfig = DefaultConfig.pde_solver_barrier_call):
        super().__init__(config)
        reduced_j = np.max(np.nonzero(self.config.underlier_price_grid <= self.barrier))

        call_config = BlackScholesConfig(
            underlier_price=self.x[:reduced_j, :],
            strike=self.strike,
            expiry=self.max_t - self.t[:reduced_j, :],
            interest_rate=self.r,
            volatility=self.sigma,
        )
        self.grid[:reduced_j, :] = BlackScholesCall(call_config).price()

        call_config = BlackScholesConfig(
            underlier_price=self.barrier,
            strike=self.strike,
            expiry=self.max_t - self.t[0, :],
            interest_rate=self.r,
            volatility=self.sigma,
        )
        self.f = BlackScholesCall(call_config).price()


class AmericanBarrierDownAndInPutPDE(AmericanBarrierDownPDE):
    underlier_lower_boundary = lambda self: self.strike
    underlier_upper_boundary = lambda self: 0.0
    payoff = lambda self: np.maximum(self.strike - self.x[:, -1], 0.0) * (
        self.x[:, -1] < self.barrier
    )

    def __init__(self, config: PDESolverConfig = DefaultConfig.pde_solver_barrier_call):
        super().__init__(config)
        reduced_j = np.max(np.nonzero(self.config.underlier_price_grid <= self.barrier))

        put_grid_cls = AmericanBlackScholesPutPDE(self.config)
        put_grid_cls.solve()
        self.grid[:reduced_j, :] = put_grid_cls.grid[:reduced_j, :]
        self.f = put_grid_cls.price([[self.barrier, self.max_t - t] for t in self.t[0, :]])
