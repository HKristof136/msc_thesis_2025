import sys
import os
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from pricer.config_base import BlackScholesConfig, PDESolverConfig, DefaultConfig
from pricer.analytical import BlackScholesCall, BlackScholesPut
from pricer.pde_solver_base import CrankNicolsonPDESolver, BarrierUpPDE, BarrierDownPDE, AmericanBarrierUpPDE, \
    AmericanBarrierDownPDE, AmericanOptionPDE, ADISolver
from scipy.sparse.linalg import splu
from scipy.sparse import identity, csc_matrix
from tqdm import tqdm


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

    def solve(self):
        super().solve()
        self.grid = np.maximum(0.0, self.grid)


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

    def solve(self):
        super().solve()
        self.grid = np.maximum(0.0, self.grid)

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
            expiry=(self.max_t - self.t) + 10**(-6),
            interest_rate=self.r,
            volatility=self.sigma,
        )
        call_grid = BlackScholesCall(call_config).price()

        ko_grid_cls = BarrierUpAndOutCallPDE(self.config)
        ko_grid_cls.solve()

        self.grid = call_grid - ko_grid_cls.grid
        self.grid = np.maximum(0.0, self.grid)
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
            expiry=(self.max_t - self.t) + 10**(-6),
            interest_rate=self.r,
            volatility=self.sigma,
        )
        put_grid = BlackScholesPut(put_config).price()

        reduced_j = np.min(np.nonzero(self.config.underlier_price_grid >= self.barrier))
        self.grid[reduced_j:, :] = put_grid[reduced_j:, :]

        self.f = BlackScholesPut(BlackScholesConfig(
            underlier_price=self.barrier,
            strike=self.strike,
            expiry=(self.max_t - self.t)[0, :] + 10**(-6),
            interest_rate=self.r,
            volatility=self.sigma,
        )).price()

        super()._solve()


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

    def solve(self):
        super().solve()
        self.grid = np.maximum(0.0, self.grid)


class BarrierDownAndOutPutPDE(BarrierDownPDE):
    underlier_lower_boundary = lambda self: 0
    underlier_upper_boundary = lambda self: 0
    payoff = lambda self: np.maximum(self.strike - self.x[:, -1], 0) * (
            self.x[:, -1] > self.barrier
    )

    def __init__(self, config: PDESolverConfig = DefaultConfig.pde_solver_barrier_put):
        super().__init__(config)

    def solve(self):
        super().solve()
        self.grid = np.maximum(0.0, self.grid)


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
            expiry=(self.max_t - self.t) + 10**(-6),
            interest_rate=self.r,
            volatility=self.sigma,
        )
        call_grid = BlackScholesCall(call_config).price()
        reduced_j = np.max(np.nonzero(self.config.underlier_price_grid <= self.barrier))
        self.grid[:reduced_j+1, :] = call_grid[:reduced_j+1, :]

        self.f = BlackScholesCall(BlackScholesConfig(
            underlier_price=self.barrier,
            strike=self.strike,
            expiry=(self.max_t - self.t)[0, :] + 10**(-6),
            interest_rate=self.r,
            volatility=self.sigma,
        )).price()

        super().solve()


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
            expiry=(self.max_t - self.t) + 10**(-6),
            interest_rate=self.r,
            volatility=self.sigma,
        )
        put_grid = BlackScholesPut(put_config).price()

        ko_grid_cls = BarrierDownAndOutPutPDE(self.config)
        ko_grid_cls.solve()

        self.grid = put_grid - ko_grid_cls.grid
        self.grid = np.maximum(0.0, self.grid)
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

class ADICallPDE(ADISolver):
    pass

class ADIBarrierUpAndOutCallPDE(ADISolver):
    type = "adi"
    input_names = ["strike", "interest_rate", "corr", "kappa", "variance_theta", "sigma", "barrier"]

    def __init__(self, config: PDESolverConfig = DefaultConfig.pde_solver):
        super().__init__(config)
        self.barrier = config.barrier

        c = self.barrier / 10
        non_uniform_s = np.zeros(self.m)
        for i in range(self.m):
            dk_i = (1 / self.m) * math.asinh(self.barrier / c)
            non_uniform_s[i] = self.barrier - abs(c * math.sinh(i * dk_i))
        non_uniform_s = non_uniform_s[::-1]
        non_uniform_s[-1] += self.strike / 100

        V = 5 * self.sigma
        d = V / 500

        non_uniform_v = np.zeros(self.n)
        for j in range(self.n):
            dk_i = (1 / self.n) * math.asinh(V / d)
            non_uniform_v[j] = abs(d * math.sinh(j * dk_i))

        non_uniform_v = np.array(sorted(list(non_uniform_v) + [V]))
        self.n += 1

        self.vv, self.tt, self.xx = np.meshgrid(
            non_uniform_v,
            config.time_grid,
            non_uniform_s
        )

        self.dxx = np.zeros_like(self.xx[0, 0, :])
        self.dxx[1:] = self.xx[0, 0, 1:] - self.xx[0, 0, :-1]
        self.dxx = np.hstack([self.dxx, self.dxx[-1]])
        self.dxx = tuple(self.dxx)

        self.dvv = np.zeros_like(self.vv[0, :, 0])
        self.dvv[1:] = self.vv[0, 1:, 0] - self.vv[0, :-1, 0]
        self.dvv = tuple(self.dvv)

        self.grid = np.zeros_like(self.xx)

        #payoff boundary condition
        self.grid[-1, :, :] = (self.xx[-1, :, :] < self.barrier) * np.maximum(self.xx[-1, :, :] - self.strike, 0)

        # s=0 boundary condidition
        self.grid[:, :, 0] = 0.0

        # v = V boundary condition
        # self.grid[:, -1, :] = np.maximum(self.xx[:, -1, :] * np.exp(-self.rf * self.tt[:, -1, :]) - self.strike * (np.exp(-self.rd * self.tt[:, -1, :])), 0)
        self.grid[:, -1, :] = 0.0

    def solve(self):
        A0 = {}
        A1 = {}
        A2 = {}

        for i in range(0, self.n - 1):
            for j in range(1, self.m):
                idx = i * self.m + j
                if not i:
                    for o in [-1, 0, 1]:
                        A1[(idx, idx + o)] = A1.get((idx, idx + o), 0.0) + int(j != (self.m - 1)) * ((self.rd - self.rf) * self.xx[0, 0, j] * self.beta_coef(self.dxx, j, o)) - int(o == 0) * self.rd / 2

                    for o in [0, 1, 2]:
                        A2[(idx, idx + o * self.m)] = A2.get((idx, idx + o * self.m), 0.0) + self.k * (self.th - self.vv[0, i, 0]) * self.gamma_coef(self.dvv, i, o) - int(o == 0) * self.rd / 2

                else:
                    if self.corr and (j != (self.m - 1)):
                        cross_coef = self.corr * self.sigma * self.xx[0, 0, j] * self.vv[0, i, 0]
                        for k in range(-1, 2):
                            for l in range(-1, 2):
                                A0[(idx, idx + k * self.m + l)] = A0.get((idx, idx + k * self.m + l), 0.0) + (
                                        cross_coef * self.beta_coef(self.dxx, j, k) * self.beta_coef(self.dvv, i, l)
                                )

                    for o in [-1, 0, 1]:
                        A1[(idx, idx + o)] = A1.get((idx, idx + o), 0.0) + int(j != (self.m - 1)) * (
                                0.5 * self.xx[0, 0, j] ** 2 * self.vv[0, i, 0] * self.delta_coef(self.dxx, j, o) + (self.rd - self.rf) * self.xx[0, 0, j] * self.beta_coef(self.dxx, j, o)
                        ) - int(o == 0) * self.rd / 2

                    if self.vv[0, i, 0] <= 1:
                        for o in [-1, 0, 1]:
                            A2[(idx, idx + o * self.m)] = A2.get((idx, idx + o * self.m), 0.0) + (
                                    0.5 * self.sigma ** 2 * self.vv[0, i, 0] * self.delta_coef(self.dvv, i, o) + self.k * (self.th - self.vv[0, i, 0]) * self.beta_coef(self.dvv, i, o)
                            ) - int(o == 0) * self.rd / 2

                    else:
                        for o in [-1, 0, 1]:
                            A2[(idx, idx + o * self.m)] = A2.get((idx, idx + o * self.m), 0.0) + (
                                    0.5 * self.sigma ** 2 * self.vv[0, i, 0] * self.delta_coef(self.dvv, i, o)
                            ) - int(o == 0) * self.rd / 2

                        for o in [-2, -1, 0]:
                            A2[(idx, idx + o * self.m)] = A2.get((idx, idx + o * self.m), 0.0) + (
                                    self.k * (self.th - self.vv[0, i, 0]) * self.alpha_coef(self.dvv, i, o)
                            )
        A0 = csc_matrix((list(A0.values()), tuple(zip(*A0.keys()))), shape=(self.n * self.m, self.n * self.m), dtype=np.float32)
        A1 = csc_matrix((list(A1.values()), tuple(zip(*A1.keys()))), shape=(self.n * self.m, self.n * self.m), dtype=np.float32)
        A2 = csc_matrix((list(A2.values()), tuple(zip(*A2.keys()))), shape=(self.n * self.m, self.n * self.m), dtype=np.float32)

        i, j = np.meshgrid(np.arange(self.n), np.arange(self.m), indexing='ij')
        nbi_mask = (i != self.n - 1) & (j != 0) & (j != self.m - 1)
        non_boundary_indices = np.where(nbi_mask)
        nbi = i[non_boundary_indices] * self.m + j[non_boundary_indices]
        dt = self.tt[-1, 0, 0] - self.tt[-2, 0, 0]

        A = A0 + A1 + A2

        A1_LU_sp = splu(identity(nbi.shape[0], format="csc") - self.adi_param * dt * A1[:, nbi][nbi, :])
        A2_LU_sp = splu(identity(nbi.shape[0], format="csc") - self.adi_param * dt * A2[:, nbi][nbi, :])

        b0_prev, b1_prev, b2_prev = None, None, None
        b0_curr, b1_curr, b2_curr = None, None, None

        for t in tqdm(range(-2, -self.tt.shape[0] - 1, -1), disable=not self.verbose):
            if b0_prev is None:
                b0_prev = np.zeros(self.n * self.m, dtype=np.float32)
                b0_prev += A0[:, ::self.m] @ self.grid[t + 1, :, 0] # S=0
                b0_prev += A0[:, -self.m:] @ self.grid[t + 1, -1, :] # V = V_max
                b0_prev -= (A0[:, 0] * self.grid[t + 1, -1, 0]).toarray().reshape(-1) # intersection
            else:
                b0_prev = b0_curr
            if b1_prev is None:
                b1_prev = np.zeros(self.n * self.m, dtype=np.float32)
                b1_prev += A1[:, ::self.m] @ self.grid[t + 1, :, 0]
                b1_prev += A1[:, -self.m:] @ self.grid[t + 1, -1, :]
                b1_prev[self.m - 1::self.m] += (self.rd - self.rf) * self.xx[t + 1, :, -1] * (0.0 - self.grid[t + 1, :, -2]) / (self.barrier - self.xx[t + 1, :, -2])
                b1_prev -= (A1[:, 0] * self.grid[t + 1, -1, 0]).toarray().reshape(-1)
            else:
                b1_prev = b1_curr
            if b2_prev is None:
                b2_prev = np.zeros(self.n * self.m, dtype=np.float32)
                b2_prev += A2[:, ::self.m] @ self.grid[t + 1, :, 0]
                b2_prev += A2[:, -self.m:] @ self.grid[t + 1, -1, :]
                b2_prev -= (A2[:, 0] * self.grid[t + 1, -1, 0]).toarray().reshape(-1)
            else:
                b2_prev = b2_curr

            b0_curr = np.zeros(self.n * self.m, dtype=np.float32)
            b1_curr = np.zeros(self.n * self.m, dtype=np.float32)
            b2_curr = np.zeros(self.n * self.m, dtype=np.float32)

            b0_curr += A0[:, ::self.m] @ self.grid[t, :, 0]
            b0_curr += A0[:, -self.m:] @ self.grid[t, -1, :]
            b0_curr -= (A0[:, 0] * self.grid[t, -1, 0]).toarray().reshape(-1)

            b1_curr += A1[:, ::self.m] @ self.grid[t, :, 0]
            b1_curr += A1[:, -self.m:] @ self.grid[t, -1, :]
            b1_curr[self.m-1::self.m] += (self.rd - self.rf) * self.xx[t, :, -1] * np.maximum(-10.0, (0.0 - self.grid[t + 1, :, -2]) / (self.barrier - self.xx[t + 1, :, -2]))
            b1_curr -= (A1[:, 0] * self.grid[t, -1, 0]).toarray().reshape(-1)

            b2_curr += A2[:, ::self.m] @ self.grid[t, :, 0]
            b2_curr += A2[:, -self.m:] @ self.grid[t, -1, :]
            b2_curr -= (A2[:, 0] * self.grid[t, -1, 0]).toarray().reshape(-1)

            b_prev = b0_prev + b1_prev + b2_prev

            y0 = self.grid[t + 1, :-1, 1:-1].reshape(-1, 1) + dt * (A[:, nbi][nbi, :] @ self.grid[t + 1, :-1, 1:-1].reshape(-1, 1) + b_prev[nbi].reshape(-1, 1))
            y0 = y0.reshape(-1, 1)

            y1 = A1_LU_sp.solve(
                y0 - self.adi_param * dt * (A1[:, nbi][nbi, :] @ self.grid[t + 1, :-1, 1:-1].reshape(-1, 1) + b1_prev[nbi].reshape(-1, 1) - b1_curr[nbi].reshape(-1, 1))
            )

            y2 = A2_LU_sp.solve(
                y1 - self.adi_param * dt * (A2[:, nbi][nbi, :] @ self.grid[t + 1, :-1, 1:-1].reshape(-1, 1) + b2_prev[nbi].reshape(-1, 1) - b2_curr[nbi].reshape(-1, 1))
            )
            if self.corr:
                y0_hat = y0 + self.adi_param * dt * (A0[:, nbi][nbi, :] @ y2.reshape(-1, 1) + b0_curr[nbi].reshape(-1, 1) - (A0[:, nbi][nbi, :] @ self.grid[t + 1, :-1, 1:-1].reshape(-1, 1) + b0_prev[nbi].reshape(-1, 1)))
                y1_hat = A1_LU_sp.solve(
                    y0_hat - self.adi_param * dt * (A1[:, nbi][nbi, :] @ self.grid[t + 1, :-1, 1:-1].reshape(-1, 1) + b1_prev[nbi].reshape(-1, 1) - b1_curr[nbi].reshape(-1, 1))
                )
                y2_hat = A2_LU_sp.solve(
                    y1_hat - self.adi_param * dt * (A2[:, nbi][nbi, :] @ self.grid[t + 1, :-1, 1:-1].reshape(-1, 1) + b2_prev[nbi].reshape(-1, 1) - b2_curr[nbi].reshape(-1, 1))
                )
                self.grid[t, :-1, 1:-1] = y2_hat.reshape(self.n - 1, self.m - 2)
            else:
                self.grid[t, :-1, 1:-1] = y2.reshape(self.n - 1, self.m - 2)
        self.solved = True

if __name__ == "__main__":
    import cProfile
    import pstats
    import io
    profiler = cProfile.Profile()
    profiler.enable()

    K = 10
    B = 16
    S_0 = 12

    config = PDESolverConfig(
        underlier_price_grid=np.array([]),
        time_grid=np.linspace(0, 0.65125, 101),
        strike=K / K,
        barrier=B / K,
        interest_rate=0.054,
        volatility=0.34744,
        corr=-0.5,
        kappa=2.0,
        variance_theta=0.04,
        sigma=0.2,
        adi_param=0.5,
    )
    solver = ADIBarrierUpAndOutCallPDE(config)
    solver.solve()
    points = [
        [0.04, 0.64125, S_0 / K],
        [0.04, 0.54125, S_0 / K],
        [0.04, 0.44125, S_0 / K],
    ]
    print(K * solver.price(points))
    print(solver.delta(points))
    print(solver.vega(points))
    print(solver.theta(points))

    config = PDESolverConfig(
        underlier_price_grid=np.array([]),
        time_grid=np.linspace(0, 0.65125, 101),
        strike=K,
        barrier=B,
        interest_rate=0.054,
        volatility=0.34744,
        corr=-0.5,
        kappa=2.0,
        variance_theta=0.04,
        sigma=0.2,
        adi_param=0.5,
    )
    solver = ADIBarrierUpAndOutCallPDE(config)
    solver.solve()
    points = [
        [0.04, 0.64125, S_0],
        [0.04, 0.54125, S_0],
        [0.04, 0.44125, S_0],
    ]
    print(solver.price(points))
    print(solver.delta(points))
    print(solver.vega(points))
    print(solver.theta(points))

    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
    stats.print_stats()
    print(stream.getvalue())
