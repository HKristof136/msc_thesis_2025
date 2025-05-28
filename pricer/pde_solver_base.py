import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import splu
from scipy.sparse import identity, csc_matrix
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
from pricer.config_base import PDESolverConfig, DefaultConfig


class CrankNicolsonPDESolver:
    type = "pde_grid"
    underlier_lower_boundary = lambda self: 0.0
    underlier_upper_boundary = lambda self: 0.0
    payoff = lambda self: 0.0

    def __init__(self, config: PDESolverConfig = DefaultConfig.pde_solver):
        self.solved = False
        self.config = config
        self.verbose = config.verbose
        self.max_t = np.max(config.time_grid)
        self.strike = config.strike
        self.r = config.interest_rate
        self.sigma = config.volatility

        self.t, self.x = np.meshgrid(config.time_grid, config.underlier_price_grid)

        self.grid = np.zeros(self.x.shape)
        self.grid[0, :] = self.underlier_lower_boundary()
        self.grid[-1, :] = self.underlier_upper_boundary()
        self.grid[:, -1] = self.payoff()

    def solve(self):
        if self.config.vectorize_solver:
            solver_func = np.vectorize(self._solve)
        else:
            solver_func = self._solve
        solver_func()

    def _solve(self):
        if self.solved:
            print("Already solved")
            return None
        dt = self.config.time_grid[1] - self.config.time_grid[0]

        calc_A = True
        A = np.zeros(
            (
                self.config.underlier_price_grid.shape[0],
                self.config.underlier_price_grid.shape[0],
            )
        )
        B = np.zeros(
            (
                self.config.underlier_price_grid.shape[0],
                self.config.underlier_price_grid.shape[0],
            )
        )

        iterator = (
            tqdm(range(-2, -self.t.shape[1] - 1, -1))
            if self.verbose
            else range(-2, -self.t.shape[1] - 1, -1)
        )

        for i in iterator:
            if calc_A:
                for j in range(1, self.x.shape[0] - 1):
                    a_j = 0.25 * dt * (self.sigma**2 * j**2 - self.r * j)
                    b_j = -0.5 * dt * (self.sigma**2 * j**2 + self.r)
                    c_j = 0.25 * dt * (self.sigma**2 * j**2 + self.r * j)

                    A[j, j - 1 : j + 2] = [-a_j, 1 - b_j, -c_j]
                    B[j, j - 1 : j + 2] = [a_j, 1 + b_j, c_j]

                a_1 = A[1, 0]
                c_m_1 = A[-2, -1]
                A = A[1:-1, 1:-1]
                B = B[1:-1, :]
                lu_A, piv_A = lu_factor(A)
                calc_A = False

            b = B @ self.grid[:, i + 1].copy().reshape(-1, 1)

            b[0] -= a_1 * self.grid[0, i]
            b[-1] -= c_m_1 * self.grid[-1, i]
            self.grid[1:-1, i] = lu_solve((lu_A, piv_A), b).flatten()

        self.solved = True

    def price(self, points):
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        if not self.solved:
            self.solve()
        interp_func = RegularGridInterpolator(
            (self.config.underlier_price_grid, self.config.time_grid),
            self.grid,
            method="linear",
        )
        interp_points = np.column_stack([points[:, 0], self.max_t - points[:, 1]])
        return interp_func(interp_points)

    def _calc_greek(self, points, greek, order, precision):
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        if not self.solved:
            self.solve()
        if greek in ["delta", "theta"]:
            points_up = points.copy()
            if greek == "delta":
                points_up[:, 0] += precision
            else:
                points_up[:, 1] += precision
            price_up = self.price(points_up)

            points_down = points.copy()
            if greek == "delta":
                points_down[:, 0] -= precision
            else:
                points_down[:, 1] -= precision
            price_down = self.price(points_down)

            if order == "first":
                return (price_up - price_down) / (2 * precision)
            else:
                return (price_up - 2 * self.price(points) + price_down) / precision**2
        else:
            up_config = PDESolverConfig(
                underlier_price_grid=self.config.underlier_price_grid,
                time_grid=self.config.time_grid,
                strike=self.strike,
                interest_rate=self.r + precision if greek == "rho" else self.r,
                volatility=self.sigma + precision if greek == "vega" else self.sigma,
                barrier=self.config.barrier,
                verbose=self.verbose,
            )
            price_up = self.__class__(up_config).price(points)

            down_config = PDESolverConfig(
                underlier_price_grid=self.config.underlier_price_grid,
                time_grid=self.config.time_grid,
                strike=self.strike,
                interest_rate=self.r - precision if greek == "rho" else self.r,
                volatility=self.sigma - precision if greek == "vega" else self.sigma,
                barrier=self.config.barrier,
                verbose=self.verbose,
            )
            price_down = self.__class__(down_config).price(points)
            if order == "first":
                return (price_up - price_down) / (2 * precision)
            else:
                return (price_up - 2 * self.price(points) + price_down) / precision**2

    def delta(self, points):
        return self._calc_greek(points, greek="delta", order="first", precision=0.001)

    def gamma(self, points):
        return self._calc_greek(points, greek="delta", order="second", precision=0.001)

    def theta(self, points):
        return self._calc_greek(points, greek="theta", order="first", precision=0.001)

    def vega(self, points):
        return self._calc_greek(points, greek="vega", order="first", precision=0.001)

    def rho(self, points):
        return self._calc_greek(points, greek="rho", order="first", precision=0.001)

    @staticmethod
    def TDMA(a, b, c, d):
        n = len(d)
        w = np.zeros(n - 1, float)
        g = np.zeros(n, float)
        p = np.zeros(n, float)

        w[0] = c[0] / b[0]
        g[0] = d[0] / b[0]

        for i in range(1, n - 1):
            w[i] = c[i] / (b[i] - a[i - 1] * w[i - 1])
        for i in range(1, n):
            g[i] = (d[i] - a[i - 1] * g[i - 1]) / (b[i] - a[i - 1] * w[i - 1])
        p[n - 1] = g[n - 1]
        for i in range(n - 1, 0, -1):
            p[i - 1] = g[i - 1] - w[i - 1] * p[i]
        return p


class BarrierUpPDE(CrankNicolsonPDESolver):
    input_names = [
        "underlier_price_grid", "time_grid",
        "strike", "interest_rate", "volatility", "barrier"]

    def __init__(self, config: PDESolverConfig = DefaultConfig.pde_solver_barrier_call):
        self.barrier = config.barrier
        super().__init__(config)
        self.f = np.zeros_like(self.config.time_grid, dtype=np.float32)

    def _solve(self):
        if self.solved:
            print("Already solved")
            return None
        dt = self.config.time_grid[1] - self.config.time_grid[0]

        reduced_j = np.min(np.nonzero(self.config.underlier_price_grid >= self.barrier))
        reduced_scope = self.config.underlier_price_grid[: reduced_j + 1]
        A = np.zeros((reduced_scope.shape[0], reduced_scope.shape[0]))
        B = np.zeros((reduced_scope.shape[0], reduced_scope.shape[0]))

        iterator = (
            tqdm(range(-2, -self.t.shape[1] - 1, -1))
            if self.verbose
            else range(-2, -self.t.shape[1] - 1, -1)
        )

        for j in range(1, reduced_scope.shape[0] - 1):
            a_j = 0.25 * dt * (self.sigma ** 2 * j ** 2 - self.r * j)
            b_j = -0.5 * dt * (self.sigma ** 2 * j ** 2 + self.r)
            c_j = 0.25 * dt * (self.sigma ** 2 * j ** 2 + self.r * j)

            A[j, j - 1: j + 2] = [-a_j, 1 - b_j, -c_j]
            B[j, j - 1: j + 2] = [a_j, 1 + b_j, c_j]

        a_1 = A[1, 0]
        c_m_1 = A[-2, -1]
        A = A[1:-1, 1:-1]
        B = B[1:-1, :]
        lu_A, piv_A = lu_factor(A)

        for i in iterator:
            b = B @ self.grid[: reduced_j + 1, i + 1].copy().reshape(-1, 1)

            b[0] -= a_1 * self.grid[0, i]
            if "In" in self.__class__.__name__:
                b[-1] -= c_m_1 * max(
                    min(10.0, (self.f[i+1] - self.grid[reduced_j - 1, i + 1]) / (
                            self.barrier - reduced_scope[-2]
                    )) * (reduced_scope[-1] - reduced_scope[-2]) + self.grid[reduced_j - 1, i + 1],
                    self.f[i+1]
                )
            else:
                b[-1] -= c_m_1 * min(
                    max(-10.0 , (self.f[i+1] - self.grid[reduced_j - 1, i + 1]) / (
                        self.barrier - reduced_scope[-2]
                    )) * (reduced_scope[-1] - reduced_scope[-2]) + self.grid[reduced_j - 1, i + 1],
                    self.f[i+1]
                )
            self.grid[1:reduced_j, i] = lu_solve((lu_A, piv_A), b).flatten()

        self.solved = True


class BarrierDownPDE(CrankNicolsonPDESolver):
    input_names = [
        "underlier_price_grid", "time_grid",
        "strike", "interest_rate", "volatility", "barrier"]

    def __init__(self, config: PDESolverConfig = DefaultConfig.pde_solver_barrier_call):
        self.barrier = config.barrier
        super().__init__(config)
        self.f = np.zeros_like(self.config.time_grid, dtype=np.float32)

    def solve(self):
        if self.solved:
            print("Already solved")
            return None
        dt = self.config.time_grid[1] - self.config.time_grid[0]

        reduced_j = np.max(np.nonzero(self.config.underlier_price_grid <= self.barrier))
        reduced_scope = self.config.underlier_price_grid[reduced_j:]
        A = np.zeros((reduced_scope.shape[0], reduced_scope.shape[0]))
        B = np.zeros((reduced_scope.shape[0], reduced_scope.shape[0]))

        iterator = (
            tqdm(range(-2, -self.t.shape[1] - 1, -1))
            if self.verbose
            else range(-2, -self.t.shape[1] - 1, -1)
        )

        for j in range(1, reduced_scope.shape[0] - 1):
            a_j = 0.25 * dt * (self.sigma ** 2 * (reduced_j + 1 + j) ** 2 - self.r * (reduced_j + 1 + j))
            b_j = -0.5 * dt * (self.sigma ** 2 * (reduced_j + 1 + j) ** 2 + self.r)
            c_j = 0.25 * dt * (self.sigma ** 2 * (reduced_j + 1 + j) ** 2 + self.r * (reduced_j + 1 + j))

            A[j, j - 1: j + 2] = [-a_j, 1 - b_j, -c_j]
            B[j, j - 1: j + 2] = [a_j, 1 + b_j, c_j]

        a_1 = A[1, 0]
        c_m_1 = A[-2, -1]
        A = A[1:-1, 1:-1]
        B = B[1:-1, :]
        lu_A, piv_A = lu_factor(A)

        for i in iterator:
            b = B @ self.grid[reduced_j:, i + 1].copy().reshape(-1, 1)

            if "In" in self.__class__.__name__:
                b[0] -= a_1 * max(
                    max(-10.0, (self.f[i+1] - self.grid[reduced_j + 1, i + 1]) / (
                            self.barrier - reduced_scope[1]
                    )) * (reduced_scope[0] - reduced_scope[1]) + self.grid[reduced_j + 1, i + 1],
                    self.f[i+1]
                )
            else:
                b[0] -= a_1 * min(
                    min(10.0 , (self.f[i+1] - self.grid[reduced_j + 1, i + 1]) / (
                        self.barrier - reduced_scope[1]
                    )) * (reduced_scope[0] - reduced_scope[1]) + self.grid[reduced_j + 1, i + 1],
                    self.f[i+1]
                )

            b[-1] -= c_m_1 * self.grid[-1, i]
            self.grid[reduced_j + 1 : -1, i] = lu_solve((lu_A, piv_A), b).flatten()

        self.solved = True


class AmericanOptionPDE(CrankNicolsonPDESolver):
    def solve(self):
        if self.solved:
            print("Already solved")
            return None
        dt = self.config.time_grid[1] - self.config.time_grid[0]

        calc_A = True
        A = np.zeros(
            (
                self.config.underlier_price_grid.shape[0],
                self.config.underlier_price_grid.shape[0],
            )
        )
        B = np.zeros(
            (
                self.config.underlier_price_grid.shape[0],
                self.config.underlier_price_grid.shape[0],
            )
        )

        iterator = (
            tqdm(range(-2, -self.t.shape[1] - 1, -1))
            if self.verbose
            else range(-2, -self.t.shape[1] - 1, -1)
        )

        for i in iterator:
            if calc_A:
                for j in range(1, self.x.shape[0] - 1):
                    a_j = 0.25 * dt * (self.sigma**2 * j**2 - self.r * j)
                    b_j = -0.5 * dt * (self.sigma**2 * j**2 + self.r)
                    c_j = 0.25 * dt * (self.sigma**2 * j**2 + self.r * j)

                    A[j, j - 1 : j + 2] = [-a_j, 1 - b_j, -c_j]
                    B[j, j - 1 : j + 2] = [a_j, 1 + b_j, c_j]

                a_1 = A[1, 0]
                c_m_1 = A[-2, -1]
                A = A[1:-1, 1:-1]
                B = B[1:-1, :]
                lu_A, piv_A = lu_factor(A)
                calc_A = False

            b = B @ self.grid[:, i + 1].copy().reshape(-1, 1)

            b[0] -= a_1 * self.grid[0, i]
            b[-1] -= c_m_1 * self.grid[-1, i]
            self.grid[1:-1, i] = lu_solve((lu_A, piv_A), b).flatten()

        self.solved = True


class AmericanBarrierUpPDE(BarrierUpPDE):
    def solve(self):
        if self.solved:
            print("Already solved")
            return None
        dt = self.config.time_grid[1] - self.config.time_grid[0]

        calc_A = True

        reduced_j = np.min(np.nonzero(self.config.underlier_price_grid >= self.barrier))
        reduced_scope = self.config.underlier_price_grid[: reduced_j + 1]
        A = np.zeros((reduced_scope.shape[0], reduced_scope.shape[0]))
        B = np.zeros((reduced_scope.shape[0], reduced_scope.shape[0]))

        iterator = (
            tqdm(range(-2, -self.t.shape[1] - 1, -1))
            if self.verbose
            else range(-2, -self.t.shape[1] - 1, -1)
        )

        for i in iterator:
            if calc_A:
                for j in range(1, reduced_scope.shape[0] - 1):
                    a_j = 0.25 * dt * (self.sigma**2 * j**2 - self.r * j)
                    b_j = -0.5 * dt * (self.sigma**2 * j**2 + self.r)
                    c_j = 0.25 * dt * (self.sigma**2 * j**2 + self.r * j)

                    A[j, j - 1 : j + 2] = [-a_j, 1 - b_j, -c_j]
                    B[j, j - 1 : j + 2] = [a_j, 1 + b_j, c_j]

                a_1 = A[1, 0]
                c_m_1 = A[-2, -1]
                A = A[1:-1, 1:-1]
                # A[-1, -1] += c_m_1 * (
                #     (reduced_scope[-1] - reduced_scope[-2])
                #     / (reduced_scope[-2] - self.barrier)
                # )
                lu_A, piv_A = lu_factor(A)
                B = B[1:-1, :]
                calc_A = False

            b = B @ self.grid[: reduced_j + 1, i + 1].copy().reshape(-1, 1)

            b[0] -= a_1 * self.grid[0, i]
            b[-1] -= c_m_1 * self.grid[reduced_j, i]
            self.grid[1:reduced_j, i] = lu_solve((lu_A, piv_A), b).flatten()

        self.solved = True


class AmericanBarrierDownPDE(BarrierDownPDE):
    def solve(self):
        if self.solved:
            print("Already solved")
            return None
        dt = self.config.time_grid[1] - self.config.time_grid[0]

        calc_A = True

        reduced_j = np.max(np.nonzero(self.config.underlier_price_grid <= self.barrier))
        reduced_scope = self.config.underlier_price_grid[reduced_j:]
        A = np.zeros((reduced_scope.shape[0], reduced_scope.shape[0]))
        B = np.zeros((reduced_scope.shape[0], reduced_scope.shape[0]))

        iterator = (
            tqdm(range(-2, -self.t.shape[1] - 1, -1))
            if self.verbose
            else range(-2, -self.t.shape[1] - 1, -1)
        )

        for i in iterator:
            if calc_A:
                for j in range(1, reduced_scope.shape[0] - 1):
                    a_j = (
                        0.25
                        * dt
                        * (
                            self.sigma**2 * (reduced_j + 1 + j) ** 2
                            - self.r * (reduced_j + 1 + j)
                        )
                    )
                    b_j = (
                        -0.5 * dt * (self.sigma**2 * (reduced_j + 1 + j) ** 2 + self.r)
                    )
                    c_j = (
                        0.25
                        * dt
                        * (
                            self.sigma**2 * (reduced_j + 1 + j) ** 2
                            + self.r * (reduced_j + 1 + j)
                        )
                    )

                    A[j, j - 1 : j + 2] = [-a_j, 1 - b_j, -c_j]
                    B[j, j - 1 : j + 2] = [a_j, 1 + b_j, c_j]

                a_1 = A[1, 0]
                c_m_1 = A[-2, -1]
                A = A[1:-1, 1:-1]
                A[0, 0] += a_1 * (
                    (reduced_scope[1] - reduced_scope[0])
                    / (reduced_scope[0] - self.barrier)
                )
                B = B[1:-1, :]
                calc_A = False

            b = B @ self.grid[reduced_j:, i + 1].copy().reshape(-1, 1)

            b[0] -= a_1 * (
                    (reduced_scope[1] - reduced_scope[0])
                    / (reduced_scope[0] - self.barrier)
                ) * - self.f[i]
            b[-1] -= c_m_1 * self.grid[-1, i]
            omega = 1.5
            price_vals = self.grid[reduced_j + 1 : -1, i + 1].copy()
            error = 1
            while error > 1e-5:
                error = 0
                temp = price_vals.copy()
                for j in range(len(price_vals)):
                    temp[j] = (
                        price_vals[j]
                        + omega * (b[j, 0] - np.dot(A[j, :], temp)) / A[j, j]
                    )
                    temp[j] = np.max([self.grid[j, -1], temp[j]])
                    error += (temp[j] - price_vals[j]) ** 2
                price_vals = temp.copy()

            self.grid[reduced_j + 1 : -1, i] = price_vals

        self.solved = True

class ADISolver:
    type = "adi"
    input_names = ["strike", "interest_rate", "corr", "kappa", "variance_theta", "sigma"]

    def __init__(self, config: PDESolverConfig = DefaultConfig.pde_solver):
        self.solved = False
        self.config = config
        self.verbose = config.verbose
        self.adi_param = config.adi_param

        self.n, self.m = config.n, config.m
        self.max_t = np.max(config.time_grid)
        self.strike = config.strike
        self.rd = config.interest_rate
        self.rf = config.foreign_interest_rate
        self.corr = config.corr
        self.k = config.kappa
        self.th = config.variance_theta
        self.sigma = config.sigma

        c = self.strike / 10
        non_uniform_s = np.zeros(self.m)
        for i in range(self.m):
            dk_i = (1 / self.m) * (math.asinh((3 * self.strike - self.strike) / c) - math.asinh(-self.strike / c))
            k_i = math.asinh(-self.strike / c) + i * dk_i
            non_uniform_s[i] = abs(round(self.strike + c * np.sinh(k_i), 5))

        V = 3.0 # 5 * self.sigma
        d = V / 500

        non_uniform_v = np.zeros(self.n)
        for j in range(self.n):
            dk_i = (1 / self.n) * math.asinh(V / d)
            non_uniform_v[j] = abs(d * math.sinh(j * dk_i))

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
        self.grid[-1, :, :] = np.maximum(self.xx[-1, :, :] - self.strike, 0)

        # s=0 boundary condidition
        self.grid[:, :, 0] = 0.0

        # v = V boundary condition
        self.grid[:, -1, :] = np.maximum(self.xx[:, -1, :] * np.exp(-self.rf * self.tt[:, -1, :]) - self.strike * (np.exp(-self.rd * self.tt[:, -1, :])), 0)


    def solve(self):
        A0 = {}
        A1 = {}
        A2 = {}

        for i in range(0, self.n - 1):
            for j in range(1, self.m):
                idx = i * self.m + j
                if not i:
                    for o in [-1, 0, 1]:
                        A1[(idx, idx + o)] = A1.get((idx, idx + o), 0.0) + int(j != (self.m - 1)) * ((self.rd - self.rf) * self.xx[0, 0, j] * self.beta_coef(self.dxx, j, o)) #- int(o == 0) * self.rd / 2

                    for o in [0, 1, 2]:
                        A2[(idx, idx + o * self.m)] = A2.get((idx, idx + o * self.m), 0.0) + self.k * (self.th - self.vv[0, i, 0]) * self.gamma_coef(self.dvv, i, o) - int(o == 0) * self.rd# / 2

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
                        )# - int(o == 0) * self.rd / 2

                    if self.vv[0, i, 0] <= 1:
                        for o in [-1, 0, 1]:
                            A2[(idx, idx + o * self.m)] = A2.get((idx, idx + o * self.m), 0.0) + (
                                    0.5 * self.sigma ** 2 * self.vv[0, i, 0] * self.delta_coef(self.dvv, i, o) + self.k * (self.th - self.vv[0, i, 0]) * self.beta_coef(self.dvv, i, o)
                            ) - int(o == 0) * self.rd# / 2

                    else:
                        for o in [-1, 0, 1]:
                            A2[(idx, idx + o * self.m)] = A2.get((idx, idx + o * self.m), 0.0) + (
                                    0.5 * self.sigma ** 2 * self.vv[0, i, 0] * self.delta_coef(self.dvv, i, o)
                            ) - int(o == 0) * self.rd# / 2

                        for o in [-2, -1, 0]:
                            A2[(idx, idx + o * self.m)] = A2.get((idx, idx + o * self.m), 0.0) + (
                                    self.k * (self.th - self.vv[0, i, 0]) * self.alpha_coef(self.dvv, i, o)
                            )
        self.A0 = csc_matrix((list(A0.values()), tuple(zip(*A0.keys()))), shape=(self.n * self.m, self.n * self.m), dtype=np.float32)
        self.A1 = csc_matrix((list(A1.values()), tuple(zip(*A1.keys()))), shape=(self.n * self.m, self.n * self.m), dtype=np.float32)
        self.A2 = csc_matrix((list(A2.values()), tuple(zip(*A2.keys()))), shape=(self.n * self.m, self.n * self.m), dtype=np.float32)

        i, j = np.meshgrid(np.arange(self.n), np.arange(self.m), indexing='ij')
        nbi_mask = (i != self.n - 1) & (j != 0)
        non_boundary_indices = np.where(nbi_mask)
        self.nbi = i[non_boundary_indices] * self.m + j[non_boundary_indices]
        dt = self.tt[-1, 0, 0] - self.tt[-2, 0, 0]

        A = self.A0 + self.A1 + self.A2

        self.A1_LU_sp = splu(identity(self.nbi.shape[0], format="csc") - self.adi_param * dt * self.A1[:, self.nbi][self.nbi, :])
        self.A2_LU_sp = splu(identity(self.nbi.shape[0], format="csc") - self.adi_param * dt * self.A2[:, self.nbi][self.nbi, :])

        b0_prev, b1_prev, b2_prev = None, None, None
        b0_curr, b1_curr, b2_curr = None, None, None

        for t in tqdm(range(-2, -self.tt.shape[0] - 1, -1), disable=not self.verbose):
            if b0_prev is None:
                b0_prev = np.zeros(self.n * self.m, dtype=np.float32)
                b0_prev += self.A0[:, ::self.m] @ self.grid[t + 1, :, 0]
                b0_prev += self.A0[:, -self.m:] @ self.grid[t + 1, -1, :]
                b0_prev -= (self.A0[:, 0] * self.grid[t + 1, -1, 0]).toarray().reshape(-1)
            else:
                b0_prev = b0_curr
            if b1_prev is None:
                b1_prev = np.zeros(self.n * self.m, dtype=np.float32)
                b1_prev += self.A1[:, ::self.m] @ self.grid[t + 1, :, 0]
                b1_prev += self.A1[:, -self.m:] @ self.grid[t + 1, -1, :]
                b1_prev[self.m - 1::self.m] += (self.rd - self.rf) * self.xx[t + 1, :, -1] * np.exp(-self.rf * self.tt[t + 1, :, -1])
                b1_prev -= (self.A1[:, 0] * self.grid[t + 1, -1, 0]).toarray().reshape(-1)
            else:
                b1_prev = b1_curr
            if b2_prev is None:
                b2_prev = np.zeros(self.n * self.m, dtype=np.float32)
                b2_prev += self.A2[:, ::self.m] @ self.grid[t + 1, :, 0]
                b2_prev += self.A2[:, -self.m:] @ self.grid[t + 1, -1, :]
                b2_prev -= (self.A2[:, 0] * self.grid[t + 1, -1, 0]).toarray().reshape(-1)
            else:
                b2_prev = b2_curr

            b0_curr = np.zeros(self.n * self.m, dtype=np.float32)
            b1_curr = np.zeros(self.n * self.m, dtype=np.float32)
            b2_curr = np.zeros(self.n * self.m, dtype=np.float32)

            b0_curr += self.A0[:, ::self.m] @ self.grid[t, :, 0]
            b0_curr += self.A0[:, -self.m:] @ self.grid[t, -1, :]
            b0_curr -= (self.A0[:, 0] * self.grid[t, -1, 0]).toarray().reshape(-1)

            b1_curr += self.A1[:, ::self.m] @ self.grid[t, :, 0]
            b1_curr += self.A1[:, -self.m:] @ self.grid[t, -1, :]
            b1_curr[self.m-1::self.m] += (self.rd - self.rf) * self.xx[t, :, -1] * np.exp(-self.rf * self.tt[t, :, -1])
            b1_curr -= (self.A1[:, 0] * self.grid[t, -1, 0]).toarray().reshape(-1)

            b2_curr += self.A2[:, ::self.m] @ self.grid[t, :, 0]
            b2_curr += self.A2[:, -self.m:] @ self.grid[t, -1, :]
            b2_curr -= (self.A2[:, 0] * self.grid[t, -1, 0]).toarray().reshape(-1)

            b_prev = b0_prev + b1_prev + b2_prev

            y0 = self.grid[t + 1, :-1, 1:].reshape(-1, 1) + dt * (A[:, self.nbi][self.nbi, :] @ self.grid[t + 1, :-1, 1:].reshape(-1, 1) + b_prev[self.nbi].reshape(-1, 1))
            y0 = y0.reshape(-1, 1)

            y1 = self.A1_LU_sp.solve(
                y0 - self.adi_param * dt * (self.A1[:, self.nbi][self.nbi, :] @ self.grid[t + 1, :-1, 1:].reshape(-1, 1) + b1_prev[self.nbi].reshape(-1, 1) - b1_curr[self.nbi].reshape(-1, 1))
            )

            y2 = self.A2_LU_sp.solve(
                y1 - self.adi_param * dt * (self.A2[:, self.nbi][self.nbi, :] @ self.grid[t + 1, :-1, 1:].reshape(-1, 1) + b2_prev[self.nbi].reshape(-1, 1) - b2_curr[self.nbi].reshape(-1, 1))
            )
            if self.corr:
                y0_hat = y0 + self.adi_param * dt * (self.A0[:, self.nbi][self.nbi, :] @ y2.reshape(-1, 1) + b0_curr[self.nbi].reshape(-1, 1) - (self.A0[:, self.nbi][self.nbi, :] @ self.grid[t + 1, :-1, 1:].reshape(-1, 1) + b0_prev[self.nbi].reshape(-1, 1)))
                y1_hat = self.A1_LU_sp.solve(
                    y0_hat - self.adi_param * dt * (self.A1[:, self.nbi][self.nbi, :] @ self.grid[t + 1, :-1, 1:].reshape(-1, 1) + b1_prev[self.nbi].reshape(-1, 1) - b1_curr[self.nbi].reshape(-1, 1))
                )
                y2_hat = self.A2_LU_sp.solve(
                    y1_hat - self.adi_param * dt * (self.A2[:, self.nbi][self.nbi, :] @ self.grid[t + 1, :-1, 1:].reshape(-1, 1) + b2_prev[self.nbi].reshape(-1, 1) - b2_curr[self.nbi].reshape(-1, 1))
                )
                self.grid[t, :-1, 1:] = y2_hat.reshape(self.n - 1, self.m - 1)
            else:
                self.grid[t, :-1, 1:] = y2.reshape(self.n - 1, self.m - 1)
        self.solved = True

    @staticmethod
    def alpha_coef(arr, i, offset):
        if offset == -2:
            return arr[i] / (arr[i - 1] * (arr[i - 1] + arr[i]))
        elif offset == -1:
            return (-arr[i - 1] - arr[i]) / (arr[i - 1] * arr[i])
        elif offset == 0:
            return (arr[i - 1] + 2 * arr[i]) / (arr[i] * (arr[i - 1] + arr[i]))
        return None

    @staticmethod
    def beta_coef(arr, i, offset):
        if offset == -1:
            return (- arr[i + 1]) / (arr[i] * (arr[i] + arr[i + 1]))
        elif offset == 0:
            return (arr[i + 1] - arr[i]) / (arr[i] * arr[i + 1])
        elif offset == 1:
            return (arr[i]) / (arr[i + 1] * (arr[i] + arr[i + 1]))
        return None

    @staticmethod
    def gamma_coef(arr, i, offset):
        if offset == 0:
            return (-2 * arr[i + 1] - arr[i + 2]) / (arr[i + 1] * (arr[i + 1] + arr[i + 2]))
        elif offset == 1:
            return (arr[i + 1] + arr[i + 2]) / (arr[i + 1] * arr[i + 2])
        elif offset == 2:
            return (-arr[i + 1]) / (arr[i + 2] * (arr[i + 1] + arr[i + 2]))
        return None

    @staticmethod
    def delta_coef(arr, i , offset):
        if offset == -1:
            return 2 / (arr[i] * (arr[i] + arr[i + 1]))
        elif offset == 0:
            return (-2) / (arr[i] * arr[i + 1])
        elif offset == 1:
            return 2 / (arr[i + 1] * (arr[i] + arr[i + 1]))
        return None

    def price(self, points):
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        if not self.solved:
            self.solve()

        interp_points = np.column_stack([points[:, 0], self.max_t - points[:, 1], points[:, 2]])
        interp_func = RegularGridInterpolator(
                (self.vv[0, :, 0], self.tt[:, 0, 0], self.xx[0, 0, :]),
                np.moveaxis(self.grid[:, :, :], [0, 1, 2], [1, 0, 2]),
                method='linear',
        )
        return interp_func(interp_points)

    def _calc_greek(self, points, greek, order, precision, n_prec=26, m_prec=26):
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        if not self.solved:
            self.solve()
        if greek in ["delta", "theta", "vega"]:
            points_up = points.copy()
            if greek == "delta":
                points_up[:, 2] += precision
            elif greek == "theta":
                points_up[:, 1] += precision
            else:
                points_up[:, 0] += precision
            price_up = self.price(points_up)

            points_down = points.copy()
            if greek == "delta":
                points_down[:, 2] -= precision
            elif greek == "theta":
                points_down[:, 1] -= precision
            else:
                points_down[:, 0] -= precision
            price_down = self.price(points_down)

            if order == "first":
                return (price_up - price_down) / (2 * precision)
            else:
                return (price_up - 2 * self.price(points) + price_down) / precision**2
        else:
            up_config = PDESolverConfig(
                underlier_price_grid=self.config.underlier_price_grid,
                time_grid=np.linspace(self.config.time_grid[0], self.config.time_grid[-1], 51),
                variance_grid=self.config.variance_grid,
                strike=self.config.strike,
                interest_rate=self.config.interest_rate + precision if greek == "rho" else self.config.interest_rate,
                corr=self.config.corr + precision if greek == "partial_deriv_correlation" else self.config.corr,
                kappa=self.config.kappa + precision if greek == "partial_deriv_kappa" else self.config.kappa,
                variance_theta=self.config.variance_theta + precision if greek == "partial_deriv_theta" else self.config.variance_theta,
                sigma=self.config.sigma + precision if greek == "partial_deriv_sigma" else self.config.sigma,
                n=n_prec,
                m=m_prec,
                barrier=self.config.barrier,
                verbose=False
            )
            price_up = self.__class__(up_config).price(points)

            down_config = PDESolverConfig(
                underlier_price_grid=self.config.underlier_price_grid,
                time_grid=np.linspace(self.config.time_grid[0], self.config.time_grid[-1], 51),
                variance_grid=self.config.variance_grid,
                strike=self.config.strike,
                interest_rate=self.config.interest_rate - precision if greek == "rho" else self.config.interest_rate,
                corr=self.config.corr - precision if greek == "partial_deriv_correlation" else self.config.corr,
                kappa=self.config.kappa - precision if greek == "partial_deriv_kappa" else self.config.kappa,
                variance_theta=self.config.variance_theta - precision if greek == "partial_deriv_theta" else self.config.variance_theta,
                sigma=self.config.sigma - precision if greek == "partial_deriv_sigma" else self.config.sigma,
                n=n_prec,
                m=m_prec,
                barrier=self.config.barrier,
                verbose=False
            )
            price_down = self.__class__(down_config).price(points)
            if order == "first":
                return (price_up - price_down) / (2 * precision)
            else:
                return (price_up - 2 * self.price(points) + price_down) / precision**2

    def delta(self, points):
        return self._calc_greek(points, greek="delta", order="first", precision=0.001)

    def gamma(self, points):
        return self._calc_greek(points, greek="delta", order="second", precision=0.001)

    def theta(self, points):
        return self._calc_greek(points, greek="theta", order="first", precision=0.001)

    def vega(self, points):
        return self._calc_greek(points, greek="vega", order="first", precision=0.001)

    def rho(self, points):
        return self._calc_greek(points, greek="rho", order="first", precision=0.001)

    def partial_deriv_correlation(self, points):
        return self._calc_greek(points, greek="partial_deriv_correlation", order="first", precision=0.001)

    def partial_deriv_kappa(self, points):
        return self._calc_greek(points, greek="partial_deriv_kappa", order="first", precision=0.001)

    def partial_deriv_theta(self, points):
        return self._calc_greek(points, greek="partial_deriv_theta", order="first", precision=0.001)

    def partial_deriv_sigma(self, points):
        return self._calc_greek(points, greek="partial_deriv_sigma", order="first", precision=0.001)


if __name__ == "__main__":
    import time
    import pandas as pd

    s = time.perf_counter()
    config = PDESolverConfig(
        underlier_price_grid=np.array([]),
        time_grid=np.linspace(0, 1.01, 51, dtype=np.float32),
        strike=1.0,
        interest_rate=0.05,
        corr=-0.5,
        kappa=2.0,
        variance_theta=0.04,
        sigma=0.3,
        n=101,
        m=101,
    )
    solver = ADISolver(config)
    solver.solve()

    points = np.array([[0.04, 1.0, 1.1]])
    print(f"Price: {solver.price(points)}")
    print(f"delta: {solver.delta(points)}")
    print(f"theta: {solver.theta(points)}")
    print(f"vega: {solver.vega(points)}")
    print(f"rho {solver.rho(points)}")
    print(f"partial_correlation: {solver.partial_deriv_correlation(points)}")
    print(f"partial_kappa: {solver.partial_deriv_kappa(points)}")
    print(f"partial_theta: {solver.partial_deriv_theta(points)}")
    print(f"partial_sigma: {solver.partial_deriv_sigma(points)}")

    print(f"Took {time.perf_counter() - s:.5f} seconds")

    run_time_array = np.zeros((5, 5))

    for i, n in enumerate([5, 25, 50, 100, 250]):
        for j, m in enumerate([5, 25, 50, 100, 250]):
            config = PDESolverConfig(
                underlier_price_grid=np.array([]),
                time_grid=np.linspace(0, 1.01, 101, dtype=np.float32),
                strike=1.0,
                interest_rate=0.04,
                corr=-0.3,
                kappa=2.0,
                variance_theta=0.04,
                sigma=0.2,
                n=n,
                m=m,
                verbose=False
            )
            run_times = []
            for _ in range(10):
                print(i, n, j, m, _)
                s = time.perf_counter()
                call_price = ADISolver(config).solve()
                run_times.append(time.perf_counter() - s)
            run_time_array[i, j] = np.mean(run_times)
    pd.DataFrame(run_time_array, columns=[5, 25, 50, 100, 250], index=[5, 25, 50, 100, 250]).to_csv("adi_runtimes.csv")
