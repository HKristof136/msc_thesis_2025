import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import numpy as np
import pandas as pd
from numba import jit
from scipy.linalg import lu_factor, lu_solve, solve_banded
from scipy.sparse.linalg import splu, inv
from scipy.sparse import identity, csc_matrix, diags
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
        ds = self.config.underlier_price_grid[1] - self.config.underlier_price_grid[0]

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
                A[-1, -1] += c_m_1 * (
                    (reduced_scope[-1] - reduced_scope[-2])
                    / (reduced_scope[-2] - self.barrier)
                )
                B = B[1:-1, :]
                lu_A, piv_A = lu_factor(A)
                calc_A = False

            b = B @ self.grid[: reduced_j + 1, i + 1].copy().reshape(-1, 1)

            b[0] -= a_1 * self.grid[0, i]
            # b[-1] -= c_m_1 * self.grid[reduced_j, i]
            self.grid[1:reduced_j, i] = lu_solve((lu_A, piv_A), b).flatten()

        self.solved = True


class BarrierDownPDE(CrankNicolsonPDESolver):
    input_names = ["strike", "interest_rate", "volatility", "barrier"]

    def __init__(self, config: PDESolverConfig = DefaultConfig.pde_solver_barrier_call):
        self.barrier = config.barrier
        super().__init__(config)
        self.f = np.zeros_like(self.config.time_grid, dtype=np.float32)

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
                lu_A, piv_A = lu_factor(A)
                calc_A = False

            b = B @ self.grid[reduced_j:, i + 1].copy().reshape(-1, 1)

            # b[0] -= a_1 * (self.f[i] / alpha)
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
            # omega = 1.5
            # price_vals = self.grid[1:-1, i + 1].copy()
            # error = 1
            # while error > 1e-5:
            #     error = 0
            #     temp = price_vals.copy()
            #     for j in range(len(price_vals)):
            #         temp[j] = (
            #             price_vals[j]
            #             + omega * (b[j, 0] - np.dot(A[j, :], temp)) / A[j, j]
            #         )
            #         temp[j] = np.max([self.grid[j, -1], temp[j]])
            #         error += (temp[j] - price_vals[j]) ** 2
            #     price_vals = temp.copy()
            #
            # self.grid[1:-1, i] = price_vals
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
            # b[-1] -= c_m_1 * (
            #         (reduced_scope[-1] - reduced_scope[-2])
            #         / (reduced_scope[-2] - self.barrier)
            #     ) * - self.f[i]
            # omega = 1.75
            # price_vals = self.grid[1:reduced_j, i + 1].copy()
            # error = 1
            # while error > 1e-3:
            #     error = 0
            #     temp = price_vals.copy()
            #     for j in range(len(price_vals)):
            #         temp[j] = (
            #             price_vals[j]
            #             + omega * (b[j, 0] - np.dot(A[j, :], temp)) / A[j, j]
            #         )
            #         temp[j] = np.max([self.grid[j, -1], temp[j]])
            #         error += (temp[j] - price_vals[j]) ** 2
            #     price_vals = temp.copy()

            # self.grid[1:reduced_j, i] = price_vals
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


@jit(nopython=True)
def meshgrid(x, y, z):
    xx = np.empty(shape=(x.size, y.size, z.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size, z.size), dtype=y.dtype)
    zz = np.empty(shape=(x.size, y.size, z.size), dtype=z.dtype)
    for i in range(z.size):
        for j in range(y.size):
            for k in range(x.size):
                xx[i,j,k] = x[k]  # change to x[k] if indexing xy
                yy[i,j,k] = y[j]  # change to y[j] if indexing xy
                zz[i,j,k] = z[i]  # change to z[i] if indexing xy
    return zz, yy, xx

class DouglasADI:
    def __init__(self, config: PDESolverConfig = DefaultConfig.pde_solver):
        self.solved = False
        self.config = config
        self.verbose = config.verbose
        self.adi_param = config.adi_param

        self.n, self.m = config.variance_grid.shape[0], config.underlier_price_grid.shape[0]
        self.max_t = np.max(config.time_grid)
        self.strike = config.strike
        self.rd = config.interest_rate
        self.rf = config.foreign_interest_rate
        self.rho = config.rho
        self.k = 1.0
        self.th = 0.4
        self.volofvol = 0.5

        c = self.strike / 10
        non_uniform_s = np.zeros(self.m)
        for i in range(self.m):
            dk_i = (1 / self.m) * (math.asinh((3 * self.strike - self.strike) / c) - math.asinh(-self.strike / c))
            k_i = math.asinh(-self.strike / c) + i * dk_i
            non_uniform_s[i] = abs(round(self.strike + c * np.sinh(k_i), 5))

        self.volofvol = 0.5
        V = 5 * self.volofvol
        d = V / 500

        non_uniform_v = np.zeros(self.n)
        for j in range(self.n):
            dk_i = (1 / self.n) * math.asinh(V / d)
            non_uniform_v[j] = abs(d * math.sinh(j * dk_i))

        self.vv, self.tt, self.xx = np.meshgrid(
            non_uniform_v,
            np.linspace(0, self.max_t, 201),
            non_uniform_s
        )

        self.dxx = np.zeros_like(self.xx[0, 0, :])
        self.dxx[1:] = self.xx[0, 0, 1:] - self.xx[0, 0, :-1]
        self.dxx = np.hstack([self.dxx, self.dxx[-1]])

        self.dvv = np.zeros_like(self.vv[0, :, 0])
        self.dvv[1:] = self.vv[0, 1:, 0] - self.vv[0, :-1, 0]

        self.grid = np.zeros_like(self.xx)

        #payoff boundary condition
        self.grid[-1, :, :] = np.maximum(self.xx[-1, :, :] - self.strike, 0)

        # s=0 boundary condidition
        self.grid[:, :, 0] = 0.0

        # v = V boundary condition
        self.grid[:, -1, :] = np.maximum(self.xx[:, -1, :] * np.exp(-self.rf * self.tt[:, -1, :]) - self.strike * (np.exp(-self.rd * self.tt[:, -1, :])), 0)

    def solve(self):
        solve_func = np.vectorize(self._solve)
        s = time.perf_counter()
        solve_func()
        print(f"Vectorized solve took: {time.perf_counter() - s:.4f} sec")


    def _solve(self):
        A0 = np.zeros((self.n * self.m, self.n * self.m), dtype=np.float32)
        A1 = np.zeros((self.n * self.m, self.n * self.m), dtype=np.float32)
        A2 = np.zeros((self.n * self.m, self.n * self.m), dtype=np.float32)

        for i in range(0, self.n - 1):
            for j in range(1, self.m):
                A0_row = np.zeros((self.n, self.m), dtype=np.float32)
                A1_row = np.zeros((self.n, self.m), dtype=np.float32)
                A2_row = np.zeros((self.n, self.m), dtype=np.float32)
                
                if (i == 0) and (j != (self.m - 1)):
                    A1_row[i, j - 1] += (self.rd - self.rf) * self.xx[0, 0, j] * self.beta_coef(self.dxx, j, -1)
                    A1_row[i, j] += (self.rd - self.rf) * self.xx[0, 0, j] * self.beta_coef(self.dxx, j, 0)
                    A1_row[i, j] += (-self.rd / 2)
                    A1_row[i, j + 1] += (self.rd - self.rf) * self.xx[0, 0, j] * self.beta_coef(self.dxx, j, 1)

                    A2_row[i, j] += self.k * (self.th - self.vv[0, i, 0]) * self.gamma_coef(self.dvv, i, 0)
                    A2_row[i, j] += (-self.rd / 2)
                    A2_row[i + 1, j] += self.k * (self.th - self.vv[0, i, 0]) * self.gamma_coef(self.dvv, i, 1)
                    A2_row[i + 2, j] += self.k * (self.th - self.vv[0, i, 0]) * self.gamma_coef(self.dvv, i, 2)

                elif (i == 0) and (j == (self.m - 1)):
                    A1_row[i, j] += (-self.rd / 2)

                    A2_row[i, j] += self.k * (self.th - self.vv[0, i, 0]) * self.gamma_coef(self.dvv, i, 0)
                    A2_row[i, j] += (-self.rd / 2)
                    A2_row[i + 1, j] += self.k * (self.th - self.vv[0, i, 0]) * self.gamma_coef(self.dvv, i, 1)
                    A2_row[i + 2, j] += self.k * (self.th - self.vv[0, i, 0]) * self.gamma_coef(self.dvv, i, 2)

                elif i != 0:
                    if j != (self.m - 1):
                        cross_coef = self.rho * self.volofvol * self.xx[0, 0, j] * self.vv[0, i, 0]
                        for k in range(-1, 2):
                            for l in range(-1, 2):
                                A0_row[i + k, j + l] += cross_coef * self.beta_coef(self.dxx, j, k) * self.beta_coef(self.dvv, i, l)

                        A1_row[i, j - 1] += 0.5 * self.xx[0, 0, j] ** 2 * self.vv[0, i, 0] * self.delta_coef(self.dxx, j, -1)
                        A1_row[i, j - 1] += (self.rd - self.rf) * self.xx[0, 0, j] * self.beta_coef(self.dxx, j, -1)
                        A1_row[i, j] += 0.5 * self.xx[0, 0, j] ** 2 * self.vv[0, i, 0] * self.delta_coef(self.dxx, j, 0)
                        A1_row[i, j] += (self.rd - self.rf) * self.xx[0, 0, j] * self.beta_coef(self.dxx, j, 0)
                        A1_row[i, j] += (-self.rd / 2)
                        A1_row[i, j + 1] += 0.5 * self.xx[0, 0, j] ** 2 * self.vv[0, i, 0] * self.delta_coef(self.dxx, j, 1)
                        A1_row[i, j + 1] += (self.rd - self.rf) * self.xx[0, 0, j] * self.beta_coef(self.dxx, j, 1)
                    else:
                        A1_row[i, j] += (-self.rd / 2)

                    if self.vv[0, i, 0] <= 1:
                        A2_row[i - 1, j] += 0.5 * self.volofvol ** 2 * self.vv[0, i, 0] * self.delta_coef(self.dvv, i, -1)
                        A2_row[i - 1, j] += self.k * (self.th - self.vv[0, i, 0]) * self.beta_coef(self.dvv, i, -1)
                        A2_row[i, j] += 0.5 * self.volofvol ** 2 * self.vv[0, i, 0] * self.delta_coef(self.dvv, i, 0)
                        A2_row[i, j] += self.k * (self.th - self.vv[0, i, 0]) * self.beta_coef(self.dvv, i, 0)
                        A2_row[i, j] += (-self.rd / 2)
                        A2_row[i + 1, j] += 0.5 * self.volofvol ** 2 * self.vv[0, i, 0] * self.delta_coef(self.dvv, i, 1)
                        A2_row[i + 1, j] += self.k * (self.th - self.vv[0, i, 0]) * self.beta_coef(self.dvv, i, 1)
                    else:
                        A2_row[i - 2, j] += self.k * (self.th - self.vv[0, i, 0]) * self.alpha_coef(self.dvv, i, -2)
                        A2_row[i - 1, j] += self.k * (self.th - self.vv[0, i, 0]) * self.alpha_coef(self.dvv, i, -1)
                        A2_row[i, j] += self.k * (self.th - self.vv[0, i, 0]) * self.alpha_coef(self.dvv, i, 0)
                        A2_row[i - 1, j] += 0.5 * self.volofvol ** 2 * self.vv[0, i, 0] * self.delta_coef(self.dvv, i, -1)
                        A2_row[i, j] += 0.5 * self.volofvol ** 2 * self.vv[0, i, 0] * self.delta_coef(self.dvv, i, 0)
                        A2_row[i + 1, j] += 0.5 * self.volofvol ** 2 * self.vv[0, i, 0] * self.delta_coef(self.dvv, i, 1)
                        A2_row[i, j] += (-self.rd / 2)
                else:
                    raise NotImplementedError(f"Not implemented for the grid, indices: {i, j}")

                A0[i * self.m + j, :] = A0_row.reshape(-1)
                A1[i * self.m + j, :] = A1_row.reshape(-1)
                A2[i * self.m + j, :] = A2_row.reshape(-1) # order with "F" and modify just for A2 as its pentadiagonal

        A0 = csc_matrix(A0)
        A1 = csc_matrix(A1)
        A2 = csc_matrix(A2)

        i, j = np.meshgrid(np.arange(self.n), np.arange(self.m), indexing='ij')
        nbi_mask = (i != self.n - 1) & (j != 0)
        non_boundary_indices = np.where(nbi_mask)
        nbi = i[non_boundary_indices] * self.m + j[non_boundary_indices]
        dt = self.tt[-1, 0, 0] - self.tt[-2, 0, 0]

        A = A0 + A1 + A2

        A1_LU_sp = splu(identity(nbi.shape[0], format="csc") - self.adi_param * dt * A1[:, nbi][nbi, :])
        A2_LU_sp = splu(identity(nbi.shape[0], format="csc") - self.adi_param * dt * A2[:, nbi][nbi, :])
        # A1_inv = inv(identity(nbi.shape[0], format="csc") - self.adi_param * dt * A1[:, nbi][nbi, :])
        # A2_inv = inv(identity(nbi.shape[0], format="csc") - self.adi_param * dt * A2[:, nbi][nbi, :])

        b0_prev, b1_prev, b2_prev = None, None, None
        b0_curr, b1_curr, b2_curr = None, None, None

        for t in tqdm(range(-2, -self.tt.shape[0] - 1, -1), disable=not self.verbose):
            if b0_prev is None:
                b0_prev = np.zeros(self.n * self.m, dtype=np.float32)
                b0_prev += A0[:, ::self.m] @ self.grid[t + 1, :, 0]
                b0_prev += A0[:, -self.m:] @ self.grid[t + 1, -1, :]
                b0_prev -= (A0[:, 0] * self.grid[t + 1, -1, 0]).toarray().reshape(-1)
            else:
                b0_prev = b0_curr
            if b1_prev is None:
                b1_prev = np.zeros(self.n * self.m, dtype=np.float32)
                b1_prev += A1[:, ::self.m] @ self.grid[t + 1, :, 0]
                b1_prev += A1[:, -self.m:] @ self.grid[t + 1, -1, :]
                b1_prev[self.m - 1::self.m] += (self.rd - self.rf) * self.xx[t + 1, :, -1] * np.exp(-self.rf * self.tt[t + 1, :, -1])
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
            b1_curr[self.m-1::self.m] += (self.rd - self.rf) * self.xx[t, :, -1] * np.exp(-self.rf * self.tt[t, :, -1])
            b1_curr -= (A1[:, 0] * self.grid[t, -1, 0]).toarray().reshape(-1)

            b2_curr += A2[:, ::self.m] @ self.grid[t, :, 0]
            b2_curr += A2[:, -self.m:] @ self.grid[t, -1, :]
            b2_curr -= (A2[:, 0] * self.grid[t, -1, 0]).toarray().reshape(-1)

            b_prev = b0_prev + b1_prev + b2_prev

            y0 = self.grid[t + 1, :-1, 1:].reshape(-1, 1) + dt * (A[:, nbi][nbi, :] @ self.grid[t + 1, :-1, 1:].reshape(-1, 1) + b_prev[nbi].reshape(-1, 1))
            y0 = y0.reshape(-1, 1)

            y1 = A1_LU_sp.solve(
                y0 - self.adi_param * dt * (A1[:, nbi][nbi, :] @ self.grid[t + 1, :-1, 1:].reshape(-1, 1) + b1_prev[nbi].reshape(-1, 1) - b1_curr[nbi].reshape(-1, 1))
            )

            # y1 = A1_inv @ (y0 - self.adi_param * dt * (A1[:, nbi][nbi, :] @ self.grid[t + 1, :-1, 1:].reshape(-1, 1) + b1_prev[nbi].reshape(-1, 1) - b1_curr[nbi].reshape(-1, 1)))

            # y1 = self.TDMA(
            #     # a=np.diag(- self.adi_param * dt * A1[:, nbi][nbi, :], k=-1),
            #     a=(- self.adi_param * dt * A1[:, nbi][nbi, :]).diagonal(k=-1),
            #     # b=np.diag(identity(nbi.shape[0]) - self.adi_param * dt * A1[:, nbi][nbi, :]),
            #     b=(identity(nbi.shape[0]) - self.adi_param * dt * A1[:, nbi][nbi, :]).diagonal(),
            #     # c=np.diag(- self.adi_param * dt * A1[:, nbi][nbi, :], k=1),
            #     c=(- self.adi_param * dt * A1[:, nbi][nbi, :]).diagonal(k=1),
            #     d=(y0 - self.adi_param * dt * (A1[:, nbi][nbi, :] @ self.grid[t + 1, :-1, 1:].reshape(-1, 1) + b1_prev[nbi].reshape(-1, 1) - b1_curr[nbi].reshape(-1, 1))).reshape(-1)
            # ).reshape(-1, 1)

            y2 = A2_LU_sp.solve(
                y1 - self.adi_param * dt * (A2[:, nbi][nbi, :] @ self.grid[t + 1, :-1, 1:].reshape(-1, 1) + b2_prev[nbi].reshape(-1, 1) - b2_curr[nbi].reshape(-1, 1))
            )

            # y2 = A2_inv @ (y1 - self.adi_param * dt * (A2[:, nbi][nbi, :] @ self.grid[t + 1, :-1, 1:].reshape(-1, 1) + b2_prev[nbi].reshape(-1, 1) - b2_curr[nbi].reshape(-1, 1)))

            # nbi_v = [i * self.m + j for j in range(self.m) for i in range(self.n) if ((i != self.n - 1) & (j != 0))]
            # A2_ab = self.to_banded(
            #     np.eye(A2[nbi_v, :][:, nbi_v].shape[0]) - self.adi_param * dt * A2[nbi_v, :][:, nbi_v],
            #     2, 2
            # )
            #
            # y2 = solve_banded(
            #     (2, 2),
            #     A2_ab,
            #     y1 - self.adi_param * dt * (A2[nbi_v, :][:, nbi_v] @ self.grid[t + 1, :-1, 1:].reshape(-1, 1, order="F") + b2_prev[nbi_v].reshape(-1, 1) - b2_curr[nbi_v].reshape(-1, 1))
            # )

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

    @staticmethod
    def beta_coef(arr, i, offset):
        if offset == -1:
            return (- arr[i + 1]) / (arr[i] * (arr[i] + arr[i + 1]))
        elif offset == 0:
            return (arr[i + 1] - arr[i]) / (arr[i] * arr[i + 1])
        elif offset == 1:
            return (arr[i]) / (arr[i + 1] * (arr[i] + arr[i + 1]))

    @staticmethod
    def gamma_coef(arr, i, offset):
        if offset == 0:
            return (-2 * arr[i + 1] - arr[i + 2]) / (arr[i + 1] * (arr[i + 1] + arr[i + 2]))
        elif offset == 1:
            return (arr[i + 1] + arr[i + 2]) / (arr[i + 1] * arr[i + 2])
        elif offset == 2:
            return (-arr[i + 1]) / (arr[i + 2] * (arr[i + 1] + arr[i + 2]))

    @staticmethod
    def delta_coef(arr, i , offset):
        if offset == -1:
            return 2 / (arr[i] * (arr[i] + arr[i + 1]))
        elif offset == 0:
            return (-2) / (arr[i] * arr[i + 1])
        elif offset == 1:
            return 2 / (arr[i + 1] * (arr[i] + arr[i + 1]))

    @staticmethod
    def TDMA(a, b, c, d):  # https://stackoverflow.com/a/43214907
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

    @staticmethod
    def to_banded(A, lower_bandwidth, upper_bandwidth):
        n = A.shape[0]
        ab = np.zeros((lower_bandwidth + upper_bandwidth + 1, n))

        for i in range(n):
            l, u = max(0, i - lower_bandwidth), min(n, i + upper_bandwidth + 1)
            ab[lower_bandwidth + l - i:lower_bandwidth + u - i, i] = A[i, l:u]
        return ab

if __name__ == "__main__":
    # non_uniform_s
    K = 100
    c = K / 10
    n = 101
    non_uniform_s = np.zeros(n)
    for i in range(n):
        dk_i = (1/n) * (math.asinh((3 * K - K) / c) - math.asinh(-K / c))
        k_i = math.asinh(-K / c) + i * dk_i
        non_uniform_s[i] = abs(round(K + c * np.sinh(k_i), 5))

    volofvol = 0.3
    V = 5 * volofvol
    d = V / 500

    m = 51
    non_uniform_v = np.zeros(m)
    for j in range(m):
        dk_i = (1/m) * math.asinh(V / d)
        non_uniform_v[j] = abs(d * math.sinh(j * dk_i))

    config = PDESolverConfig(
        underlier_price_grid=np.array(non_uniform_s, dtype=np.float32),
        time_grid=np.linspace(0, 1, 200, dtype=np.float32),
        variance_grid=np.array(non_uniform_v, dtype=np.float32),
        strike=K,
        interest_rate=0.03,
        rho=-0.5,
        kappa=2.0,
        theta=0.04,
        volofvol=volofvol,
    )
    solver = DouglasADI(config)
    solver.solve()
    print(solver)
