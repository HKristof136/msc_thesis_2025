import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
from pricer.config_base import PDESolverConfig, DefaultConfig


class CrankNicolsonPDESolver:
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
        return self._calc_greek(points, greek="delta", order="first", precision=0.01)

    def gamma(self, points):
        return self._calc_greek(points, greek="delta", order="second", precision=0.01)

    def theta(self, points):
        return self._calc_greek(points, greek="theta", order="first", precision=0.001)

    def vega(self, points):
        return self._calc_greek(points, greek="vega", order="first", precision=0.001)

    def rho(self, points):
        return self._calc_greek(points, greek="rho", order="first", precision=0.001)


class BarrierUpPDE(CrankNicolsonPDESolver):
    def __init__(self, config: PDESolverConfig = DefaultConfig.pde_solver_barrier_call):
        self.barrier = config.barrier
        super().__init__(config)
        self.f = np.zeros_like(self.config.time_grid, dtype=np.float32)

    def solve(self):
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
