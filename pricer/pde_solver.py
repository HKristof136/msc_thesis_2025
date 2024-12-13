import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
from pricer.analytical import BlackScholesCall, BlackScholesPut

class CrankNicolsonPDESolver:
    underlier_lower_boundary = lambda self: np.zeros(self.t.shape[1])
    underlier_upper_boundary = lambda self: np.zeros(self.t.shape[1])
    payoff = lambda self: np.zeros(self.x.shape[0])
    backward_step = lambda self, x, i: np.zeros(x.shape[0])

    def __init__(self, config: dict):
        self.solved = False
        self.config = config
        self.verbose = config.get("verbose", True)
        self.term = np.max(config["t"])
        self.strike = config["strike"]
        self.r = config["r"]
        self.sigma = config["sigma"]

        self.t, self.x = np.meshgrid(self.config["t"], self.config["x"])

        self.grid = np.zeros(self.x.shape)
        self.grid[0, :] = self.underlier_lower_boundary()
        self.grid[-1, :] = self.underlier_upper_boundary()
        self.grid[:, -1] = self.payoff()

    def solve(self):
        if self.solved:
            print("Already solved")
            return None
        dt = self.config["t"][1] - self.config["t"][0]

        calc_A = True
        A = np.zeros((self.config["x"].shape[0], self.config["x"].shape[0]))
        B = np.zeros((self.config["x"].shape[0], self.config["x"].shape[0]))

        iterator = tqdm(range(-2, -self.t.shape[1] - 1, -1)) if self.verbose else range(-2, -self.t.shape[1] - 1, -1)

        for i in iterator:
            if calc_A:
                for j in range(1, self.x.shape[0] - 1):
                    a_j = 0.25 * dt * (self.sigma**2 * j**2 - self.r * j)
                    b_j = -0.5 * dt * (self.sigma**2 * j**2 + self.r)
                    c_j = 0.25 * dt * (self.sigma**2 * j**2 + self.r * j)

                    A[j, j - 1 : j + 2] = [-a_j, 1 - b_j, -c_j]
                    B[j, j - 1 : j + 2] = [a_j, 1 + b_j, c_j]

                a_1 = B[1, 0]
                c_m_1 = B[-2, -1]
                A = A[1:-1, 1:-1]
                B = B[1:-1, 1:-1]
                lu_A, piv_A = lu_factor(A)
                calc_A = False

            b = B @ self.grid[1:-1, i + 1].copy().reshape(-1, 1)

            b[0] += a_1 * (self.grid[0, i] + self.grid[0, i + 1])
            b[-1] += c_m_1 * (self.grid[-1, i] + self.grid[-1, i + 1])
            self.grid[1:-1, i] = lu_solve((lu_A, piv_A), b).flatten()

            self.grid[:, i] = self.backward_step(self.grid[:, i], i)
            
        self.solved = True

    def price(self, points):
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        if not self.solved:
            self.solve()
        interp_func = RegularGridInterpolator(
            (self.config["x"], self.config["t"]), self.grid, method="linear"
        )
        interp_points = np.column_stack([points[:, 0], self.term - points[:, 1]])
        return interp_func(interp_points)

class BlackScholesCallPDE(CrankNicolsonPDESolver):
    underlier_lower_boundary = lambda self: 0
    underlier_upper_boundary = lambda self: np.maximum(self.x[-1, :] - np.exp(-self.r * (np.max(self.term) - self.t[-1, :])) * self.strike, 0)
    payoff = lambda self: np.maximum(self.x[:, -1] - self.strike, 0)
    backward_step = lambda self, x, i: x

class BlackScholesPutPDE(CrankNicolsonPDESolver):
    underlier_lower_boundary = lambda self: np.exp(-self.r * (np.max(self.term) - self.t[-1, :])) * self.strike
    underlier_upper_boundary = lambda self: 0
    payoff = lambda self: np.maximum(self.strike - self.x[:, -1], 0)
    backward_step = lambda self, x, i: x
    
    def delta(self, points):
        shift = 0.01

        if not isinstance(points, np.ndarray):
            points = np.array(points)
        
        down_shift_points = points.copy()
        down_shift_points[:, 0] -= shift
        up_shift_points = points.copy()
        up_shift_points[:, 0] += shift

        price_down = self.price(down_shift_points)
        price_up = self.price(up_shift_points)

        return (price_up - price_down) / (2 * shift)

class AmericanBlackScholesPutPDE(BlackScholesPutPDE):
    underlier_lower_boundary = lambda self: self.strike
    backward_step = lambda self, x, i: np.maximum(self.strike - self.x[:, i], x)

class BarrierUpPDE(CrankNicolsonPDESolver):
    def __init__(self, config):
        self.barrier = config["barrier"]
        super().__init__(config)
    
    def solve(self):
        if self.solved:
            print("Already solved")
            return None
        dt = self.config["t"][1] - self.config["t"][0]

        calc_A = True

        reduced_j = np.max(np.nonzero(self.config["x"] < self.barrier)) + 2
        reduced_scope = self.config["x"][:reduced_j]
        A = np.zeros((reduced_scope.shape[0], reduced_scope.shape[0]))
        B = np.zeros((reduced_scope.shape[0], reduced_scope.shape[0]))

        iterator = tqdm(range(-2, -self.t.shape[1] - 1, -1)) if self.verbose else range(-2, -self.t.shape[1] - 1, -1)

        for i in iterator:
            if calc_A:
                for j in range(1, reduced_scope.shape[0] - 1):
                    a_j = 0.25 * dt * (self.sigma**2 * j**2 - self.r * j)
                    b_j = -0.5 * dt * (self.sigma**2 * j**2 + self.r)
                    c_j = 0.25 * dt * (self.sigma**2 * j**2 + self.r * j)

                    A[j, j - 1 : j + 2] = [-a_j, 1 - b_j, -c_j]
                    B[j, j - 1 : j + 2] = [a_j, 1 + b_j, c_j]

                a_1 = B[1, 0]
                c_m_1 = B[-2, -1]
                A = A[1:-1, 1:-1]
                B = B[1:-1, 1:-1]
                lu_A, piv_A = lu_factor(A)
                calc_A = False

            b = B @ self.grid[1:reduced_j-1, i + 1].copy().reshape(-1, 1)

            b[0] += a_1 * (self.grid[0, i] + self.grid[0, i + 1])
            b[-1] += c_m_1 * (self.grid[reduced_j-1, i] + self.grid[reduced_j-1, i + 1])
            self.grid[1:reduced_j-1, i] = lu_solve((lu_A, piv_A), b).flatten()

            self.grid[:, i] = self.backward_step(self.grid[:, i], i)
            
        self.solved = True

class BarrierUpAndOutCallPDE(BarrierUpPDE):
    underlier_lower_boundary = lambda self: 0
    underlier_upper_boundary = lambda self: 0
    payoff = lambda self: np.maximum(self.x[:, -1] - self.strike, 0) * (self.x[:, -1] < self.barrier)
    backward_step = lambda self, x, i: x * (self.x[:, i] < self.barrier)

    def __init__(self, config):
        self.barrier = config["barrier"]
        super().__init__(config)

class AmericanBarrierUpAndOutCallPDE(BarrierUpAndOutCallPDE):
    backward_step = lambda self, x, i: np.maximum(self.x[:, i] - self.strike, x * (self.x[:, i] < self.barrier)) * (self.x[:, i] < self.barrier)

class BarrierUpAndOutPutPDE(BarrierUpPDE):
    underlier_lower_boundary = lambda self: np.exp(-self.r * (np.max(self.term) - self.t[-1, :])) * self.strike
    underlier_upper_boundary = lambda self: 0
    payoff = lambda self: np.maximum(self.strike - self.x[:, -1], 0) * (self.x[:, -1] < self.barrier)
    backward_step = lambda self, x, i: x * (self.x[:, i] < self.barrier)

    def __init__(self, config):
        self.barrier = config["barrier"]
        super().__init__(config)

class AmericanBarrierUpAndOutPutPDE(BarrierUpAndOutPutPDE):
    underlier_lower_boundary = lambda self: self.strike
    backward_step = lambda self, x, i: np.maximum(self.strike - self.x[:, i], x * (self.x[:, i] < self.barrier)) * (self.x[:, i] < self.barrier)

class BarrierUpAndInCallPDE(BarrierUpPDE):
    underlier_lower_boundary = lambda self: 0
    underlier_upper_boundary = lambda self: np.maximum(self.x[-1, :] - np.exp(-self.r * (np.max(self.term) - self.t[-1, :])) * self.strike, 0)
    payoff = lambda self: np.maximum(self.x[:, -1] - self.strike, 0) * (self.x[:, -1] > self.barrier)
    backward_step = lambda self, x, i: x * (self.x[:, i] <= self.barrier) + self.call_grid[:, i] * (self.x[:, i] > self.barrier)

    def __init__(self, config):
        self.barrier = config["barrier"]
        super().__init__(config)

        self.call_grid = BlackScholesCall(
            underlier_price=self.x,
            strike=np.ones_like(self.x) * self.strike,
            expiry=np.max(self.term) - self.t,
            interest_rate=np.ones_like(self.x) * self.r,
            volatility=np.ones_like(self.x) * self.sigma
        ).price()
        knock_in_ind = np.max(np.nonzero(self.config["x"] < self.barrier)) + 2
        self.grid[knock_in_ind-1, :] = self.call_grid[knock_in_ind-1, :]

    # def solve(self):
    #     if self.solved:
    #         print("Already solved")
    #         return None
    #     dt = self.config["t"][1] - self.config["t"][0]

    #     calc_A = True

    #     reduced_j = np.max(np.nonzero(self.config["x"] < self.barrier)) + 2
    #     self.grid[reduced_j-1, :] = self.call_grid[reduced_j-1, :]

    #     reduced_scope = self.config["x"][:reduced_j]
    #     A = np.zeros((reduced_scope.shape[0], reduced_scope.shape[0]))
    #     B = np.zeros((reduced_scope.shape[0], reduced_scope.shape[0]))

    #     iterator = tqdm(range(-2, -self.t.shape[1] - 1, -1)) if self.verbose else range(-2, -self.t.shape[1] - 1, -1)

    #     for i in iterator:
    #         if calc_A:
    #             for j in range(1, reduced_scope.shape[0] - 1):
    #                 a_j = 0.25 * dt * (self.sigma**2 * j**2 - self.r * j)
    #                 b_j = -0.5 * dt * (self.sigma**2 * j**2 + self.r)
    #                 c_j = 0.25 * dt * (self.sigma**2 * j**2 + self.r * j)

    #                 A[j, j - 1 : j + 2] = [-a_j, 1 - b_j, -c_j]
    #                 B[j, j - 1 : j + 2] = [a_j, 1 + b_j, c_j]

    #             a_1 = B[1, 0]
    #             c_m_1 = B[-2, -1]
    #             A = A[1:-1, 1:-1]
    #             B = B[1:-1, 1:-1]
    #             lu_A, piv_A = lu_factor(A)
    #             calc_A = False

    #         b = B @ self.grid[1:reduced_j-1, i + 1].copy().reshape(-1, 1)

    #         b[0] += a_1 * (self.grid[0, i] + self.grid[0, i + 1])
    #         b[-1] += c_m_1 * (self.grid[reduced_j - 1, i] + self.grid[reduced_j - 1, i + 1])
    #         self.grid[1:reduced_j-1, i] = lu_solve((lu_A, piv_A), b).flatten()

    #         self.grid[:, i] = self.backward_step(self.grid[:, i], i)
            
    #     self.solved = True

class AmericanBarrierUpAndInCallPDE(BarrierUpAndInCallPDE):
    underlier_upper_boundary = lambda self: np.maximum(self.x[-1, :] - self.strike, 0)
    backward_step = lambda self, x, i: x * (self.x[:, i] <= self.barrier) + np.maximum(self.x[:, i] - self.strike,
                                                                            self.call_grid[:, i] * (self.x[:, i] > self.barrier)
                                                                            ) * (self.x[:, i] > self.barrier)

class BarrierUpAndInPutPDE(BarrierUpPDE):
    underlier_lower_boundary = lambda self: 0
    underlier_upper_boundary = lambda self: 0
    payoff = lambda self: np.maximum(self.strike - self.x[:, -1], 0) * (self.x[:, -1] > self.barrier)
    backward_step = lambda self, x, i: x * (self.x[:, i] <= self.barrier) + self.put_grid[:, i] * (self.x[:, i] > self.barrier)

    def __init__(self, config):
        self.barrier = config["barrier"]
        super().__init__(config)

        self.put_grid = BlackScholesPut(
            underlier_price=self.x,
            strike=np.ones_like(self.x) * self.strike,
            expiry=np.max(self.term) - self.t,
            interest_rate=np.ones_like(self.x) * self.r,
            volatility=np.ones_like(self.x) * self.sigma
        ).price()
        knock_in_ind = np.max(np.nonzero(self.config["x"] < self.barrier)) + 2
        self.grid[knock_in_ind-1, :] = self.put_grid[knock_in_ind-1, :]

class AmericanBarrierUpAndInPutPDE(BarrierUpAndInPutPDE):
    def __init__(self, config):
        super().__init__(config)

        put_grid_cls = AmericanBlackScholesPutPDE(config)
        put_grid_cls.solve()
        self.put_grid = put_grid_cls.grid

class BarrierDownPDE(CrankNicolsonPDESolver):
    def __init__(self, config):
        self.barrier = config["barrier"]
        super().__init__(config)
    
    def solve(self):
        pass

class BarrierDownAndOutCallPDE(CrankNicolsonPDESolver):
    underlier_lower_boundary = lambda self: 0
    underlier_upper_boundary = lambda self: np.maximum(self.x[-1, :] - np.exp(-self.r * (np.max(self.term) - self.t[-1, :])) * self.strike, 0)
    payoff = lambda self: np.maximum(self.x[:, -1] - self.strike, 0) * (self.x[:, -1] > self.barrier)
    backward_step = lambda self, x, i: x * (self.x[:, i] > self.barrier)

    def __init__(self, config):
        self.barrier = config["barrier"]
        super().__init__(config)

class AmericanBarrierDownAndOutCallPDE(BarrierDownAndOutCallPDE):
    underlier_upper_boundary = lambda self: np.maximum(self.x[-1, :] - self.strike, 0)
    backward_step = lambda self, x, i: np.maximum(self.x[:, i] - self.strike, self.x[:, i] * (self.x[:, i] > self.barrier)) * (self.x[:, i] > self.barrier)

class BarrierDownAndOutPutPDE(CrankNicolsonPDESolver):
    underlier_lower_boundary = lambda self: 0
    underlier_upper_boundary = lambda self: 0
    payoff = lambda self: np.maximum(self.strike - self.x[:, -1], 0) * (self.x[:, -1] > self.barrier)
    backward_step = lambda self, x, i: x * (self.x[:, i] > self.barrier)

    def __init__(self, config):
        self.barrier = config["barrier"]
        super().__init__(config)

class AmericanBarrierDownAndOutPutPDE(BarrierDownAndOutPutPDE):
    backward_step = lambda self, x, i: np.maximum(self.strike - self.x[:, i], x * (self.x[:, i] > self.barrier)) * (self.x[:, i] > self.barrier)

class BarrierDownAndInCallPDE(CrankNicolsonPDESolver):
    underlier_lower_boundary = lambda self: 0
    underlier_upper_boundary = lambda self: 0
    payoff = lambda self: np.maximum(self.x[:, -1] - self.strike, 0) * (self.x[:, -1] < self.barrier)
    backward_step = lambda self, x, i: x * (self.x[:, i] >= self.barrier) + self.call_grid[:, i] * (self.x[:, i] < self.barrier)

    def __init__(self, config):
        self.barrier = config["barrier"]
        super().__init__(config)

        self.call_grid = BlackScholesCall(
            underlier_price=self.x,
            strike=np.ones_like(self.x) * self.strike,
            expiry=np.max(self.term) - self.t,
            interest_rate=np.ones_like(self.x) * self.r,
            volatility=np.ones_like(self.x) * self.sigma
        ).price()

class AmericanBarrierDownAndInCallPDE(BarrierDownAndInCallPDE):
    backward_step = lambda self, x, i: x * (self.x[:, i] >= self.barrier) + np.maximum(self.x[:, i] - self.strike,
                                                                            self.call_grid[:, i] * (self.x[:, i] < self.barrier)
                                                                            ) * (self.x[:, i] < self.barrier)

class BarrierDownAndInPutPDE(CrankNicolsonPDESolver):
    underlier_lower_boundary = lambda self: np.exp(-self.r * (np.max(self.term) - self.t[-1, :])) * self.strike
    underlier_upper_boundary = lambda self: 0
    payoff = lambda self: np.maximum(self.strike - self.x[:, -1], 0) * (self.x[:, -1] < self.barrier)
    backward_step = lambda self, x, i: x * (self.x[:, i] >= self.barrier) + self.put_grid[:, i] * (self.x[:, i] < self.barrier)

    def __init__(self, config):
        self.barrier = config["barrier"]
        super().__init__(config)

        self.put_grid = BlackScholesPut(
            underlier_price=self.x,
            strike=np.ones_like(self.x) * self.strike,
            expiry=np.max(self.term) - self.t,
            interest_rate=np.ones_like(self.x) * self.r,
            volatility=np.ones_like(self.x) * self.sigma
        ).price()

class AmericanBarrierDownAndInPutPDE(BarrierDownAndInPutPDE):
    def __init__(self, config):
        super().__init__(config)

        put_grid_cls = AmericanBlackScholesPutPDE(config)
        put_grid_cls.solve()
        self.put_grid = put_grid_cls.grid

if __name__ == "__main__":
    # config = {
    #     "x": np.linspace(0, 200, 10),
    #     "t": np.linspace(0, 1, 1001),
    #     "strike": 100,
    #     "r": 0.05,
    #     "sigma": 0.2,
    #     "barrier": 150,
    # }

    config = {
    "x": np.linspace(0, 200, 50),
    "t": np.linspace(0, 1, 100),
    "strike": 100,
    "r": 0.05,
    "sigma": 0.2,
    "barrier": 180
}

    cls_to_test = BarrierUpAndOutCallPDE
    solver = cls_to_test(config)
    solver.solve()
    print(solver.price([[36, 1]]))
