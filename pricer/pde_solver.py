import numpy as np
from scipy.linalg import lu_factor, lu_solve
from analytical import (
    BarrierDownAndOutCall,
    BarrierUpAndOutCall,
    BarrierDownAndInCall,
    BarrierUpAndInCall,
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

class CrankNicolsonPDESolver:
    def __init__(self, config: dict):
        self.solved = False
        self.config = config
        self.term = np.max(config["t"])
        self.strike = config["strike"]
        self.r = config["r"]
        self.sigma = config["sigma"]

        self.call_fl = -1 if config.get("call_fl") == "put" else 1
        self.barrier = config.get("barrier", None)
        self.barrier_type = config.get("barrier_type", "")

        if "in" in self.barrier_type:
            call_cls = CrankNicolsonPDESolver(
                {
                    "x": config["x"],
                    "t": config["t"],
                    "strike": self.strike,
                    "r": self.r,
                    "sigma": self.sigma,
                }
            )
            call_cls.solve()
            self.call_grid = call_cls.grid

        self.american = config.get("american", False)

        self.t, self.x = np.meshgrid(self.config["t"], self.config["x"])

        self.grid = np.zeros(self.x.shape)
        # TODO: Implement general initial conditions that can be passed in the config
        self.grid[0, :] = 0
        self.grid[-1, :] = np.maximum(
            self.x[-1, :]
            - np.exp(-self.r * (np.max(self.term) - self.t[-1, :])) * self.strike,
            0,
        )
        self.grid[:, -1] = np.maximum(self.x[:, -1] - self.strike, 0)

        if self.barrier:
            if self.barrier_type in ["down-and-out", "up-and-in"]:
                self.grid[0, :] = 0
                self.grid[-1, :] = np.maximum(
                    self.x[-1, :]
                    - np.exp(-self.r * (np.max(self.term) - self.t[-1, :]))
                    * self.strike,
                    0,
                )
                self.grid[self.x[:, -1] <= self.barrier, -1] = 0
            elif self.barrier_type in ["up-and-out", "down-and-in"]:
                self.grid[0, :] = 0
                self.grid[-1, :] = 0
                self.grid[self.x[:, -1] >= self.barrier, -1] = 0

    def solve(self):
        if self.solved:
            print("Already solved")
            return None
        dt = self.config["t"][1] - self.config["t"][0]

        calc_A = True
        A = np.zeros((self.config["x"].shape[0], self.config["x"].shape[0]))
        B = np.zeros((self.config["x"].shape[0], self.config["x"].shape[0]))

        for i in tqdm(range(-2, -self.t.shape[1] - 1, -1)):
            if calc_A:
                for j in range(1, self.x.shape[0] - 1):
                    a_j = 0.25 * dt * (self.sigma**2 * j**2 - self.r * j)
                    b_j = -0.5 * dt * (self.sigma**2 * j**2 + self.r)
                    c_j = 0.25 * dt * (self.sigma**2 * j**2 + self.r * j)

                    A[j, j - 1:j + 2] = [-a_j, 1 - b_j, -c_j]
                    B[j, j - 1:j + 2] = [a_j, 1 + b_j, c_j]

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

            if self.american:
                self.grid[:, i] = np.maximum(
                    self.grid[:, i], self.x[:, i] - self.strike
                )
            if self.barrier:
                if self.barrier_type == "down-and-out":
                    self.grid[self.x[:, i] <= self.barrier, i] = 0
                elif self.barrier_type == "up-and-out":
                    self.grid[self.x[:, i] >= self.barrier, i] = 0
                elif self.barrier_type == "down-and-in":
                    self.grid[self.x[:, i] <= self.barrier, i] = self.call_grid[
                        self.x[:, i] <= self.barrier, i
                    ]
                elif self.barrier_type == "up-and-in":
                    self.grid[self.x[:, i] >= self.barrier, i] = self.call_grid[
                        self.x[:, i] >= self.barrier, i
                    ]
        self.solved = True
