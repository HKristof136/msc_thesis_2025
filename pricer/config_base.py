from dataclasses import dataclass
from typing import Union, Any, Optional
import numpy as np

@dataclass
class BlackScholesConfig:
    underlier_price: Union[float, np.ndarray[Any, np.dtype[np.floating]]]
    strike: Union[float, np.ndarray[Any, np.dtype[np.floating]]]
    expiry: Union[float, np.ndarray[Any, np.dtype[np.floating]]]
    interest_rate: Union[float, np.ndarray[Any, np.dtype[np.floating]]]
    volatility: Union[float, np.ndarray[Any, np.dtype[np.floating]]]
    barrier: Union[float, np.ndarray[Any, np.dtype[np.floating]]] = None

@dataclass
class PDESolverConfig:
    underlier_price_grid: np.ndarray[Any, np.dtype[np.floating]]
    time_grid: np.ndarray[Any, np.dtype[np.floating]]
    strike: float
    interest_rate: float
    volatility: float = None
    barrier: float = None
    verbose: bool = True
    foreign_interest_rate: Optional[float] = 0.0
    variance_grid: Optional[np.ndarray[Any, np.dtype[np.floating]]] = None
    corr: Optional[float] = None
    kappa: Optional[float] = None
    variance_theta: Optional[float] = None
    sigma: Optional[float] = None
    adi_param: Optional[float] = 0.5
    vectorize_solver: Optional[bool] = True
    n: Optional[int] = 26
    m: Optional[int] = 51

class DefaultConfig:
    black_scholes: BlackScholesConfig = BlackScholesConfig(
        underlier_price=np.linspace(0.01, 200, 1000, dtype=np.float32),
        strike=100.0,
        expiry=1.0,
        interest_rate=0.05,
        volatility=0.2
    )
    black_scholes_barrier_call: BlackScholesConfig = BlackScholesConfig(
        underlier_price=np.linspace(0.01, 200, 1000, dtype=np.float32),
        strike=100.0,
        expiry=1.0,
        interest_rate=0.05,
        volatility=0.2,
        barrier=110.0
    )
    black_scholes_barrier_put: BlackScholesConfig = BlackScholesConfig(
        underlier_price=np.linspace(0.01, 200, 1000, dtype=np.float32),
        strike=100.0,
        expiry=1.0,
        interest_rate=0.05,
        volatility=0.2,
        barrier=90.0
    )
    pde_solver: PDESolverConfig = PDESolverConfig(
        underlier_price_grid=np.linspace(0.01, 200, 1000, dtype=np.float32),
        time_grid=np.linspace(0.0, 1.0, 1000, dtype=np.float32),
        strike=100.0,
        interest_rate=0.05,
        volatility=0.2
    )
    pde_solver_barrier_call: PDESolverConfig = PDESolverConfig(
        underlier_price_grid=np.linspace(0.01, 200, 1000, dtype=np.float32),
        time_grid=np.linspace(0.0, 1.0, 1000, dtype=np.float32),
        strike=100.0,
        interest_rate=0.05,
        volatility=0.2,
        barrier=110.0
    )
    pde_solver_barrier_put: PDESolverConfig = PDESolverConfig(
        underlier_price_grid=np.linspace(0.01, 200, 1000, dtype=np.float32),
        time_grid=np.linspace(0.0, 1.0, 1000, dtype=np.float32),
        strike=100.0,
        interest_rate=0.05,
        volatility=0.2,
        barrier=90.0
    )
