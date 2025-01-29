from dataclasses import dataclass
from typing import Optional, Any
import numpy as np

# @dataclass
# class DataGenConfig:
#     variables: list[str]
#     parameter_variables: list[str]
#     underlier_price_range: np.ndarray[Any, np.dtype[np.floating]]
#     expiry_time_range: np.ndarray[Any, np.dtype[np.floating]]
#     interest_rate: np.ndarray[Any, np.dtype[np.floating]]
#     volatility: np.ndarray[Any, np.dtype[np.floating]]
#     strike: Optional[np.ndarray[Any, np.dtype[np.floating]]] = None
#     derived_variables: Optional[dict[str, callable]] = None
#     normalize: bool = True
#     n: int = 10 ** 6 # number of observations
#     m: Optional[int] = 10 ** 4 # number of pricing instances
#     x_step: Optional[int] = None
#     t_step: Optional[int] = None
#     seed: Optional[int] = 42
#     price_points_sampling_std_pct: Optional[float] = None

@dataclass
class DataGenConfig:
    variables: list[str]
    underlier_price: np.ndarray[Any, np.dtype[np.floating]]
    expiry: np.ndarray[Any, np.dtype[np.floating]]
    interest_rate: np.ndarray[Any, np.dtype[np.floating]]
    volatility: np.ndarray[Any, np.dtype[np.floating]]
    strike: Optional[np.ndarray[Any, np.dtype[np.floating]]] = None
    derived_variables: Optional[dict[str, callable]] = None
    normalize: bool = True
    n: int = 10 ** 6 # number of observations
    m: Optional[int] = 10 ** 4 # number of pricing instances
    x_step: Optional[int] = None
    t_step: Optional[int] = None
    seed: Optional[int] = 42
    price_points_sampling_std_pct: Optional[float] = None
    strike_lower_bound_pct: Optional[float] = 0.75
    strike_upper_bound_pct: Optional[float] = 1.25
    strike_sampling_std_pct: Optional[float] = 0.15
    parameter_variables: Optional[list[str]] = None
    barrier_distance_mean_pct: Optional[float] = 1.4
    barrier_sampling_std_pct: Optional[float] = 1.1
    barrier_distance_min_pct: Optional[float] = 0.15

@dataclass
class ModelConfig:
    model_class: Any
    neuron_per_layer: int
    layer_number: int
    hidden_layer_activation: str
    learning_rate: float
    batch_size: int
    epochs: int
    optimizer: Any
    l1: float
    l2: float
    dropout: float
    input_variables: list[str]
    target_variables: dict[str, str]
    greeks: dict[str, list[dict[str, str]]]
    tensorflow_greeks: dict[str, int]
    jacobian_batch_size: int

@dataclass
class PipeLineConfig:
    pricing_model: Any
    data_gen_func: callable
    data: DataGenConfig
    test_data: DataGenConfig
    model: ModelConfig
    regenerate_data: bool
    save_path: Optional[str] = None
    test_save_path: Optional[str] = None
