from dataclasses import dataclass
from typing import Optional, Any
import datetime

@dataclass
class VariableConfig:
    name: str
    distribution: Optional[str] = "uniform"
    lower_bound: Optional[float] = 0.0
    upper_bound: Optional[float] = 1.0
    mean: Optional[float] = 0.5
    std: Optional[float] = 0.1
    lower_clip: Optional[float] = None
    upper_clip: Optional[float] = None
    mean_override: Optional[str] = None
    generator_function: Optional[callable] = None

@dataclass
class DataGeneratorConfig:
    pricer: Any
    pricer_config: Any
    n: int
    m: int
    parameter_variables: list[VariableConfig]
    additional_variables: list[VariableConfig]
    derived_variables: list[VariableConfig]
    black_scholes_normalize: Optional[bool] = True
    seed: Optional[int] = int(datetime.date.today().strftime("%Y%m%d"))
    data_run_type: Optional[str] = "train"
    feller_condition: Optional[bool] = False
    implied_volatility: Optional[bool] = False
