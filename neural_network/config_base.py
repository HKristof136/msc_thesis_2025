from dataclasses import dataclass
from typing import Optional, Any
from data_generator.data_generator_base import DataGenerator


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
    target_variables: list[str]
    greeks: dict[str, tuple[str]]
    lambda_param: float
    jacobian_batch_size: int
    calc_greek_regularization: bool
    greek_weighting: bool

@dataclass
class PipeLineConfig:
    pricing_model: Any
    train_data: DataGenerator
    validation_data: DataGenerator
    test_data: DataGenerator
    model: ModelConfig
    regenerate_data: bool
    train_save_path: Optional[str] = None
    validation_save_path: Optional[str] = None
    test_save_path: Optional[str] = None
