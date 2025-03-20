import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pricer.analytical import BlackScholesCall, BlackScholesPut
from pricer.pde_solver import (
    BarrierUpAndOutCallPDE, BarrierUpAndOutPutPDE,
    BarrierDownAndOutCallPDE, BarrierDownAndOutPutPDE,
)
from pricer.config_base import BlackScholesConfig, PDESolverConfig
from neural_network.torch_model import PricerNetTorch
from config_base import PipeLineConfig, ModelConfig
from data_generator.config import DataGeneratorConfig, VariableConfig
from data_generator.data_generator_base import DataGenerator
from utils import (
    price_function, delta_function, gamma_function,
    vega_function, theta_function, rho_function,
)

common_model_config = ModelConfig(
        model_class=PricerNetTorch,
        neuron_per_layer=32,
        layer_number=5,
        hidden_layer_activation="tanh",
        learning_rate=0.03,
        batch_size=2 ** 10,
        epochs=5,
        optimizer="SGD",
        l1=0.0,
        l2=0.0,
        dropout=0.0,
        input_variables=['underlier_price', 'expiry', 'volatility', 'interest_rate'],
        target_variables=["price"],
        greeks={
            'delta': (0, 0),
            'theta': (1, 1),
            'vega': (2, 2),
            'rho': (3, 3)
        },
        greeks_relative_weight=0.25,
        jacobian_batch_size=5 * 10 ** 4,
    )

common_barrier_config = ModelConfig(
        model_class=PricerNetTorch,
        neuron_per_layer=32,
        layer_number=5,
        hidden_layer_activation="tanh",
        learning_rate=0.03,
        batch_size=2 ** 14,
        epochs=10,
        optimizer="SGD",
        l1=0.0,
        l2=0.0,
        dropout=0.05,
        input_variables=["underlier_price", "interest_rate", "volatility", "expiry", "barrier"],
        target_variables={"price": "relu"},
        greeks={'delta': (0, 0), 'theta': (1, 1), 'vega': (2, 2), 'rho': (3, 3)},
        greeks_relative_weight=0.05,
        jacobian_batch_size=5 * 10 ** 4,
    )

bs_call_pipeline_config = PipeLineConfig(
    pricing_model=BlackScholesCall,
    train_data=DataGenerator(
        DataGeneratorConfig(
            pricer=BlackScholesCall,
            pricer_config=BlackScholesConfig,
            n=10 ** 6,
            m=1,
            parameter_variables=[
                VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
                VariableConfig("volatility", distribution="uniform", lower_bound=0.05, upper_bound=0.36),
                VariableConfig("interest_rate", distribution="uniform", lower_bound=0.01, upper_bound=0.21),
                VariableConfig("expiry", distribution="uniform", lower_bound=1 / 365, upper_bound=1.0),
            ],
            additional_variables=[
                VariableConfig("underlier_price", distribution="normal", lower_clip=0.1, upper_clip=1.9, std=0.05,
                               mean_override="discounted_strike"),
            ],
            derived_variables=[
                VariableConfig("price", generator_function=price_function),
                VariableConfig("delta", generator_function=delta_function),
                VariableConfig("gamma", generator_function=gamma_function),
                VariableConfig("vega", generator_function=vega_function),
                VariableConfig("theta", generator_function=theta_function),
                VariableConfig("rho", generator_function=rho_function),
            ],
            black_scholes_normalize=False,
        )
    ),
    validation_data=DataGenerator(
        DataGeneratorConfig(
            pricer=BlackScholesCall,
            pricer_config=BlackScholesConfig,
            n=10 ** 6,
            m=1,
            parameter_variables=[
                VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
                VariableConfig("volatility", distribution="uniform", lower_bound=0.05, upper_bound=0.36),
                VariableConfig("interest_rate", distribution="uniform", lower_bound=0.01, upper_bound=0.21),
                VariableConfig("expiry", distribution="uniform", lower_bound=1 / 365, upper_bound=1.0),
            ],
            additional_variables=[
                VariableConfig("underlier_price", distribution="normal", lower_clip=0.1, upper_clip=1.9, std=0.05,
                               mean_override="discounted_strike"),
            ],
            derived_variables=[
                VariableConfig("price", generator_function=price_function),
                VariableConfig("delta", generator_function=delta_function),
                VariableConfig("gamma", generator_function=gamma_function),
                VariableConfig("vega", generator_function=vega_function),
                VariableConfig("theta", generator_function=theta_function),
                VariableConfig("rho", generator_function=rho_function),
            ],
            black_scholes_normalize=False,
        )
    ),
    test_data=DataGenerator(
        DataGeneratorConfig(
            pricer=BlackScholesCall,
            pricer_config=BlackScholesConfig,
            n=10 ** 6,
            m=1,
            parameter_variables=[
                VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
                VariableConfig("volatility", distribution="uniform", lower_bound=0.10, upper_bound=0.26),
                VariableConfig("interest_rate", distribution="uniform", lower_bound=0.05, upper_bound=0.16),
                VariableConfig("expiry", distribution="uniform", lower_bound=14 / 365, upper_bound=300 / 365),
            ],
            additional_variables=[
                VariableConfig("underlier_price", distribution="normal", lower_clip=0.25, upper_clip=1.75, std=0.05,
                               mean_override="discounted_strike"),
            ],
            derived_variables=[
                VariableConfig("price", generator_function=price_function),
                VariableConfig("delta", generator_function=delta_function),
                VariableConfig("gamma", generator_function=gamma_function),
                VariableConfig("vega", generator_function=vega_function),
                VariableConfig("theta", generator_function=theta_function),
                VariableConfig("rho", generator_function=rho_function),
            ],
            black_scholes_normalize=False,
        )
    ),
    model=common_model_config,
    regenerate_data=False,
)

bs_put_pipeline_config = PipeLineConfig(
    pricing_model=BlackScholesPut,
    train_data=DataGenerator(
        DataGeneratorConfig(
            pricer=BlackScholesCall,
            pricer_config=BlackScholesConfig,
            n=5 * 10 ** 6,
            m=1,
            parameter_variables=[
                VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
                VariableConfig("volatility", distribution="uniform", lower_bound=0.05, upper_bound=0.36),
                VariableConfig("interest_rate", distribution="uniform", lower_bound=0.01, upper_bound=0.21),
                VariableConfig("expiry", distribution="uniform", lower_bound=1 / 365, upper_bound=1.0),
            ],
            additional_variables=[
                VariableConfig("underlier_price", distribution="normal", lower_clip=0.6, upper_clip=1.4, std=0.025,
                               mean_override="discounted_strike"),
            ],
            derived_variables=[
                VariableConfig("price", generator_function=price_function),
                VariableConfig("delta", generator_function=delta_function),
                VariableConfig("gamma", generator_function=gamma_function),
                VariableConfig("vega", generator_function=vega_function),
                VariableConfig("theta", generator_function=theta_function),
                VariableConfig("rho", generator_function=rho_function),
            ],
            black_scholes_normalize=False,
        )
    ),
    validation_data=DataGenerator(
        DataGeneratorConfig(
            pricer=BlackScholesCall,
            pricer_config=BlackScholesConfig,
            n=10 ** 6,
            m=1,
            parameter_variables=[
                VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
                VariableConfig("volatility", distribution="uniform", lower_bound=0.05, upper_bound=0.36),
                VariableConfig("interest_rate", distribution="uniform", lower_bound=0.01, upper_bound=0.21),
                VariableConfig("expiry", distribution="uniform", lower_bound=1 / 365, upper_bound=1.0),
            ],
            additional_variables=[
                VariableConfig("underlier_price", distribution="normal", lower_clip=0.6, upper_clip=1.4, std=0.025,
                               mean_override="discounted_strike"),
            ],
            derived_variables=[
                VariableConfig("price", generator_function=price_function),
                VariableConfig("delta", generator_function=delta_function),
                VariableConfig("gamma", generator_function=gamma_function),
                VariableConfig("vega", generator_function=vega_function),
                VariableConfig("theta", generator_function=theta_function),
                VariableConfig("rho", generator_function=rho_function),
            ],
            black_scholes_normalize=False,
        )
    ),
    test_data=DataGenerator(
        DataGeneratorConfig(
            pricer=BlackScholesCall,
            pricer_config=BlackScholesConfig,
            n=10 ** 6,
            m=1,
            parameter_variables=[
                VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
                VariableConfig("volatility", distribution="uniform", lower_bound=0.10, upper_bound=0.26),
                VariableConfig("interest_rate", distribution="uniform", lower_bound=0.05, upper_bound=0.16),
                VariableConfig("expiry", distribution="uniform", lower_bound=14 / 365, upper_bound=300 / 365),
            ],
            additional_variables=[
                VariableConfig("underlier_price", distribution="normal", lower_clip=0.75, upper_clip=1.25, std=0.025,
                               mean_override="discounted_strike"),
            ],
            derived_variables=[
                VariableConfig("price", generator_function=price_function),
                VariableConfig("delta", generator_function=delta_function),
                VariableConfig("gamma", generator_function=gamma_function),
                VariableConfig("vega", generator_function=vega_function),
                VariableConfig("theta", generator_function=theta_function),
                VariableConfig("rho", generator_function=rho_function),
            ],
            black_scholes_normalize=False,
        )
    ),
    model=common_model_config,
    regenerate_data=False,
)

bs_uo_call_pipeline_config = PipeLineConfig(
    pricing_model=BarrierUpAndOutCallPDE,
    train_data=DataGenerator(
        DataGeneratorConfig(
            pricer=BarrierUpAndOutCallPDE,
            pricer_config=PDESolverConfig,
            n=5 * 10 ** 4,
            m=10 ** 2,
            parameter_variables=[
                VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
                VariableConfig("barrier", distribution="uniform", lower_bound=1.1, upper_bound=1.8),
                VariableConfig("volatility", distribution="uniform", lower_bound=0.05, upper_bound=0.36),
                VariableConfig("interest_rate", distribution="uniform", lower_bound=0.01, upper_bound=0.21),
                VariableConfig("pricer_expiry", distribution="uniform", lower_bound=30 / 365, upper_bound=1.0),
            ],
            additional_variables=[],
            derived_variables=[
                VariableConfig("price", generator_function=price_function),
                VariableConfig("delta", generator_function=delta_function),
                VariableConfig("gamma", generator_function=gamma_function),
                VariableConfig("vega", generator_function=vega_function),
                VariableConfig("theta", generator_function=theta_function),
                VariableConfig("rho", generator_function=rho_function),
            ],
            black_scholes_normalize=False,
        )
    ),
    validation_data=DataGenerator(
        DataGeneratorConfig(
            pricer=BarrierUpAndOutCallPDE,
            pricer_config=PDESolverConfig,
            n=10 ** 3,
            m=10 ** 2,
            parameter_variables=[
                VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
                VariableConfig("barrier", distribution="uniform", lower_bound=1.1, upper_bound=1.8),
                VariableConfig("volatility", distribution="uniform", lower_bound=0.05, upper_bound=0.36),
                VariableConfig("interest_rate", distribution="uniform", lower_bound=0.01, upper_bound=0.21),
                VariableConfig("pricer_expiry", distribution="uniform", lower_bound=30 / 365, upper_bound=1.0),
            ],
            additional_variables=[],
            derived_variables=[
                VariableConfig("price", generator_function=price_function),
                VariableConfig("delta", generator_function=delta_function),
                VariableConfig("gamma", generator_function=gamma_function),
                VariableConfig("vega", generator_function=vega_function),
                VariableConfig("theta", generator_function=theta_function),
                VariableConfig("rho", generator_function=rho_function),
            ],
            black_scholes_normalize=False,
        )
    ),
    test_data=DataGenerator(
        DataGeneratorConfig(
            pricer=BarrierUpAndOutCallPDE,
            pricer_config=PDESolverConfig,
            n=10 ** 4,
            m=10 ** 2,
            parameter_variables=[
                VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
                VariableConfig("barrier", distribution="uniform", lower_bound=1.2, upper_bound=1.7),
                VariableConfig("volatility", distribution="uniform", lower_bound=0.10, upper_bound=0.26),
                VariableConfig("interest_rate", distribution="uniform", lower_bound=0.05, upper_bound=0.16),
                VariableConfig("pricer_expiry", distribution="uniform", lower_bound=61 / 365, upper_bound=300 / 365),
            ],
            additional_variables=[],
            derived_variables=[
                VariableConfig("price", generator_function=price_function),
                VariableConfig("delta", generator_function=delta_function),
                VariableConfig("gamma", generator_function=gamma_function),
                VariableConfig("vega", generator_function=vega_function),
                VariableConfig("theta", generator_function=theta_function),
                VariableConfig("rho", generator_function=rho_function),
            ],
            black_scholes_normalize=False,
        )
    ),
    model=common_barrier_config,
    regenerate_data=False,
)

pipeline_configs = {
    "bs_call": bs_call_pipeline_config,
    "bs_put": bs_put_pipeline_config,
    "bs_uo_call": bs_uo_call_pipeline_config,
}

