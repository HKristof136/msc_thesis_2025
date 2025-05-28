import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pricer.analytical as analytical
import pricer.pde_solver as pde_solver
from neural_network.torch_model import PricerNetTorch
from neural_network.config_base import PipeLineConfig, ModelConfig
from data_generator.config import DataGeneratorConfig, VariableConfig
from data_generator.data_generator_base import DataGenerator
from neural_network.utils import (
    price_function, delta_function, gamma_function,
    vega_function, theta_function, rho_function,
    partial_deriv_correlation_function, partial_deriv_kappa_function,
    partial_deriv_sigma_function, partial_deriv_theta_function
)

common_model_config = ModelConfig(
        model_class=PricerNetTorch,
        neuron_per_layer=32,
        layer_number=5,
        hidden_layer_activation="tanh",
        learning_rate=0.001,
        batch_size=2 ** 8,
        epochs=15,
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
        lambda_param=1.0,
        jacobian_batch_size=5 * 10 ** 4,
        calc_greek_regularization=True,
        greek_weighting=False,
    )

common_barrier_config = ModelConfig(
        model_class=PricerNetTorch,
        neuron_per_layer=32,
        layer_number=5,
        hidden_layer_activation="tanh",
        learning_rate=0.005,
        batch_size=2 ** 8,
        epochs=100,
        optimizer="SGD",
        l1=0.0,
        l2=0.0,
        dropout=0.0,
        input_variables=['underlier_price', 'expiry', 'volatility', 'interest_rate', 'barrier'],
        target_variables=["price"],
        greeks={
            'delta': (0, 0),
            'theta': (1, 1),
            'vega': (2, 2),
            'rho': (3, 3)
        },
        lambda_param=0.05,
        jacobian_batch_size=5 * 10 ** 4,
        calc_greek_regularization=True,
        greek_weighting=False,
    )

def common_bs_vanilla_train_data_generator(model, n, run_type, implied_vol=False):
    return DataGenerator(
        DataGeneratorConfig(
            pricer=model,
            pricer_config=analytical.BlackScholesConfig,
            n=n,
            m=1,
            parameter_variables=[
                VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
                VariableConfig("volatility", distribution="uniform", lower_bound=0.10, upper_bound=0.36),
                VariableConfig("interest_rate", distribution="uniform", lower_bound=0.01, upper_bound=0.21),
                VariableConfig("expiry", distribution="uniform", lower_bound=7 / 365, upper_bound=365 / 365),
            ],
            additional_variables=[
                VariableConfig("underlier_price", distribution="normal", lower_clip=0.25, upper_clip=2.5, std=0.1,
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
            data_run_type=run_type,
            implied_volatility=implied_vol
        )
    )

def common_bs_vanilla_test_data_generator(model, n, implied_vol=False):
    return DataGenerator(
        DataGeneratorConfig(
            pricer=model,
            pricer_config=analytical.BlackScholesConfig,
            n=n,
            m=1,
            parameter_variables=[
                VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
                VariableConfig("volatility", distribution="uniform", lower_bound=0.15, upper_bound=0.26),
                VariableConfig("interest_rate", distribution="uniform", lower_bound=0.05, upper_bound=0.16),
                VariableConfig("expiry", distribution="uniform", lower_bound=14 / 365, upper_bound=300 / 365),
            ],
            additional_variables=[
                VariableConfig("underlier_price", distribution="normal", lower_clip=0.25, upper_clip=2.25, std=0.05,
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
            data_run_type="test",
            implied_volatility=implied_vol
        )
    )

def common_bs_barrier_up_train_data_generator(model, n, m, b_low=1.1, b_high=1.8, run_type="train"):
    return DataGenerator(
        DataGeneratorConfig(
            pricer=model,
            pricer_config=pde_solver.PDESolverConfig,
            n=n,
            m=m,
            parameter_variables=[
                VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
                VariableConfig("volatility", distribution="uniform", lower_bound=0.1, upper_bound=0.36),
                VariableConfig("interest_rate", distribution="uniform", lower_bound=0.01, upper_bound=0.21),
                VariableConfig("pricer_expiry", distribution="uniform", lower_bound=60 / 365, upper_bound=365 / 365),
                VariableConfig("barrier", distribution="uniform", lower_bound=b_low, upper_bound=b_high),
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
            data_run_type=run_type
        )
    )

def common_bs_barrier_up_test_data_generator(model, n, m, b_low=1.2, b_high=1.7):
    return DataGenerator(
        DataGeneratorConfig(
            pricer=model,
            pricer_config=pde_solver.PDESolverConfig,
            n=n,
            m=m,
            parameter_variables=[
                VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
                VariableConfig("volatility", distribution="uniform", lower_bound=0.15, upper_bound=0.26),
                VariableConfig("interest_rate", distribution="uniform", lower_bound=0.05, upper_bound=0.16),
                VariableConfig("pricer_expiry", distribution="uniform", lower_bound=75 / 365, upper_bound=300 / 365),
                VariableConfig("barrier", distribution="uniform", lower_bound=b_low, upper_bound=b_high),
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
            data_run_type="test"
        )
    )

def common_bs_barrier_down_train_data_generator(model, n, m, b_low=1 / 1.8, b_high=1 / 1.1, run_type="train"):
    return DataGenerator(
        DataGeneratorConfig(
            pricer=model,
            pricer_config=pde_solver.PDESolverConfig,
            n=n,
            m=m,
            parameter_variables=[
                VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
                VariableConfig("volatility", distribution="uniform", lower_bound=0.1, upper_bound=0.36),
                VariableConfig("interest_rate", distribution="uniform", lower_bound=0.01, upper_bound=0.21),
                VariableConfig("pricer_expiry", distribution="uniform", lower_bound=60 / 365, upper_bound=365 / 365),
                VariableConfig("barrier", distribution="uniform", lower_bound=b_low, upper_bound=b_high),
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
            data_run_type=run_type
        )
    )

def common_bs_barrier_down_test_data_generator(model, n, m, b_low=1 / 1.8, b_high=1 / 1.1):
    return DataGenerator(
        DataGeneratorConfig(
            pricer=model,
            pricer_config=pde_solver.PDESolverConfig,
            n=n,
            m=m,
            parameter_variables=[
                VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
                VariableConfig("volatility", distribution="uniform", lower_bound=0.15, upper_bound=0.26),
                VariableConfig("interest_rate", distribution="uniform", lower_bound=0.05, upper_bound=0.16),
                VariableConfig("pricer_expiry", distribution="uniform", lower_bound=75 / 365, upper_bound=300 / 365),
                VariableConfig("barrier", distribution="uniform", lower_bound=b_low, upper_bound=b_high),
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
            data_run_type="test"
        )
    )

bs_call_pipeline_config = PipeLineConfig(
    pricing_model=analytical.BlackScholesCall,
    train_data=common_bs_vanilla_train_data_generator(analytical.BlackScholesCall, 10 ** 6, "train"),
    validation_data=common_bs_vanilla_train_data_generator(analytical.BlackScholesCall, 10 ** 4, "validation"),
    test_data=common_bs_vanilla_test_data_generator(analytical.BlackScholesCall, 10 ** 6),
    model=common_model_config,
    regenerate_data=False,
)

bs_put_pipeline_config = PipeLineConfig(
    pricing_model=analytical.BlackScholesPut,
    train_data=common_bs_vanilla_train_data_generator(analytical.BlackScholesPut, 10 ** 6, "train"),
    validation_data=common_bs_vanilla_train_data_generator(analytical.BlackScholesPut, 10 ** 4, "validation"),
    test_data=common_bs_vanilla_test_data_generator(analytical.BlackScholesPut, 10 ** 6),
    model=common_model_config,
    regenerate_data=False,
)

bs_uo_call_pipeline_config = PipeLineConfig(
    pricing_model=pde_solver.BarrierUpAndOutCallPDE,
    train_data=common_bs_barrier_up_train_data_generator(pde_solver.BarrierUpAndOutCallPDE, 10 ** 5, 10),
    validation_data=common_bs_barrier_up_train_data_generator(pde_solver.BarrierUpAndOutCallPDE, 10 ** 3, 10, run_type="validation"),
    test_data=common_bs_barrier_up_test_data_generator(pde_solver.BarrierUpAndOutCallPDE, 10 ** 4, 10),
    model=common_barrier_config,
    regenerate_data=False,
)

bs_ui_put_pipeline_config = PipeLineConfig(
    pricing_model=pde_solver.BarrierUpAndInPutPDE,
    train_data=common_bs_barrier_up_train_data_generator(pde_solver.BarrierUpAndInPutPDE, 10 ** 5, 10,
                                                         b_low=1 / 1.8, b_high=1 / 1.1),
    validation_data=common_bs_barrier_up_train_data_generator(pde_solver.BarrierUpAndInPutPDE, 10 ** 3, 10,
                                                              b_low=1 / 1.8, b_high=1 / 1.1, run_type="validation"),
    test_data=common_bs_barrier_up_test_data_generator(pde_solver.BarrierUpAndInPutPDE, 10 ** 4, 10,
                                                       b_low=1 / 1.7, b_high=1 / 1.2),
    model=common_barrier_config,
    regenerate_data=False,
)

bs_di_call_pipeline_config = PipeLineConfig(
    pricing_model=pde_solver.BarrierDownAndInCallPDE,
    train_data=common_bs_barrier_down_train_data_generator(pde_solver.BarrierDownAndInCallPDE, 10 ** 5, 10,
                                                           b_low=1.1, b_high=1.8),
    validation_data=common_bs_barrier_down_train_data_generator(pde_solver.BarrierDownAndInCallPDE, 10 ** 3, 10,
                                                           b_low=1.1, b_high=1.8, run_type="validation"),
    test_data=common_bs_barrier_down_test_data_generator(pde_solver.BarrierDownAndInCallPDE, 10 ** 4, 10,
                                                           b_low=1.2, b_high=1.7),
    model=common_barrier_config,
    regenerate_data=False,
)

bs_do_put_pipeline_config = PipeLineConfig(
    pricing_model=pde_solver.BarrierDownAndOutPutPDE,
    train_data=common_bs_barrier_down_train_data_generator(pde_solver.BarrierDownAndOutPutPDE, 10 ** 5, 10),
    validation_data=common_bs_barrier_down_train_data_generator(pde_solver.BarrierDownAndOutPutPDE, 10 ** 3, 10, run_type="validation"),
    test_data=common_bs_barrier_down_test_data_generator(pde_solver.BarrierDownAndOutPutPDE, 10 ** 4, 10),
    model=common_barrier_config,
    regenerate_data=False,
)


heston_call_pipeline_config = PipeLineConfig(
    pricing_model=pde_solver.ADICallPDE,
    train_data=DataGenerator(
        DataGeneratorConfig(
            pricer=pde_solver.ADICallPDE,
            pricer_config=pde_solver.PDESolverConfig,
            n=10 ** 4,
            m=100,
            parameter_variables=[
                VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
                VariableConfig("interest_rate", distribution="uniform", lower_bound=0.01, upper_bound=0.11),
                VariableConfig("pricer_expiry", distribution="uniform", lower_bound=14 / 365, upper_bound=365 / 365),
                VariableConfig("corr", distribution="uniform", lower_bound=-0.5, upper_bound=0.1),
                VariableConfig("kappa", distribution="uniform", lower_bound=1.0, upper_bound=4.0),
                VariableConfig("variance_theta", distribution="uniform", lower_bound=0.10 ** 2, upper_bound=0.25 ** 2),
                VariableConfig("sigma", distribution="uniform", lower_bound=0.1, upper_bound=0.6)
            ],
            additional_variables=[],
            derived_variables=[
                VariableConfig("price", generator_function=price_function),
                VariableConfig("delta", generator_function=delta_function),
                VariableConfig("vega", generator_function=vega_function),
                VariableConfig("theta", generator_function=theta_function),
                VariableConfig("rho", generator_function=rho_function),
                VariableConfig("partial_deriv_correlation", generator_function=partial_deriv_correlation_function),
                VariableConfig("partial_deriv_kappa", generator_function=partial_deriv_kappa_function),
                VariableConfig("partial_deriv_theta", generator_function=partial_deriv_theta_function),
                VariableConfig("partial_deriv_sigma", generator_function=partial_deriv_sigma_function),
            ],
            black_scholes_normalize=False,
            data_run_type="train",
            feller_condition=True,
        )
    ),
    validation_data=DataGenerator(
        DataGeneratorConfig(
            pricer=pde_solver.ADICallPDE,
            pricer_config=pde_solver.PDESolverConfig,
            n=10 ** 3,
            m=100,
            parameter_variables=[
                VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
                VariableConfig("interest_rate", distribution="uniform", lower_bound=0.01, upper_bound=0.11),
                VariableConfig("pricer_expiry", distribution="uniform", lower_bound=14 / 365, upper_bound=365 / 365),
                VariableConfig("corr", distribution="uniform", lower_bound=-0.5, upper_bound=0.1),
                VariableConfig("kappa", distribution="uniform", lower_bound=1.0, upper_bound=4.0),
                VariableConfig("variance_theta", distribution="uniform", lower_bound=0.10 ** 2, upper_bound=0.25 ** 2),
                VariableConfig("sigma", distribution="uniform", lower_bound=0.1, upper_bound=0.6)
            ],
            additional_variables=[],
            derived_variables=[
                VariableConfig("price", generator_function=price_function),
                VariableConfig("delta", generator_function=delta_function),
                VariableConfig("vega", generator_function=vega_function),
                VariableConfig("theta", generator_function=theta_function),
                VariableConfig("rho", generator_function=rho_function),
                VariableConfig("partial_deriv_correlation", generator_function=partial_deriv_correlation_function),
                VariableConfig("partial_deriv_kappa", generator_function=partial_deriv_kappa_function),
                VariableConfig("partial_deriv_theta", generator_function=partial_deriv_theta_function),
                VariableConfig("partial_deriv_sigma", generator_function=partial_deriv_sigma_function),
            ],
            black_scholes_normalize=False,
            data_run_type="validation",
            feller_condition=True,
        )
    ),
    test_data=DataGenerator(
        DataGeneratorConfig(
            pricer=pde_solver.ADICallPDE,
            pricer_config=pde_solver.PDESolverConfig,
            n=10 ** 3,
            m=100,
            parameter_variables=[
                VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
                VariableConfig("interest_rate", distribution="uniform", lower_bound=0.01, upper_bound=0.11),
                VariableConfig("pricer_expiry", distribution="uniform", lower_bound=14 / 365, upper_bound=365 / 365),
                VariableConfig("corr", distribution="uniform", lower_bound=-0.5, upper_bound=0.1),
                VariableConfig("kappa", distribution="uniform", lower_bound=1.0, upper_bound=4.0),
                VariableConfig("variance_theta", distribution="uniform", lower_bound=0.10 ** 2, upper_bound=0.25 ** 2),
                VariableConfig("sigma", distribution="uniform", lower_bound=0.1, upper_bound=0.6)
            ],
            additional_variables=[],
            derived_variables=[
                VariableConfig("price", generator_function=price_function),
                VariableConfig("delta", generator_function=delta_function),
                VariableConfig("vega", generator_function=vega_function),
                VariableConfig("theta", generator_function=theta_function),
                VariableConfig("rho", generator_function=rho_function),
                VariableConfig("partial_deriv_correlation", generator_function=partial_deriv_correlation_function),
                VariableConfig("partial_deriv_kappa", generator_function=partial_deriv_kappa_function),
                VariableConfig("partial_deriv_theta", generator_function=partial_deriv_theta_function),
                VariableConfig("partial_deriv_sigma", generator_function=partial_deriv_sigma_function),
            ],
            black_scholes_normalize=False,
            data_run_type="test",
            feller_condition=True,
        )
    ),
    model=ModelConfig(
        model_class=PricerNetTorch,
        neuron_per_layer=32,
        layer_number=5,
        hidden_layer_activation="tanh",
        learning_rate=0.002,
        batch_size=2**11,
        epochs=100,
        optimizer="SGD",
        l1=0.0,
        l2=0.0,
        dropout=0.0,
        input_variables=["underlier_price", "expiry", "initial_variance", "interest_rate", "corr", "kappa", "variance_theta", "sigma"],
        target_variables=["price"],
        greeks={
            'delta': (0, 0),
            'theta': (1, 1),
            'vega': (2, 2),
            'rho': (3, 3),
            'partial_deriv_correlation': (4, 4),
            'partial_deriv_kappa': (5, 5),
            'partial_deriv_theta': (6, 6),
            'partial_deriv_sigma': (7, 7)
        },
        lambda_param=1.0,
        jacobian_batch_size=5 * 10 ** 4,
        calc_greek_regularization=True,
        greek_weighting=False,
    ),
    regenerate_data=False,
)

heston_barrier_uo_call_pipeline_config = PipeLineConfig(
    pricing_model=pde_solver.ADIBarrierUpAndOutCallPDE,
    train_data=DataGenerator(
        DataGeneratorConfig(
            pricer=pde_solver.ADIBarrierUpAndOutCallPDE,
            pricer_config=pde_solver.PDESolverConfig,
            n=10 ** 4,
            m=100,
            parameter_variables=[
                VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
                VariableConfig("barrier", distribution="uniform", lower_bound=1.1, upper_bound=1.8),
                VariableConfig("interest_rate", distribution="uniform", lower_bound=0.01, upper_bound=0.11),
                VariableConfig("pricer_expiry", distribution="uniform", lower_bound=120 / 365, upper_bound=365 / 365),
                VariableConfig("corr", distribution="uniform", lower_bound=-0.5, upper_bound=0.1),
                VariableConfig("kappa", distribution="uniform", lower_bound=1.0, upper_bound=4.0),
                VariableConfig("variance_theta", distribution="uniform", lower_bound=0.10 ** 2, upper_bound=0.25 ** 2),
                VariableConfig("sigma", distribution="uniform", lower_bound=0.1, upper_bound=0.6)
            ],
            additional_variables=[],
            derived_variables=[
                VariableConfig("price", generator_function=price_function),
                VariableConfig("delta", generator_function=delta_function),
                VariableConfig("vega", generator_function=vega_function),
                VariableConfig("theta", generator_function=theta_function),
                VariableConfig("rho", generator_function=rho_function),
                VariableConfig("partial_deriv_correlation", generator_function=partial_deriv_correlation_function),
                VariableConfig("partial_deriv_kappa", generator_function=partial_deriv_kappa_function),
                VariableConfig("partial_deriv_theta", generator_function=partial_deriv_theta_function),
                VariableConfig("partial_deriv_sigma", generator_function=partial_deriv_sigma_function),
            ],
            black_scholes_normalize=False,
            data_run_type="train",
            feller_condition=True,
        )
    ),
    validation_data=DataGenerator(
        DataGeneratorConfig(
            pricer=pde_solver.ADIBarrierUpAndOutCallPDE,
            pricer_config=pde_solver.PDESolverConfig,
            n=10 ** 3,
            m=100,
            parameter_variables=[
                VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
                VariableConfig("barrier", distribution="uniform", lower_bound=1.1, upper_bound=1.8),
                VariableConfig("interest_rate", distribution="uniform", lower_bound=0.01, upper_bound=0.11),
                VariableConfig("pricer_expiry", distribution="uniform", lower_bound=120 / 365, upper_bound=365 / 365),
                VariableConfig("corr", distribution="uniform", lower_bound=-0.5, upper_bound=0.1),
                VariableConfig("kappa", distribution="uniform", lower_bound=1.0, upper_bound=4.0),
                VariableConfig("variance_theta", distribution="uniform", lower_bound=0.10 ** 2, upper_bound=0.25 ** 2),
                VariableConfig("sigma", distribution="uniform", lower_bound=0.1, upper_bound=0.6)
            ],
            additional_variables=[],
            derived_variables=[
                VariableConfig("price", generator_function=price_function),
                VariableConfig("delta", generator_function=delta_function),
                VariableConfig("vega", generator_function=vega_function),
                VariableConfig("theta", generator_function=theta_function),
                VariableConfig("rho", generator_function=rho_function),
                VariableConfig("partial_deriv_correlation", generator_function=partial_deriv_correlation_function),
                VariableConfig("partial_deriv_kappa", generator_function=partial_deriv_kappa_function),
                VariableConfig("partial_deriv_theta", generator_function=partial_deriv_theta_function),
                VariableConfig("partial_deriv_sigma", generator_function=partial_deriv_sigma_function),
            ],
            black_scholes_normalize=False,
            data_run_type="validation",
            feller_condition=True,
        )
    ),
    test_data=DataGenerator(
        DataGeneratorConfig(
            pricer=pde_solver.ADIBarrierUpAndOutCallPDE,
            pricer_config=pde_solver.PDESolverConfig,
            n=10 ** 3,
            m=100,
            parameter_variables=[
                VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
                VariableConfig("barrier", distribution="uniform", lower_bound=1.1, upper_bound=1.8),
                VariableConfig("interest_rate", distribution="uniform", lower_bound=0.01, upper_bound=0.11),
                VariableConfig("pricer_expiry", distribution="uniform", lower_bound=120 / 365, upper_bound=365 / 365),
                VariableConfig("corr", distribution="uniform", lower_bound=-0.5, upper_bound=0.1),
                VariableConfig("kappa", distribution="uniform", lower_bound=1.0, upper_bound=4.0),
                VariableConfig("variance_theta", distribution="uniform", lower_bound=0.10 ** 2, upper_bound=0.25 ** 2),
                VariableConfig("sigma", distribution="uniform", lower_bound=0.1, upper_bound=0.6)
            ],
            additional_variables=[],
            derived_variables=[
                VariableConfig("price", generator_function=price_function),
                VariableConfig("delta", generator_function=delta_function),
                VariableConfig("vega", generator_function=vega_function),
                VariableConfig("theta", generator_function=theta_function),
                VariableConfig("rho", generator_function=rho_function),
                VariableConfig("partial_deriv_correlation", generator_function=partial_deriv_correlation_function),
                VariableConfig("partial_deriv_kappa", generator_function=partial_deriv_kappa_function),
                VariableConfig("partial_deriv_theta", generator_function=partial_deriv_theta_function),
                VariableConfig("partial_deriv_sigma", generator_function=partial_deriv_sigma_function),
            ],
            black_scholes_normalize=False,
            data_run_type="test",
            feller_condition=True,
        )
    ),
    model=ModelConfig(
        model_class=PricerNetTorch,
        neuron_per_layer=32,
        layer_number=5,
        hidden_layer_activation="tanh",
        learning_rate=0.001,
        batch_size=2 ** 11,
        epochs=50,
        optimizer="SGD",
        l1=0.0,
        l2=0.0,
        dropout=0.0,
        input_variables=["underlier_price", "expiry", "initial_variance", "interest_rate", "corr", "kappa", "variance_theta", "sigma", "barrier"],
        target_variables=["price"],
        greeks={
            'delta': (0, 0),
            'theta': (1, 1),
            'vega': (2, 2),
            'rho': (3, 3),
            'partial_deriv_correlation': (4, 4),
            'partial_deriv_kappa': (5, 5),
            'partial_deriv_theta': (6, 6),
            'partial_deriv_sigma': (7, 7)
        },
        lambda_param=1.0,
        jacobian_batch_size=5 * 10 ** 4,
        calc_greek_regularization=True,
        greek_weighting=False,
    ),
    regenerate_data=False,
)

implied_vol_pipeline_config = PipeLineConfig(
    pricing_model=analytical.ImpliedVol,
    train_data=common_bs_vanilla_train_data_generator(analytical.BlackScholesCall, 10 ** 6, "train", implied_vol=True),
    validation_data=common_bs_vanilla_train_data_generator(analytical.BlackScholesCall, 10 ** 4, "validation", implied_vol=True),
    test_data=common_bs_vanilla_test_data_generator(analytical.BlackScholesCall, 10 ** 4, implied_vol=True),
    model=ModelConfig(
        model_class=PricerNetTorch,
        neuron_per_layer=32,
        layer_number=5,
        hidden_layer_activation="tanh",
        learning_rate=0.001,
        batch_size=2 ** 11,
        epochs=100,
        optimizer="SGD",
        l1=0.0,
        l2=0.0,
        dropout=0.0,
        input_variables=["log_time_value",'underlier_price', 'expiry', 'interest_rate'],
        target_variables=["volatility"],
        greeks={
            "modified_vega": (0, 0)
        },
        lambda_param=0.1,
        jacobian_batch_size=5 * 10 ** 4,
        calc_greek_regularization=False,
        greek_weighting=False,
    ),
    regenerate_data=False,
)

pipeline_configs = {
    "bs_call": bs_call_pipeline_config,
    "bs_put": bs_put_pipeline_config,
    "heston_uo_call": heston_barrier_uo_call_pipeline_config,
    "bs_uo_call": bs_uo_call_pipeline_config,
    "bs_ui_put": bs_ui_put_pipeline_config,
    "bs_di_call": bs_di_call_pipeline_config,
    "bs_do_put": bs_do_put_pipeline_config,
    "heston_call": heston_call_pipeline_config,
    "implied_vol": implied_vol_pipeline_config,
}

if __name__ == "__main__":
    for key, config in pipeline_configs.items():
        df = config.train_data.get_data()
        os.makedirs(f"data/{config.train_data.seed}", exist_ok=True)
        df.to_parquet(f"data/{config.train_data.seed}/{config.pricing_model.__name__}_train.parquet")

        df = config.validation_data.get_data()
        df.to_parquet(f"data/{config.train_data.seed}/{config.pricing_model.__name__}_validation.parquet")

        df = config.test_data.get_data()
        df.to_parquet(f"data/{config.train_data.seed}/{config.pricing_model.__name__}_test.parquet")
