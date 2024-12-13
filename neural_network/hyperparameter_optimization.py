import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from pricer.analytical import BlackScholesCall
from functools import partial
from neural_network.data_gen import bs_call_data_gen
from neural_network.pipeline import Pipeline, partial_relu

config = {
    "pricing_model": BlackScholesCall,
    "data_gen_func": bs_call_data_gen,
    "regenerate_data": False,
    "data": {
        "n": 10**6,
        "seed": 8,
        "underlier_price": np.linspace(50, 250, 1001),
        "interest_rate": np.arange(0.01, 0.21, 0.001),
        "volatility": np.arange(0.1, 0.41, 0.001),
        "expiry": np.arange(30, 400) / 365,
        "strike_lower_bound_scalar": 0.9,
        "strike_upper_bound_scalar": 1.25,
        "strike_sampling_variance": 20**2,
        "unique_strike_number": 200,
        "variables": [
            "underlier_price",
            "strike",
            "expiry",
            "interest_rate",
            "volatility",
        ],
        "derived_variables": {
            "price": lambda x: x.price(),
            "delta": lambda x: x.delta(),
            "gamma": lambda x: x.gamma(),
            "vega": lambda x: x.vega(),
            "theta": lambda x: x.theta(),
            "rho": lambda x: x.rho(),
        },
        "normalize": False,
    },
    "test_data": {
        "n": 200000,
        "seed": 10,
        "underlier_price": np.linspace(75, 225, 1001),
        "interest_rate": np.arange(0.05, 0.16, 0.001),
        "volatility": np.arange(0.15, 0.31, 0.001),
        "expiry": np.arange(45, 366) / 365,
        "strike": np.linspace(75, 225, 1001),
        "strike_lower_bound_scalar": 1,
        "strike_upper_bound_scalar": 1,
        "strike_sampling_variance": 15**2,
        "unique_strike_number": 200,
        "variables": [
            "underlier_price",
            "strike",
            "expiry",
            "interest_rate",
            "volatility",
        ],
        "derived_variables": {
            "price": lambda x: x.price(),
            "delta": lambda x: x.delta(),
            "gamma": lambda x: x.gamma(),
            "vega": lambda x: x.vega(),
            "theta": lambda x: x.theta(),
            "rho": lambda x: x.rho(),
        },
        "normalize": False,
    },
    "model": {
        "model_class": tf.keras.models.Model,
        "loss_weights": [3.0, 1.0],
        "neuron_per_layer": 32,
        "layer_number": 2,
        "learning_rate": 0.005,
        "batch_size": 2**11,
        "epochs": 20,
        "l1": 0,
        "l2": 1,
        "dropout": 0,
        "input_variables": [
            "underlier_price",
            "strike",
            "expiry",
            "interest_rate",
            "volatility",
        ],
        "target_variables": {
            "price": "relu",
            "delta": partial_relu,
        },
        "greeks": {
            "underlier_price": [
                {"name": "delta", "order": "first"},
                {"name": "gamma", "order": "second"},
            ],
            "expiry": [{"name": "theta", "order": "first"}],
            "volatility": [{"name": "vega", "order": "first"}],
            "interest_rate": [{"name": "rho", "order": "first"}],
        },
        "jacobian_batch_size": 10**5,
    },
}

parameter_space = {
    "neuron_per_layer": [8, 16, 32, 64],
    "layer_number": [2, 3, 4, 5],
    "learning_rate": [0.001, 0.003, 0.005],
    "l1": [0, 0.25, 0.5, 0.75, 1],
    "l2": [0, 0.5, 1, 1.5, 2, 5, 10],
    "dropout": [0, 0.1, 0.2, 0.3],
    "loss_weights": [[3.0, 1.0], [2.0, 1.0], [1.0, 1.0]],
}

def sample_parameters(parameter_space, n_samples=100):
    samples = []
    for _ in range(n_samples):
        sample = {key: random.choice(values) for key, values in parameter_space.items()}
        samples.append(sample)
    return samples

evaluations = []

samples = sample_parameters(parameter_space, 20)
for i, sample in enumerate(samples):
    run_config = config.copy()
    run_config["model"].update(sample)
    model_save_path = f'saved_models/{i+1}_{config['pricing_model'].__name__}.keras'

    s = time.perf_counter()
    pipeline = Pipeline(run_config)
    pipeline.train()
    error_dict = pipeline.evaluate()
    sample.update(error_dict)
    sample["time"] = time.perf_counter() - s
    sample["model_save_path"] = model_save_path
    evaluations.append(sample)
    
    pipeline.model.save(model_save_path)

df = pd.DataFrame(evaluations)
df.to_csv(f"data/hyperparameter_optimization_{config['pricing_model'].__name__}.csv", index=False)