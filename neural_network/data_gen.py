import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from pricer.analytical import BlackScholesCall

def validate_config_variables(variables, variables_needed, model_name):
    not_found = []
    for variable in variables:
        if variable not in variables_needed:
            not_found.append(variable)
    if not_found:
        raise KeyError(f"Variables {variable} need for {model_name} data generation is missing in config\n expected: {variables_needed}")

def bs_call_data_gen(config):
    validate_config_variables(config.get('variables'),
                              ['underlier_price', 'strike', 'expiry', 'interest_rate', 'volatility'],
                              BlackScholesCall.__name__)
    rng = np.random.default_rng(seed=config.get('seed', 42))
    
    n = config.get('n', 10**6)
    df = pd.DataFrame(index=range(n), columns=config.get('variables'))  
    
    for col in config.get('variables'):
        if col == 'strike':
            if col in config:
                df.loc[:, col] = rng.choice(config.get(col), size=n)
            else:
                l, u = config.get('strike_lower_bound_scalar'), config.get('strike_upper_bound_scalar')
                s = config.get("underlier_price")
                strike_vals = df["underlier_price"] + (config.get("strike_sampling_variance") ** 0.5) * rng.normal(size=n)
                strike_vals = np.clip(np.floor(strike_vals),
                                      np.min(s) * l,
                                      np.max(s) * u)
                df.loc[:, col] = strike_vals
        else:
            df.loc[:, col] = rng.choice(config.get(col), size=n)

    pricing_args = df[[*config.get('variables')]].to_dict(orient='list')
    pricing_args = {k: np.array(v) for k, v in pricing_args.items()}
    pricing_model = BlackScholesCall(**pricing_args)

    for col, func in config.get('derived_variables').items():
        df.loc[:, col] = func(pricing_model)

    if config.get('normalize'):
        df.loc[:, 'underlier_price'] = (df.loc[:, 'underlier_price'] / df.loc[:, 'strike']).astype(np.float64)    
        df.loc[:, 'price'] = (df.loc[:, 'price'] / df.loc[:, 'strike']).astype(np.float64)
    
    return df.astype(np.float64).round(4)


# config = {
#     'pricing_model': BlackScholesCall,
#     'data_gen_func': bs_call_data_gen,
#     'regenerate_data': False,
#     'data': {
#         'n': 10**6,
#         'seed': 8,
#         'underlier_price': np.linspace(50, 250, 1001),
#         'interest_rate': np.arange(0.01, 0.21, 0.001),
#         'volatility': np.arange(0.1, 0.41, 0.001),
#         'expiry': np.arange(14, 366) / 365,
#         'strike_lower_bound_scalar': 0.8,
#         'strike_upper_bound_scalar': 1.2,
#         'unique_strike_number': 200,
#         'variables': ['underlier_price', 'strike', 'expiry', 'interest_rate', 'volatility'],
#         'derived_variables': {'price': lambda x: x.price(),
#                               'delta': lambda x: x.delta(),
#                               'gamma': lambda x: x.gamma(),
#                               'vega': lambda x: x.vega(),
#                               'theta': lambda x: x.theta(),
#                               'rho': lambda x: x.rho()},
#         'normalize': False,
#     },
#     "model": {
#         "model_class": tf.keras.models.Model,
#         "neuron_per_layer": 64,
#         "layer_number": 5,
#         "learning_rate": 0.003,
#         "batch_size": 2**8,
#         "epochs": 10,
#         "l1": 0,
#         "l2": 2, # more than 2.5
#         "dropout": 0,
#         "input_variables": ['underlier_price', 'strike', 'expiry', 'interest_rate', 'volatility'],
#         "target_variables": {'price': 'relu',
#                              'delta': partial(tf.keras.activations.relu, max_value=1),
#                             #  'gamma': partial(tf.keras.activations.relu, max_value=1),
#                              'vega': 'relu',
#                             #  'theta': 'relu',
#                              'rho': 'relu'
#                              },
#     }
# }