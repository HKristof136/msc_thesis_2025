import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from tqdm import tqdm
from pricer.analytical import BlackScholesCall, BlackScholesPut
from pricer.pde_solver import (
    AmericanBlackScholesPutPDE,
    BarrierUpAndOutCallPDE,
    BarrierDownAndOutPutPDE
)
from pricer.config_base import BlackScholesConfig, PDESolverConfig
from neural_network.config_base import DataGenConfig


def bs_call_data_gen(config: DataGenConfig):
    rng = np.random.default_rng(seed=config.seed)

    n = config.n
    df = pd.DataFrame(index=range(n), columns=config.variables)

    for col in config.variables:
        if col == "strike":
            if config.__dict__.get(col, False):
                df.loc[:, col] = rng.choice(config.__dict__[col], size=n)
            else:
                l, u = config.strike_lower_bound_pct, config.strike_upper_bound_pct
                s = config.underlier_price
                strike_vals = df["underlier_price"] * (
                        1 + config.strike_sampling_std_pct * rng.normal(size=n)
                )
                strike_vals = np.clip(
                    np.floor(strike_vals), df["underlier_price"] * l, df["underlier_price"] * u
                )
                df.loc[:, col] = strike_vals
        else:
            df.loc[:, col] = rng.choice(config.__dict__[col], size=n)

    pricing_args = df[[*config.variables]].to_dict(orient="list")
    pricing_args = {k: np.array(v) for k, v in pricing_args.items()}
    pricing_model_config = BlackScholesConfig(**pricing_args)
    pricing_model = BlackScholesCall(pricing_model_config)

    for col, func in config.derived_variables.items():
        df.loc[:, col] = func(pricing_model)

    if config.normalize:
        df.loc[:, "underlier_price"] = (
                df.loc[:, "underlier_price"] / df.loc[:, "strike"]
        ).astype(np.float32)
        df.loc[:, "price"] = (df.loc[:, "price"] / df.loc[:, "strike"]).astype(
            np.float32
        )
        df.loc[:, "gamma"] = (df.loc[:, "gamma"] * df.loc[:, "strike"]).astype(
            np.float32
        )
        df.loc[:, "theta"] = (df.loc[:, "theta"] / df.loc[:, "strike"]).astype(
            np.float32
        )
        df.loc[:, "vega"] = (df.loc[:, "vega"] / df.loc[:, "strike"]).astype(np.float32)
        df.loc[:, "rho"] = (df.loc[:, "rho"] / df.loc[:, "strike"]).astype(np.float32)
    return df.astype(np.float32).round(4)

def bs_put_data_gen(config: DataGenConfig):
    rng = np.random.default_rng(seed=config.seed)

    n = config.n
    df = pd.DataFrame(index=range(n), columns=config.variables)

    for col in config.variables:
        if col == "strike":
            if config.__dict__.get(col, False):
                df.loc[:, col] = rng.choice(config.__dict__[col], size=n)
            else:
                l, u = config.strike_lower_bound_pct, config.strike_upper_bound_pct
                s = config.underlier_price
                strike_vals = df["underlier_price"] * (
                        1 + config.strike_sampling_std_pct * rng.normal(size=n)
                )
                strike_vals = np.clip(
                    np.floor(strike_vals), df["underlier_price"] * l, df["underlier_price"] * u
                )
                df.loc[:, col] = strike_vals
        else:
            df.loc[:, col] = rng.choice(config.__dict__[col], size=n)

    pricing_args = df[[*config.variables]].to_dict(orient="list")
    pricing_args = {k: np.array(v) for k, v in pricing_args.items()}
    pricing_model_config = BlackScholesConfig(**pricing_args)
    pricing_model = BlackScholesPut(pricing_model_config)

    for col, func in config.derived_variables.items():
        df.loc[:, col] = func(pricing_model)

    if config.normalize:
        df.loc[:, "underlier_price"] = (
                df.loc[:, "underlier_price"] / df.loc[:, "strike"]
        ).astype(np.float32)
        df.loc[:, "price"] = (df.loc[:, "price"] / df.loc[:, "strike"]).astype(
            np.float32
        )
        df.loc[:, "gamma"] = (df.loc[:, "gamma"] * df.loc[:, "strike"]).astype(
            np.float32
        )
        df.loc[:, "theta"] = (df.loc[:, "theta"] / df.loc[:, "strike"]).astype(
            np.float32
        )
        df.loc[:, "vega"] = (df.loc[:, "vega"] / df.loc[:, "strike"]).astype(np.float32)
        df.loc[:, "rho"] = (df.loc[:, "rho"] / df.loc[:, "strike"]).astype(np.float32)
    return df.astype(np.float32).round(4)

def american_bs_put_data_gen(config: DataGenConfig):
    rng = np.random.default_rng(seed=config.seed)

    n = config.n
    m = config.m
    k = n // m

    param_df = pd.DataFrame(index=range(m), columns=["strike", "interest_rate", "volatility"])
    for col in ["strike", "interest_rate", "volatility"]:
        param_df.loc[:, col] = rng.choice(config.__dict__[col], size=m)

    df_list = []

    for _, row in tqdm(param_df.iterrows(), total=m):
        df = pd.DataFrame(index=range(k), columns=config.variables)
        df.loc[:, "strike"] = row["strike"]
        df.loc[:, "interest_rate"] = row["interest_rate"]
        df.loc[:, "volatility"] = row["volatility"]

        points = np.zeros((k, 2))
        l, u = config.strike_lower_bound_pct, config.strike_upper_bound_pct

        points[:, 0] = df["strike"] * (1 + config.price_points_sampling_std_pct * rng.normal(size=k))
        points[:, 0] = np.clip(points[:, 0], df["strike"] * l, df["strike"] * u)
        points[:, 1] = rng.choice(config.expiry, size=k)
        
        df.loc[:, ["underlier_price", "expiry"]] = points
        df = df[config.variables]
        
        pricing_model_config = PDESolverConfig(
            underlier_price_grid=np.linspace(0.5, 2 * max(points[:, 0]), config.x_step),
            time_grid=np.linspace(0, 1.05 * max(points[:, 1]), config.t_step),
            interest_rate=row["interest_rate"],
            strike=row["strike"],
            volatility=row["volatility"],
            verbose=False,
        )
        pricing_model = AmericanBlackScholesPutPDE(pricing_model_config)
        
        for col, func in config.derived_variables.items():
            df.loc[:, col] = func(pricing_model, points)
        df_list.append(df)

    df = pd.concat(df_list)
    df["price"] = np.maximum(0.0, df["price"].values)
    df["delta"] = np.clip(df["delta"].values, -1.0, 0.0)

    if config.normalize:
        df.loc[:, "underlier_price"] = (
                df.loc[:, "underlier_price"] / df.loc[:, "strike"]
        ).astype(np.float32)
        df.loc[:, "price"] = (df.loc[:, "price"] / df.loc[:, "strike"]).astype(
            np.float32
        )
        df.loc[:, "gamma"] = (df.loc[:, "gamma"] * df.loc[:, "strike"]).astype(
            np.float32
        )
        df.loc[:, "theta"] = (df.loc[:, "theta"] / df.loc[:, "strike"]).astype(
            np.float32
        )
        df.loc[:, "vega"] = (df.loc[:, "vega"] / df.loc[:, "strike"]).astype(np.float32)
        df.loc[:, "rho"] = (df.loc[:, "rho"] / df.loc[:, "strike"]).astype(np.float32)
    return df.astype(np.float32).round(4)

def up_and_out_barrier_call_data_gen(config: DataGenConfig):
    rng = np.random.default_rng(seed=config.seed)

    n = config.n
    m = config.m
    k = n // m

    param_df = pd.DataFrame(index=range(m), columns=["strike", "interest_rate", "volatility"])
    for col in ["strike", "interest_rate", "volatility"]:
        param_df.loc[:, col] = rng.choice(config.__dict__[col], size=m)

    param_df.loc[:, "barrier"] = np.clip(
        (config.barrier_distance_mean_pct * param_df["strike"]) * (1 + config.barrier_sampling_std_pct * rng.normal(size=m)),
        config.barrier_distance_min_pct * param_df["strike"],
        np.inf
    )
    df_list = []

    for _, row in tqdm(param_df.iterrows(), total=m):
        df = pd.DataFrame(index=range(k), columns=config.variables)
        df.loc[:, "strike"] = row["strike"]
        df.loc[:, "interest_rate"] = row["interest_rate"]
        df.loc[:, "volatility"] = row["volatility"]
        df.loc[:, "barrier"] = row["barrier"]

        pricing_model_config = PDESolverConfig(
            underlier_price_grid=np.linspace(0, 1.05 * row["barrier"], config.x_step),
            time_grid=np.linspace(0, 1.05 * np.max(config.expiry), config.t_step),
            interest_rate=row["interest_rate"],
            strike=row["strike"],
            volatility=row["volatility"],
            barrier=row["barrier"],
            verbose=False,
        )
        pricing_model = BarrierUpAndOutCallPDE(pricing_model_config)
        pricing_model.solve()

        points = np.zeros((k, 3))
        # points[:, 0] = ((df["strike"] + df["barrier"]) / 2) * (
        #             1 + config.price_points_sampling_std_pct * rng.normal(size=k))
        # points[:, 0] = np.clip(
        #     points[:, 0],
        #     config.underlier_price[0],
        #     np.minimum(config.underlier_price[-1], 0.9 * df["barrier"])
        # )
        # points[:, 1] = rng.choice(config.expiry, size=k)

        dist = np.maximum(pricing_model.grid[:, config.t_step//10:-(config.t_step//10)], 0.0) / np.maximum(pricing_model.grid[:, config.t_step//10:-(config.t_step//10)], 0.0).sum()
        pairs = np.indices(dimensions=(config.t_step - 2 * (config.t_step//10), config.x_step)).T
        selections = rng.choice(np.arange(config.x_step * (config.t_step - 2 * (config.t_step//10))), p=dist.reshape(-1), size=k, replace=False)
        selections = pairs.reshape(-1, 2)[selections]

        points[:, 0] = pricing_model.x[selections[:, 1], config.t_step//10 + selections[:, 0]]
        points[:, 1] = pricing_model.max_t - pricing_model.t[selections[:, 1], config.t_step//10 + selections[:, 0]]
        points[:, 2] = pricing_model.grid[selections[:, 1], config.t_step//10 + selections[:, 0]]

        df.loc[:, ["underlier_price", "expiry", "price"]] = points
        df = df[config.variables]

        for col, func in config.derived_variables.items():
            df.loc[:, col] = func(pricing_model, points)
        df_list.append(df)

    df = pd.concat(df_list)

    if config.normalize:
        df.loc[:, "underlier_price"] = (
                df.loc[:, "underlier_price"] / df.loc[:, "strike"]
        ).astype(np.float32)
        df.loc[:, "price"] = (df.loc[:, "price"] / df.loc[:, "strike"]).astype(
            np.float32
        )
        df.loc[:, "barrier"] = (df.loc[:, "barrier"] / df.loc[:, "strike"]).astype(
            np.float32
        )
        df.loc[:, "gamma"] = (df.loc[:, "gamma"] * df.loc[:, "strike"]).astype(
            np.float32
        )
        df.loc[:, "theta"] = (df.loc[:, "theta"] / df.loc[:, "strike"]).astype(
            np.float32
        )
        df.loc[:, "vega"] = (df.loc[:, "vega"] / df.loc[:, "strike"]).astype(np.float32)
        df.loc[:, "rho"] = (df.loc[:, "rho"] / df.loc[:, "strike"]).astype(np.float32)
    return df.astype(np.float32).round(4)

def down_and_out_barrier_put_data_gen(config: DataGenConfig):
    rng = np.random.default_rng(seed=config.seed)

    n = config.n
    m = config.m
    k = n // m

    param_df = pd.DataFrame(index=range(m), columns=["strike", "interest_rate", "volatility"])
    for col in ["strike", "interest_rate", "volatility"]:
        param_df.loc[:, col] = rng.choice(config.__dict__[col], size=m)

    param_df.loc[:, "barrier"] = np.clip(
        (config.barrier_distance_mean_pct * param_df["strike"]) * (1 + config.barrier_sampling_std_pct * rng.normal(size=m)),
        config.barrier_distance_min_pct * param_df["strike"],
        np.inf
    )
    df_list = []

    for _, row in tqdm(param_df.iterrows(), total=m):
        df = pd.DataFrame(index=range(k), columns=config.variables)
        df.loc[:, "strike"] = row["strike"]
        df.loc[:, "interest_rate"] = row["interest_rate"]
        df.loc[:, "volatility"] = row["volatility"]
        df.loc[:, "barrier"] = row["barrier"]

        points = np.zeros((k, 2))
        # TODO: beta distribution for price points
        points[:, 0] = df["barrier"] * (1 - (config.price_points_sampling_std_pct * rng.normal(size=k) ** 2))
        points[:, 0] = np.clip(
            points[:, 0],
            config.underlier_price[0],
            np.minimum(config.underlier_price[-1], df["barrier"])
        )
        points[:, 1] = rng.choice(config.expiry, size=k)

        df.loc[:, ["underlier_price", "expiry"]] = points
        df = df[config.variables]

        pricing_model_config = PDESolverConfig(
            underlier_price_grid=np.linspace(0, 2 * max(points[:, 0]), 500),
            time_grid=np.linspace(0, 1.05 * max(points[:, 1]), 400),
            interest_rate=row["interest_rate"],
            strike=row["strike"],
            volatility=row["volatility"],
            barrier=row["barrier"],
            verbose=False,
        )
        pricing_model = BarrierDownAndOutPutPDE(config=pricing_model_config)

        for col, func in config.derived_variables.items():
            df.loc[:, col] = func(pricing_model, points)
        df_list.append(df)

    df = pd.concat(df_list)

    if config.normalize:
        df.loc[:, "underlier_price"] = (
                df.loc[:, "underlier_price"] / df.loc[:, "strike"]
        ).astype(np.float32)
        df.loc[:, "price"] = (df.loc[:, "price"] / df.loc[:, "strike"]).astype(
            np.float32
        )
        df.loc[:, "barrier"] = (df.loc[:, "barrier"] / df.loc[:, "strike"]).astype(
            np.float32
        )
        df.loc[:, "gamma"] = (df.loc[:, "gamma"] * df.loc[:, "strike"]).astype(
            np.float32
        )
        df.loc[:, "theta"] = (df.loc[:, "theta"] / df.loc[:, "strike"]).astype(
            np.float32
        )
        df.loc[:, "vega"] = (df.loc[:, "vega"] / df.loc[:, "strike"]).astype(np.float32)
        df.loc[:, "rho"] = (df.loc[:, "rho"] / df.loc[:, "strike"]).astype(np.float32)
    return df.astype(np.float32).round(4)

if __name__ == "__main__":
    data_config = DataGenConfig(
        underlier_price_range=np.array([0.01, 300]),
        expiry_time_range=np.array([14/365, 1.0]),
        interest_rate=np.arange(0.01, 0.21, 0.01),
        volatility=np.arange(0.01, 0.41, 0.01),
        strike=np.linspace(50, 150, 1000),
        variables=["underlier_price", "strike", "interest_rate", "volatility", "expiry", "price"],
        parameter_variables=["strike", "volatility", "interest_rate"],
        derived_variables={
            # "price": lambda x: x.price(),
            "delta": lambda x: x.delta(),
            "gamma": lambda x: x.gamma(),
            "vega": lambda x: x.vega(),
            "theta": lambda x: x.theta(),
            "rho": lambda x: x.rho(),
        },
        normalize=True,
        n=10 ** 6,
        m=10 ** 4,
        x_step=100,
        t_step=100,
        seed=42,
    )

    bs_call_data_gen(data_config)