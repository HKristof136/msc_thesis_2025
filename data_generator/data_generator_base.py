import pandas as pd
import numpy as np
import concurrent.futures

from data_generator.config import DataGeneratorConfig, VariableConfig
from tqdm import tqdm

class DataGenerator:
    def __init__(self, config: DataGeneratorConfig):
        self.config = config
        self.seed = config.seed
        self.rng = np.random.default_rng(self.seed)

        self.variables = []
        self.variables += [v.name for v in config.parameter_variables]
        self.variables += [v.name for v in config.additional_variables]
        self.variables += [v.name for v in config.derived_variables]
        self.data = None

    def generate_variable(self, variable_config: VariableConfig, size,
                          pricer: None, points=None, mean_override=None):
        if variable_config.generator_function is not None:
            if points is None:
                return np.round(
                    variable_config.generator_function(pricer), 8
                ).astype(np.float32)
            else:
                return np.round(
                    variable_config.generator_function(pricer, points), 8
                ).astype(np.float32)
        if variable_config.distribution == "uniform":
            return np.round(
                self.rng.uniform(
                    low=variable_config.lower_bound,
                    high=variable_config.upper_bound,
                    size=size),8
            ).astype(np.float32)
        elif variable_config.distribution == "linspace":
            return np.round(
                np.linspace(
                    start=variable_config.lower_bound * mean_override,
                    stop=variable_config.upper_bound * mean_override,
                    num=size),8
            ).astype(np.float32).T.tolist()
        elif variable_config.distribution == "normal":
            mean = mean_override if mean_override is not None else variable_config.mean
            values = np.round(
                self.rng.normal(
                    loc=mean,
                    scale=variable_config.std,
                    size=size),8
            ).astype(np.float32)
            if variable_config.lower_clip is not None:
                values = np.maximum(values, variable_config.lower_clip * mean)
            if variable_config.upper_clip is not None:
                values = np.minimum(values, variable_config.upper_clip * mean)
            return values
        else:
            raise ValueError(f"Unsupported distribution {variable_config.distribution}")

    def generate_data(self):
        n, m = self.config.n, self.config.m
        df = pd.DataFrame(index=range(n * m), columns=self.variables, dtype=np.float32)

        param_variable_names = [v.name for v in self.config.parameter_variables]
        parameter_df = pd.DataFrame(
            index=range(n),
            columns=param_variable_names,
            dtype=np.float32
        )
        for variable in self.config.parameter_variables:
            parameter_df.loc[:, variable.name] = self.generate_variable(variable, size=n, pricer=None)

        df.loc[:, [v.name for v in self.config.parameter_variables]] = pd.concat(
            [parameter_df] * m
        ).sort_index().reset_index(drop=True).values

        if self.config.pricer.type == "analytical":
            for variable in self.config.additional_variables:
                if variable.mean_override == "discounted_strike":
                    discount_factors = np.exp(
                        - df["interest_rate"].values.astype(np.float32) \
                        * df["expiry"].values.astype(np.float32))
                    discounted_strike = df["strike"].values.astype(np.float32) * discount_factors
                    df.loc[:, variable.name] = self.generate_variable(variable, size=n * m,
                                                                      pricer=None, mean_override=discounted_strike)
                else:
                    df.loc[:, variable.name] = self.generate_variable(variable, size=n * m, pricer=None)
            pricer_config = self.config.pricer_config(
                **{v: df[v].values.astype(np.float32) for v in self.config.pricer.input_names}
            )
            pricer = self.config.pricer(pricer_config)
            for variable in self.config.derived_variables:
                df.loc[:, variable.name] = self.generate_variable(variable, size=n * m, pricer=pricer)
        elif self.config.pricer.type == "pde_grid":
            with concurrent.futures.ProcessPoolExecutor() as executor:
                derived_df_list = list(
                    tqdm(executor.map(self.process_row,
                                      [row for _, row in parameter_df.iterrows()],
                                      [m] * n,
                                      [*parameter_df.index],
                                      ), total=n))
            df.loc[:, derived_df_list[0].columns] = pd.concat(
                derived_df_list
            ).sort_values("row_ind").values.astype(np.float32)
        else:
            raise ValueError(f"Unsupported pricer type {self.config.pricer.type}")
        self.data = df

    def process_row(self, row, m, row_ind):
        temp_derived_df = pd.DataFrame(index=range(m),
                                       columns=["underlier_price", "expiry"] + [v.name for v in
                                                                                self.config.derived_variables],
                                       dtype=np.float32)
        if "UpAndOut" in self.config.pricer.__name__:
            underlier_grid = np.linspace(0, row["barrier"] * 1.1, 201)
        time_grid = np.linspace(0, row["pricer_expiry"] * 1.1, 101)
        pricer_config_args = {"underlier_price_grid": underlier_grid, "time_grid": time_grid,
                              "verbose": False}
        pricer_config_args.update(
            {v: row[v] for v in self.config.pricer.input_names if v not in pricer_config_args}
        )
        pricer_config = self.config.pricer_config(
            **pricer_config_args
        )
        pricer = self.config.pricer(pricer_config)
        pricer.solve()

        discounted_strike = pricer.strike * np.exp(-pricer.r * (pricer.max_t - pricer.t))
        mask = (pricer.x < pricer.barrier) & (pricer.x > discounted_strike * 0.6)
        mask = mask & ((pricer.max_t - pricer.t) < row["pricer_expiry"]) & (
                    (pricer.max_t - pricer.t) >= row["pricer_expiry"] / 2)
        points_x = pricer.x[mask]
        points_t = pricer.t[mask]
        points_gamma = np.abs(pricer.gamma(np.vstack([points_x, pricer.max_t - points_t]).T))
        prob_dist = points_gamma / np.sum(points_gamma)

        points_ind = np.random.choice(np.arange(points_x.shape[0]), size=m, p=prob_dist)
        points = np.zeros(shape=(m, 2))
        points[:, 0] = points_x[points_ind]
        points[:, 1] = pricer.max_t - points_t[points_ind]

        temp_derived_df.loc[:, "underlier_price"] = points[:, 0]
        temp_derived_df.loc[:, "expiry"] = points[:, 1]

        for variable in self.config.derived_variables:
            temp_derived_df.loc[:, variable.name] = self.generate_variable(
                variable, size=m,
                pricer=pricer, points=points
            )
        temp_derived_df.loc[:, "row_ind"] = row_ind
        return temp_derived_df

    def get_data(self):
        if self.data is None:
            self.generate_data()
        return self.data

if __name__ == "__main__":
    # from pricer.pde_solver import BarrierUpAndOutCallPDE
    # from pricer.config_base import PDESolverConfig
    #
    # config = DataGeneratorConfig(
    #     pricer=BarrierUpAndOutCallPDE,
    #     pricer_config=PDESolverConfig,
    #     n=10**2,
    #     m=3 * 10**2,
    #     parameter_variables=[
    #         VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
    #         VariableConfig("barrier", distribution="uniform", lower_bound=1.1, upper_bound=1.8),
    #         VariableConfig("volatility", distribution="uniform", lower_bound=0.05, upper_bound=0.36),
    #         VariableConfig("interest_rate", distribution="uniform", lower_bound=0.01, upper_bound=0.21),
    #         VariableConfig("pricer_expiry", distribution="uniform", lower_bound=1/365, upper_bound=1.0),
    #     ],
    #     additional_variables=[],
    #     derived_variables=[
    #         VariableConfig("price", generator_function=lambda x, points: x.price(points)),
    #         VariableConfig("delta", generator_function=lambda x, points: x.delta(points)),
    #         VariableConfig("gamma", generator_function=lambda x, points: x.gamma(points)),
    #         VariableConfig("vega", generator_function=lambda x, points: x.vega(points)),
    #         VariableConfig("theta", generator_function=lambda x, points: x.theta(points)),
    #         VariableConfig("rho", generator_function=lambda x, points: x.rho(points)),
    #     ],
    #     black_scholes_normalize=False,
    # )
    #
    # df = DataGenerator(config).get_data()
    # print(df.head())

    from pricer.analytical import BlackScholesCall
    from pricer.config_base import BlackScholesConfig
    from neural_network.utils import (
        price_function,
        delta_function,
        gamma_function,
        vega_function,
        theta_function,
        rho_function,
    )

    config = DataGeneratorConfig(
        pricer=BlackScholesCall,
        pricer_config=BlackScholesConfig,
        n=10 ** 6,
        m=5,
        parameter_variables=[
            VariableConfig("strike", distribution="uniform", lower_bound=1.0, upper_bound=1.0),
            VariableConfig("volatility", distribution="uniform", lower_bound=0.05, upper_bound=0.36),
            VariableConfig("interest_rate", distribution="uniform", lower_bound=0.01, upper_bound=0.21),
            VariableConfig("expiry", distribution="uniform", lower_bound=1 / 365, upper_bound=1.0),
        ],
        additional_variables=[
            VariableConfig("underlier_price", distribution="normal", lower_clip=0.6, upper_clip=1.4,
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

    df = DataGenerator(config).get_data()
    print(df.head())
