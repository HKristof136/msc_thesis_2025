import pandas as pd
import numpy as np
import concurrent.futures

from data_generator.config import DataGeneratorConfig, VariableConfig
from get_logger import get_logger
from tqdm import tqdm

logger = get_logger()

class DataGenerator:
    def __init__(self, config: DataGeneratorConfig):
        self.config = config
        if isinstance(config.seed, int):
            self.seed = config.seed
        else:
            self.seed = hash(config.seed)
        self.rng = np.random.default_rng(self.seed + (hash(self.config.pricer.__name__) + hash(self.config.data_run_type)) % 314)

        self.variables = []
        self.variables += [v.name for v in config.parameter_variables]
        self.variables += [v.name for v in config.additional_variables]
        self.variables += [v.name for v in config.derived_variables]
        self.data = None

        self.feller_condition = config.feller_condition

    def generate_variable(self, variable_config: VariableConfig, size,
                          pricer: None, points=None, mean_override=None, verbose=True):
        if verbose:
            logger.info(f"Generating variable: {variable_config.name}")
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
            if mean_override is not None:
                return np.round(
                    self.rng.uniform(
                        low=mean_override * variable_config.lower_bound,
                        high=mean_override * variable_config.upper_bound,
                        size=size), 8
                ).astype(np.float32)
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
                    scale=variable_config.std * mean,
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
        logger.info(f"Generating {n * m} data points, from {n} parameter draws, pricing {m} instances per draw")
        df = pd.DataFrame(index=range(n * m), columns=self.variables, dtype=np.float32)

        param_variable_names = [v.name for v in self.config.parameter_variables]
        parameter_df = pd.DataFrame(
            index=range(n),
            columns=param_variable_names,
            dtype=np.float32
        )
        for variable in self.config.parameter_variables:
            if variable.name == "barrier":
                parameter_df.loc[:, variable.name] = self.generate_variable(variable, size=n, pricer=None,
                                                                            mean_override=parameter_df["strike"].values)
            else:
                parameter_df.loc[:, variable.name] = self.generate_variable(variable, size=n, pricer=None)

        if self.feller_condition:
            logger.info("Checking Feller condition on parameter_df")
            parameter_df["feller_condition"] = (2 * parameter_df["kappa"] * parameter_df["variance_theta"] - parameter_df["sigma"] ** 2) > 0
            parameter_df = parameter_df[parameter_df["feller_condition"]]

            while parameter_df.shape[0] != n:
                logger.info(f"Feller condition not satisfied for all drawn parameter sets, parameter_df shape: {parameter_df.shape}")
                temp_parameter_df = pd.DataFrame(
                    index=range(n),
                    columns=param_variable_names,
                    dtype=np.float32
                )
                for variable in self.config.parameter_variables:
                    if variable.name == "barrier":
                        temp_parameter_df.loc[:, variable.name] = self.generate_variable(variable, size=n, pricer=None,
                                                                                    mean_override=temp_parameter_df[
                                                                                        "strike"].values)
                    else:
                        temp_parameter_df.loc[:, variable.name] = self.generate_variable(variable, size=n, pricer=None)
                temp_parameter_df["feller_condition"] = (
                        2 * temp_parameter_df["kappa"] * temp_parameter_df["variance_theta"] - temp_parameter_df["sigma"] ** 2
                                                        ) > 0
                temp_parameter_df = temp_parameter_df[temp_parameter_df["feller_condition"]]
                temp_parameter_df = temp_parameter_df.sample(
                    min(temp_parameter_df.shape[0], n - parameter_df.shape[0]),
                    random_state=self.rng
                )
                parameter_df = pd.concat([parameter_df, temp_parameter_df], axis=0)
            parameter_df = parameter_df.drop(columns=["feller_condition"])

        logger.info(f"Successfully created parameter_df, with shape: {parameter_df.shape}, variables: {list(parameter_df.columns)}")

        df.loc[:, [v.name for v in self.config.parameter_variables]] = pd.concat(
            [parameter_df] * m
        ).sort_index().reset_index(drop=True).values

        if self.config.pricer.type == "analytical":
            logger.info("Adding additional_variables")
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
            logger.info("Adding derived_variables")
            for variable in self.config.derived_variables:
                df.loc[:, variable.name] = self.generate_variable(variable, size=n * m, pricer=pricer)
        elif self.config.pricer.type == "pde_grid":
            logger.info("Starting multiprocessing data generation")
            with concurrent.futures.ProcessPoolExecutor() as executor:
                derived_df_list = list(
                    tqdm(executor.map(self.process_row_cn,
                                      [row for _, row in parameter_df.iterrows()],
                                      [m] * n,
                                      [*parameter_df.index],
                                      ), total=n))
            df.loc[:, derived_df_list[0].columns] = pd.concat(
                derived_df_list
            ).sort_values("row_ind").values.astype(np.float32)
        elif self.config.pricer.type == "adi":
            logger.info("Starting multiprocessing data generation")
            with concurrent.futures.ProcessPoolExecutor() as executor:
                derived_df_list = list(
                    tqdm(executor.map(self.process_row_heston,
                                      [row for _, row in parameter_df.iterrows()],
                                      [m] * n,
                                      [*parameter_df.index],
                                      ), total=n))
            df.loc[:, derived_df_list[0].columns] = pd.concat(
                derived_df_list
            ).sort_values("row_ind").values.astype(np.float32)
        else:
            raise ValueError(f"Unsupported pricer type {self.config.pricer.type}")

        if self.config.implied_volatility:
            logger.info("Constraint on vega in the dataset for ImpliedVol")
            df = df.loc[df["vega"] >= 0.1]
            df["vega"] = 1 / df["vega"]
        self.data = df

    def process_row_cn(self, row, m, row_ind):
        temp_derived_df = pd.DataFrame(index=range(m),
                                       columns=["underlier_price", "expiry"] + [v.name for v in
                                                                                self.config.derived_variables],
                                       dtype=np.float32)
        if "Up" in self.config.pricer.__name__:
            underlier_grid = np.linspace(0.01, row["barrier"] + 0.01, 201)
        else:
            ds_grid = np.linspace(0.01, row["barrier"] + 0.01, 201)
            ds = ds_grid[1] - ds_grid[0]
            underlier_grid = np.arange(0.01, row["barrier"] * 2, ds)
        time_grid = np.linspace(0, row["pricer_expiry"] + 0.01, 101)
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

        mask = (0.01 <= pricer.t) & (pricer.t <= row["pricer_expiry"] / 2)
        if "Up" in self.config.pricer.__name__:
            mask = mask & (row["barrier"] * 0.7 <= pricer.x) & (pricer.x <= (row["barrier"] - 0.01))
        else:
            mask = mask & ((row["barrier"] + 0.01) <= pricer.x) & (pricer.x <= row["barrier"] * 1.3)
        points_in_scope = np.maximum(0.0, pricer.grid[mask]).flatten()

        points_ind = self.rng.choice(np.arange(points_in_scope.shape[0]), size=m,)
        points = np.zeros(shape=(m, 2), dtype=np.float32)
        points[:, 0] = pricer.x[mask].flatten()[points_ind]
        points[:, 1] = pricer.max_t - pricer.t[mask].flatten()[points_ind]

        temp_derived_df.loc[:, "underlier_price"] = points[:, 0]
        temp_derived_df.loc[:, "expiry"] = points[:, 1]

        for variable in self.config.derived_variables:
            temp_derived_df.loc[:, variable.name] = self.generate_variable(
                variable, size=m,
                pricer=pricer, points=points, verbose=False
            )
        temp_derived_df.loc[:, "row_ind"] = row_ind
        return temp_derived_df

    def process_row_heston(self, row, m, row_ind):
        temp_derived_df = pd.DataFrame(
            index=range(m),
            columns=["underlier_price", "expiry", "initial_variance"] + [v.name for v in self.config.derived_variables],
            dtype=np.float32
        )
        time_offset = 0.01
        time_grid = np.linspace(0, row["pricer_expiry"] + time_offset, 51)
        pricer_config_args = {"underlier_price_grid": np.array([]), "time_grid": time_grid, "verbose": False}
        pricer_config_args.update(
            {v: row[v] for v in self.config.pricer.input_names if v not in pricer_config_args}
        )
        pricer_config = self.config.pricer_config(
            **pricer_config_args
        )
        pricer = self.config.pricer(pricer_config)
        pricer.solve()

        if "barrier" not in self.config.pricer.input_names:
            expiry = time_offset # self.rng.uniform(time_offset, row["pricer_expiry"] / 2, m).astype(np.float32)
            discounted_strike = pricer.strike * np.exp(-row["interest_rate"] * expiry)
            underlier_price = np.clip(
                self.rng.normal(discounted_strike, 0.1 * discounted_strike, m),
                0.1 * discounted_strike,
                1.9 * discounted_strike
            ).astype(np.float32)
            variance = self.rng.uniform(row["variance_theta"] * 0.75, row["variance_theta"] * 1.25, m).astype(np.float32)

            points = np.zeros(shape=(m, 3))
            points[:, 0] = variance
            points[:, 1] = pricer.max_t - expiry
            points[:, 2] = underlier_price
        else:
            mask = (0.01 <= pricer.tt) & (pricer.tt <= row["pricer_expiry"] / 2)
            mask = mask & (row["barrier"] * 0.75 <= pricer.xx) & (pricer.xx <= (row["barrier"] - 0.01))
            mask = mask & (pricer.vv >= pricer.th / 2) & (pricer.vv <= pricer.th * 2)
            points_in_scope = np.maximum(0.0, pricer.grid[mask]).flatten()
            points_ind = self.rng.choice(np.arange(points_in_scope.shape[0]), size=m,)

            points = np.zeros(shape=(m, 3))
            points[:, 0] = pricer.vv[mask].flatten()[points_ind]
            points[:, 1] = pricer.max_t - pricer.tt[mask].flatten()[points_ind]
            points[:, 2] = pricer.xx[mask].flatten()[points_ind]

        temp_derived_df.loc[:, "initial_variance"] = points[:, 0]
        temp_derived_df.loc[:, "underlier_price"] = points[:, 2]
        temp_derived_df.loc[:, "expiry"] = points[:, 1]

        for variable in self.config.derived_variables:
            temp_derived_df.loc[:, variable.name] = self.generate_variable(
                variable, size=m,
                pricer=pricer, points=points, verbose=False
            )
        temp_derived_df.loc[:, "row_ind"] = row_ind
        return temp_derived_df

    def get_data(self):
        logger.info(f"DataGenerator for model: {self.config.pricer.__name__}, run type: {self.config.data_run_type} initialized")
        logger.info(f"Using seed: {self.seed + (hash(self.config.pricer.__name__) + hash(self.config.data_run_type)) % 314}")
        if self.data is None:
            self.generate_data()
        logger.info(f"Dataset generated, with shape: {self.data.shape}, with columns: {list(self.data.columns)}")
        return self.data
