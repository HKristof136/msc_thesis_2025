import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf
import pandas as pd


def weighted_mse(weights):
    def loss(y_true, y_pred):
        squared_diff = tf.square(y_true - y_pred)
        weighted_squared_diff = squared_diff * weights
        return tf.reduce_mean(weighted_squared_diff, axis=-1)
    return loss

def partial_relu(x):
    return tf.keras.activations.relu(x, max_value=1)

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.regenerate_data = config.get("regenerate_data", False)
        self.model = None
        self.pricing_model = config.get("pricing_model")
        self.data_gen_func = config.get("data_gen_func", None)
        self.input_variables = config.get("model").get("input_variables")
        self.target_variables = config.get("model").get("target_variables")

        root = os.getcwd()
        self.save_path = config.get(
            "save_path",
            os.path.join(root, "data", f"{self.pricing_model.__name__}.csv"),
        )
        if os.path.exists(self.save_path) and not self.regenerate_data:
            self.data = pd.read_csv(self.save_path)
            self.input_data = self.data[self.input_variables].astype(np.float64)
            self.target = self.data[list(self.target_variables)].astype(np.float64)
        else:
            self.input_data = None
            self.target = None

        self.test_save_path = config.get(
            "test_save_path",
            os.path.join(root, "data", f"{self.pricing_model.__name__}_test.csv"),
        )
        if os.path.exists(self.test_save_path) and not self.regenerate_data:
            self.test_data = pd.read_csv(self.test_save_path)
            self.test_input_data = self.test_data[self.input_variables].astype(
                np.float64
            )
            self.test_target = self.test_data[list(self.target_variables)].astype(
                np.float64
            )
        else:
            self.test_input_data = None
            self.test_target = None

    def build_model(self):
        config = self.config.get("model")
        model_cls = config.get("model_class")
        neuron_number = config["neuron_per_layer"]
        layer_number = config["layer_number"]

        input_layer = tf.keras.layers.Input(shape=(self.input_data.shape[1],))

        i = 1
        hidden_layer = tf.keras.layers.Dense(neuron_number, activation="relu")(
            input_layer
        )

        dropout_pct = config.get("dropout")
        if dropout_pct:
            hidden_layer = tf.keras.layers.Dropout(config.get("dropout", 0))(
                hidden_layer
            )

        while i != layer_number:
            hidden_layer = tf.keras.layers.Dense(
                neuron_number,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L1L2(
                    l1=config.get("l1", 0), l2=config.get("l2", 0)
                ),
            )(hidden_layer)
            if dropout_pct:
                hidden_layer = tf.keras.layers.Dropout(config.get("dropout", 0))(
                    hidden_layer
                )
            i += 1

        if self.target.shape[1] == 1:
            output_layer = tf.keras.layers.Dense(
                self.target.shape[1], activation="relu"
            )(hidden_layer)
        else:
            output_neurons = [
                tf.keras.layers.Dense(1, activation=activation)(hidden_layer)
                for _, activation in self.target_variables.items()
            ]
            output_layer = tf.keras.layers.concatenate(output_neurons)

        model = model_cls(inputs=input_layer, outputs=output_layer)

        optimizer = config.get(
            "optimizer", tf.keras.optimizers.Nadam
        )

        weights = tf.constant(config.get("loss_weights", [1.0] * self.target.shape[1]))
        loss = weighted_mse(weights)

        model.compile(
            optimizer=optimizer(learning_rate=config["learning_rate"]),
            loss=loss,
            metrics=[tf.keras.metrics.MeanAbsoluteError()],
        )

        return model

    def train(self):
        if (
            (self.input_data is None)
            or (self.test_input_data is None)
            or self.regenerate_data
        ):
            self.data = self.data_gen_func(self.config["data"])
            self.data.to_csv(self.save_path, index=False)

            self.input_data = self.data[self.input_variables].astype(np.float64)
            self.target = self.data[list(self.target_variables)].astype(np.float64)

            self.test_data = self.data_gen_func(self.config["test_data"])
            self.test_data.to_csv(self.test_save_path, index=False)

            self.test_input_data = self.test_data[self.input_variables].astype(
                np.float64
            )
            self.test_target = self.test_data[list(self.target_variables)].astype(
                np.float64
            )

        if not self.model:
            self.model = self.build_model()
            print(self.model.summary())
            self.model.fit(
                self.input_data.values,
                self.target.values,
                epochs=self.config["model"].get("epochs", 20),
                batch_size=self.config["model"].get("batch_size", 2**10),
            )

    def evaluate(self):
        self.data.loc[:, "in_sample"] = True
        self.test_data.loc[:, "in_sample"] = False
        concat_data = pd.concat([self.data, self.test_data], axis=0).reset_index(drop=True)

        calc_batch_size = self.config["model"].get("jacobian_batch_size", 10**5)
        for i in range(concat_data.shape[0] // calc_batch_size + 1):
            input_data = tf.Variable(
                concat_data[self.input_variables][
                    i * calc_batch_size : (i + 1) * calc_batch_size
                ],
                dtype=tf.float64,
            )
            if input_data.shape[0] == 0:
                continue
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(input_data)
                model_preds = self.model(input_data)

            calc_greeks = {}
            calc_greeks["first"] = tape.batch_jacobian(model_preds, input_data)
            calc_greeks["second"] = tape.batch_jacobian(calc_greeks["first"][:, 0, :], input_data)

            concat_data.loc[
                i * calc_batch_size : (i + 1) * calc_batch_size - 1,
                [f"model_{variable}" for variable in self.target_variables],
            ] = model_preds.numpy()

            for variable, greek_dict in self.config["model"].get("greeks", {}).items():
                for greek in greek_dict:
                    concat_data.loc[
                        i * calc_batch_size : (i + 1) * calc_batch_size - 1,
                        f"model_{greek['name']}_AD",
                    ] = calc_greeks[greek["order"]][:, 0, self.input_variables.index(variable)].numpy()
        del tape
        
        output_dict = {}
        in_sample_mask = concat_data["in_sample"]
        out_sample_mask = ~concat_data["in_sample"]
        for variable in ["price", "delta", "gamma", "vega", "theta", "rho"]:
            col_name =  f"model_{variable + '_AD' if variable != 'price' else variable}"
            output_dict[f"in_sample_{variable}_mse"] = tf.keras.losses.MSE(concat_data.loc[in_sample_mask, variable],
                                                  concat_data.loc[in_sample_mask, col_name]).numpy()
            output_dict[f"out_sample_{variable}_mse"] = tf.keras.losses.MSE(concat_data.loc[out_sample_mask, variable],
                                                    concat_data.loc[out_sample_mask, col_name]).numpy()
        return output_dict


if __name__ == "__main__":
    from functools import partial
    from neural_network.data_gen import bs_call_data_gen
    from pricer.analytical import BlackScholesCall

    config = {
        # "save_path": "test_run_df.csv",
        # "test_save_path": "test_run_df_test.csv",
        "pricing_model": BlackScholesCall,
        "data_gen_func": bs_call_data_gen,
        "regenerate_data": True,
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
            "batch_size": 2**12,
            "epochs": 25,
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

    import time

    s = time.perf_counter()
    pipeline = Pipeline(config)
    pipeline.train()
    df = pipeline.evaluate()
    print(f"Evaluation took: {time.perf_counter() - s} seconds")
