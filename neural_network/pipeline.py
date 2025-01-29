import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from model import PricerNet
from config_base import DataGenConfig, ModelConfig, PipeLineConfig

class Pipeline:
    def __init__(self, config: PipeLineConfig):
        self.config = config
        self.regenerate_data = config.regenerate_data
        self.model = None
        self.pricing_model = config.pricing_model
        self.data_gen_func = config.data_gen_func
        self.input_variables = config.model.input_variables
        self.target_variables = config.model.target_variables
        self.greek_variables = config.model.tensorflow_greeks

        root = os.getcwd()
        if config.save_path:
            self.save_path = config.save_path,
        else:
            self.save_path = os.path.join(root, "data", f"{self.pricing_model.__name__}.csv")

        if os.path.exists(self.save_path) and not self.regenerate_data:
            self.data = pd.read_csv(self.save_path)
            self.data = self.data.sample(frac=1).reset_index(drop=True)
            self.input_data = self.data[self.input_variables].astype(np.float64)
            self.target = self.data[list(self.target_variables)].astype(np.float64)
            self.greeks = self.data[list(self.greek_variables.keys())].astype(np.float64)
            if self.greeks.empty:
                data_tuple = (self.input_data.values, self.target.values)
            else:
                data_tuple = (self.input_data.values, self.target.values, self.greeks.values)
            self.dataset = tf.data.Dataset.from_tensor_slices(
                data_tuple
            ).batch(self.config.model.batch_size)
        else:
            self.input_data = None
            self.target = None
            self.greeks = None
            self.dataset = None

        if config.test_save_path:
            self.test_save_path = config.test_save_path
        else:
            self.test_save_path = os.path.join(root, "data", f"{self.pricing_model.__name__}_test.csv")

        if os.path.exists(self.test_save_path) and not self.regenerate_data:
            self.test_data = pd.read_csv(self.test_save_path)
            self.test_input_data = self.test_data[self.input_variables].astype(
                np.float64
            )
            self.test_target = self.test_data[list(self.target_variables)].astype(
                np.float64
            )
            self.test_greeks = self.test_data[list(self.greek_variables.keys())].astype(np.float64)
        else:
            self.test_input_data = None
            self.test_target = None
            self.test_greeks = None

    def build_model(self):
        config = self.config.model
        model_cls = config.model_class
        neuron_number = config.neuron_per_layer
        layer_number = config.layer_number
        hidden_layer_activation = config.hidden_layer_activation

        input_layer = tf.keras.layers.Input(shape=(self.input_data.shape[1],))

        i = 1
        hidden_layer = tf.keras.layers.Dense(neuron_number, activation=hidden_layer_activation)(
            input_layer
        )

        dropout_pct = config.dropout
        if dropout_pct:
            hidden_layer = tf.keras.layers.Dropout(config.dropout)(
                hidden_layer
            )

        while i != layer_number:
            hidden_layer = tf.keras.layers.Dense(
                neuron_number,
                activation=hidden_layer_activation,
                kernel_regularizer=tf.keras.regularizers.L1L2(
                    l1=config.l1, l2=config.l2
                ),
            )(hidden_layer)
            if dropout_pct:
                hidden_layer = tf.keras.layers.Dropout(config.dropout)(
                    hidden_layer
                )
            i += 1

        # TODO: unnecessary if statement
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

        if model_cls == PricerNet:
            model = model_cls(
                    config.tensorflow_greeks,
                    inputs=input_layer,
                    outputs=output_layer,
                )
        else:
            model = model_cls(
                inputs=input_layer, outputs=output_layer
            )
        optimizer = config.optimizer

        model.compile(
            optimizer=optimizer(learning_rate=config.learning_rate),
            loss="mse"
        )

        return model

    def train(self):
        if (
            (self.input_data is None)
            or (self.test_input_data is None)
            or self.regenerate_data
        ):
            self.data = self.data_gen_func(self.config.data)
            self.data = self.data.sample(frac=1).reset_index(drop=True)
            self.data.to_csv(self.save_path, index=False)

            self.input_data = self.data[self.input_variables].astype(np.float64)
            self.target = self.data[list(self.target_variables)].astype(np.float64)
            self.greeks = self.data[list(self.greek_variables.keys())].astype(np.float64)

            if self.greeks.empty:
                data_tuple = (self.input_data.values, self.target.values)
            else:
                data_tuple = (self.input_data.values, self.target.values, self.greeks.values)
            self.dataset = tf.data.Dataset.from_tensor_slices(
                data_tuple
            ).batch(self.config.model.batch_size)

            self.test_data = self.data_gen_func(self.config.test_data)
            self.test_data.to_csv(self.test_save_path, index=False)

            self.test_input_data = self.test_data[self.input_variables].astype(
                np.float64
            )
            self.test_target = self.test_data[list(self.target_variables)].astype(
                np.float64
            )
            self.test_greeks = self.test_data[list(self.greek_variables.keys())].astype(np.float64)

        if not self.model:
            self.model = self.build_model()
            print(self.model.summary())
            self.model.fit(
                self.dataset,
                epochs=self.config.model.epochs,
                batch_size=self.config.model.batch_size,
            )

    def evaluate(self):
        self.data.loc[:, "in_sample"] = True
        self.test_data.loc[:, "in_sample"] = False
        concat_data = pd.concat([self.data, self.test_data], axis=0).reset_index(drop=True)

        calc_batch_size = self.config.model.jacobian_batch_size
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

            for variable, greek_dict in self.config.model.greeks.items():
                for greek in greek_dict:
                    concat_data.loc[
                        i * calc_batch_size : (i + 1) * calc_batch_size - 1,
                        f"model_{greek['name']}_AD",
                    ] = calc_greeks[greek["order"]][:, 0, self.input_variables.index(variable)].numpy()
            del tape

        output_dict = {"neuron_per_layer": self.config.model.neuron_per_layer,
                       "layer_number": self.config.model.layer_number}
        in_sample_mask = concat_data["in_sample"]
        out_sample_mask = ~concat_data["in_sample"]
        for variable in ["price", "delta", "gamma", "vega", "theta", "rho"]:
            col_name =  f"model_{variable + '_AD' if variable != 'price' else variable}"
            output_dict[f"in_sample_{variable}_mse"] = tf.keras.losses.MSE(concat_data.loc[in_sample_mask, variable],
                                                  concat_data.loc[in_sample_mask, col_name]).numpy()
            output_dict[f"out_sample_{variable}_mse"] = tf.keras.losses.MSE(concat_data.loc[out_sample_mask, variable],
                                                    concat_data.loc[out_sample_mask, col_name]).numpy()

        for variable in ["price", "delta", "vega", "rho", "theta"]:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(concat_data.loc[out_sample_mask, variable],
                       concat_data.loc[out_sample_mask, f"model_{variable + '_AD' if variable != 'price' else variable}"],
                       s=0.1)
            ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [ax.get_xlim()[0], ax.get_xlim()[1]], color="red")
            ax.set_xlabel("true value")
            ax.set_ylabel("model value (from AD")
            plt.savefig(f"figures/{self.config.model.neuron_per_layer}_{self.config.model.layer_number}_{self.pricing_model.__name__}_{variable}.png")
            plt.close(fig)

        return output_dict


if __name__ == "__main__":
    import time
    from config import (
        blackscholes_call_config,
        blackscholes_put_config,
        american_black_scholes_put_config,
        up_and_out_barrier_call_config
    )

    df_list = []
    for neuron_number, layer_number in [
        # (32, 2),
        # (64, 4),
        (32, 2),
        ]:
        run_config = american_black_scholes_put_config
        run_config.model.neuron_per_layer = neuron_number
        run_config.model.layer_number = layer_number
        s = time.perf_counter()
        pipeline = Pipeline(run_config)
        pipeline.train()
        df_dict = pipeline.evaluate()
