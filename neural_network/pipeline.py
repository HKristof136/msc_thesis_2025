import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datetime
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from model import PricerNet
from config_base import PipeLineConfig

class Pipeline:
    train_data = None
    train_input_data = None
    train_target = None
    train_greeks = None
    train_dataset = None

    test_data = None
    test_input_data = None
    test_target = None
    test_greeks = None
    test_dataset = None

    validation_data = None
    validation_input_data = None
    validation_target = None
    validation_greeks = None
    validation_dataset = None

    def __init__(self, config: PipeLineConfig):
        self.config = config
        self.regenerate_data = config.regenerate_data
        self.model = None
        self.pricing_model = config.pricing_model
        self.input_variables = config.model.input_variables
        self.target_variables = config.model.target_variables
        self.greek_variables = config.model.tensorflow_greeks

        root = os.getcwd()
        for df_type in ["train", "validation", "test"]:
            if config.__dict__.get(f"{df_type}_save_path"):
                setattr(self, f"{df_type}_save_path", config.__dict__[f"{df_type}_save_path"])
            else:
                seed = config.train_data.seed
                os.makedirs(os.path.join(root, "data", str(seed)), exist_ok=True)
                path = os.path.join(root,
                                    "data",
                                    str(seed),
                                    f"{self.pricing_model.__name__}_{df_type}.parquet")
                setattr(self, f"{df_type}_save_path", path)
            save_path = getattr(self, f"{df_type}_save_path")

            if os.path.exists(save_path) and not self.regenerate_data:
                data = pd.read_parquet(save_path)
                data = data.sample(frac=1, random_state=config.train_data.seed).reset_index(drop=True)

                setattr(self, f"{df_type}_data", data)
                setattr(self, f"{df_type}_input_data", data[self.input_variables].astype(np.float64))
                setattr(self, f"{df_type}_target", data[list(self.target_variables)].astype(np.float64))
                setattr(self, f"{df_type}_greeks", data[list(self.greek_variables.keys())].astype(np.float64))
                if getattr(self, f"{df_type}_greeks").empty:
                    data_tuple = (
                        getattr(self, f"{df_type}_input_data").values,
                        getattr(self, f"{df_type}_target").values
                    )
                else:
                    data_tuple = (
                        getattr(self, f"{df_type}_input_data").values,
                        getattr(self, f"{df_type}_target").values,
                        getattr(self, f"{df_type}_greeks").values
                    )

                setattr(self, f"{df_type}_dataset", tf.data.Dataset.from_tensor_slices(
                    data_tuple
                ).batch(self.config.model.batch_size))
            else:
                setattr(self, f"{df_type}_input_data", None)
                setattr(self, f"{df_type}_target", None)
                setattr(self, f"{df_type}_greeks", None)
                setattr(self, f"{df_type}_dataset", None)

    def build_model(self):
        config = self.config.model
        model_cls = config.model_class
        neuron_number = config.neuron_per_layer
        layer_number = config.layer_number
        hidden_layer_activation = config.hidden_layer_activation
        initializer = tf.keras.initializers.GlorotNormal(seed=self.config.train_data.seed)

        input_layer = tf.keras.layers.Input(shape=(self.train_input_data.shape[1],))

        i = 1
        hidden_layer = tf.keras.layers.Dense(neuron_number,
                                             activation=hidden_layer_activation,
                                             kernel_initializer=initializer)(
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
                kernel_initializer=initializer,
                bias_initializer=initializer,
                kernel_regularizer=tf.keras.regularizers.L1L2(
                    l1=config.l1, l2=config.l2
                ),
            )(hidden_layer)
            if dropout_pct:
                hidden_layer = tf.keras.layers.Dropout(config.dropout)(
                    hidden_layer
                )
            i += 1

        output_layer = tf.keras.layers.Dense(
            1, activation="relu",
            kernel_initializer=initializer,
            bias_initializer=initializer,
        )(hidden_layer)

        if model_cls == PricerNet:
            model = model_cls(
                    config.tensorflow_greeks,
                    config.greeks_relative_weight,
                    inputs=input_layer,
                    outputs=output_layer,
                )
        else:
            model = model_cls(
                inputs=input_layer, outputs=output_layer
            )
        optimizer = config.optimizer

        model.compile(
            optimizer=optimizer(learning_rate=config.learning_rate), loss="mse",
        )

        return model

    def create_datasets(self):
        for df_type in ["train", "validation", "test"]:
            if getattr(self, f"{df_type}_input_data") is None:
                # TODO: include seed in naming
                data_generator = self.config.__dict__[f"{df_type}_data"]
                data = data_generator.get_data()
                data = data.sample(frac=1,
                                   random_state=self.config.train_data.seed).reset_index(drop=True)
                data.to_parquet(getattr(self, f"{df_type}_save_path"), index=False)
                setattr(self, f"{df_type}_data", data)
                setattr(self, f"{df_type}_input_data", data[self.input_variables].astype(np.float64))
                setattr(self, f"{df_type}_target", data[list(self.target_variables)].astype(np.float64))
                setattr(self, f"{df_type}_greeks", data[list(self.greek_variables.keys())].astype(np.float64))

                if getattr(self, f"{df_type}_greeks").empty:
                    data_tuple = (
                        getattr(self, f"{df_type}_input_data").values,
                        getattr(self, f"{df_type}_target").values
                    )
                else:
                    data_tuple = (
                        getattr(self, f"{df_type}_input_data").values,
                        getattr(self, f"{df_type}_target").values,
                        getattr(self, f"{df_type}_greeks").values
                    )
                dataset = tf.data.Dataset.from_tensor_slices(
                    data_tuple
                ).batch(self.config.model.batch_size)
                setattr(self, f"{df_type}_dataset", dataset)

    def train(self):
        self.create_datasets()
        if not self.model:
            self.model = self.build_model()
            print(self.model.summary())

            log_dir = "logs/fit/" + datetime.date.today().strftime("%Y%m%d") + "_" + self.pricing_model.__name__
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            self.model.fit(
                self.train_dataset,
                epochs=self.config.model.epochs,
                batch_size=self.config.model.batch_size,
                callbacks=[tensorboard_callback]
            )

    def evaluate(self):
        self.train_data.loc[:, "in_sample"] = True
        self.test_data.loc[:, "in_sample"] = False
        concat_data = pd.concat([self.train_data, self.test_data], axis=0).reset_index(drop=True)

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

            calc_greeks = dict()
            calc_greeks["first"] = tape.batch_jacobian(model_preds, input_data)
            calc_greeks["second"] = tape.batch_jacobian(calc_greeks["first"][:, 0, :], input_data)

            del tape

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
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))

            ax[0][0].scatter(concat_data.loc[in_sample_mask, variable],
                          concat_data.loc[
                              in_sample_mask, f"model_{variable + '_AD' if variable != 'price' else variable}"],
                          s=0.1)
            ax[0][0].plot([ax[0][0].get_xlim()[0], ax[0][0].get_xlim()[1]], [ax[0][0].get_xlim()[0], ax[0][0].get_xlim()[1]],
                       color="red")
            ax[0][0].set_title(f"{variable} in sample testing for {self.pricing_model.__name__}")
            ax[0][0].set_xlabel(f"true {variable} value")
            ax[0][0].set_ylabel(f"model {variable} value {'(from AD)' if variable != 'price' else ''}")

            residuals = concat_data.loc[in_sample_mask, variable] - concat_data.loc[
                in_sample_mask, f"model_{variable + '_AD' if variable != 'price' else variable}"]
            ax[0][1].hist(residuals, bins=50)
            ax[0][1].set_title(f"{variable} error (in sample) for {self.pricing_model.__name__}")
            ax[0][1].set_xlabel(f"error")
            ax[0][1].set_ylabel(f"number of observations in a bin")

            ax[1][0].scatter(concat_data.loc[out_sample_mask, variable],
                       concat_data.loc[out_sample_mask, f"model_{variable + '_AD' if variable != 'price' else variable}"],
                       s=0.1)
            ax[1][0].plot([ax[1][0].get_xlim()[0], ax[1][0].get_xlim()[1]], [ax[1][0].get_xlim()[0], ax[1][0].get_xlim()[1]], color="red")
            ax[1][0].set_title(f"{variable} out of sample testing for {self.pricing_model.__name__}")
            ax[1][0].set_xlabel(f"true {variable} value")
            ax[1][0].set_ylabel(f"model {variable} value {'(from AD)' if variable != 'price' else ''}")

            residuals = concat_data.loc[out_sample_mask, variable] - concat_data.loc[out_sample_mask, f"model_{variable + '_AD' if variable != 'price' else variable}"]
            ax[1][1].hist(residuals, bins=50)
            ax[1][1].set_title(f"{variable} error (out of sample) for {self.pricing_model.__name__}")
            ax[1][1].set_xlabel(f"error")
            ax[1][1].set_ylabel(f"number of observations in a bin")

            plt.tight_layout()
            plt.savefig(f"figures/{self.pricing_model.__name__}_{variable}_{self.config.model.neuron_per_layer}_{self.config.model.layer_number}_{self.config.model.hidden_layer_activation}.png")
            plt.close(fig)

        fig, ax = plt.subplots(5, 2, figsize=(20, 25))

        test_case_df = pd.DataFrame(
            np.linspace(self.test_data["underlier_price"].min(), self.test_data["underlier_price"].max(), 100),
            columns=["underlier_price"]
        )
        test_case_df["interest_rate"] = self.train_data["interest_rate"].mean()
        test_case_df["volatility"] = self.train_data["volatility"].mean()
        test_case_df["strike"] = self.train_data["strike"].mean()
        if "barrier" in self.config.pricing_model.input_names:
            test_case_df["underlier_price"] = np.linspace(self.test_data["underlier_price"].min(), self.train_data["barrier"].mean(), 100)
            test_case_df["underlier_price_grid"] = np.linspace(0.0, self.train_data["barrier"].mean() * 1.1, 100)
            test_case_df["barrier"] = self.train_data["barrier"].mean()

        colors = ["k", "r", "b", "g", "c"]
        for i, t in enumerate(np.linspace(0.2, 1.0, 5)):
            if "barrier" not in self.config.pricing_model.input_names:
                test_case_df["expiry"] = t
                pricing_instance = self.config.pricing_model(
                    self.config.train_data.config.pricer_config(
                        **{v: test_case_df.loc[:, v].values for v in self.config.pricing_model.input_names}
                    )
                )
                test_case_df["price"] = pricing_instance.price()
                test_case_df["delta"] = pricing_instance.delta()
                test_case_df["vega"] = pricing_instance.vega()
                test_case_df["theta"] = pricing_instance.theta()
                test_case_df["rho"] = pricing_instance.rho()
            else:
                test_case_df["expiry"] = t
                pricing_instance_config = {
                    "time_grid": np.linspace(0, t * 1.1, 101),
                    "interest_rate": test_case_df["interest_rate"].values[0],
                    "volatility": test_case_df["volatility"].values[0],
                    "strike": test_case_df["strike"].values[0],
                    "barrier": test_case_df["barrier"].values[0],
                }
                pricing_instance_config.update(
                    {v: test_case_df.loc[:, v].values for v in self.config.pricing_model.input_names if
                     v not in pricing_instance_config}
                )
                pricing_instance = self.config.pricing_model(
                    self.config.train_data.config.pricer_config(
                        **pricing_instance_config
                    )
                )
                points = np.zeros(shape=(test_case_df.shape[0], 2))
                points[:, 0] = test_case_df["underlier_price"]
                points[:, 1] = t
                test_case_df["price"] = pricing_instance.price(points)
                test_case_df["delta"] = pricing_instance.delta(points)
                test_case_df["vega"] = pricing_instance.vega(points)
                test_case_df["theta"] = pricing_instance.theta(points)
                test_case_df["rho"] = pricing_instance.rho(points)

            test_case_df["model_price"] = self.model.predict(test_case_df[self.input_variables].values)

            ax[0][0].plot(test_case_df["underlier_price"], test_case_df["price"],
                       label=f"price, expiry={t:0.2f}",
                       color=colors[i])
            ax[0][0].plot(test_case_df["underlier_price"], test_case_df["model_price"],
                       label=f"model price, expiry = {t:0.2f}",
                       color=colors[i], linestyle="--")
            ax[0][0].set_title("Price")
            ax[0][0].set_xlabel("Moneyness")
            ax[0][0].set_ylabel("Price")
            ax[0][0].legend()

            ax[0][1].bar(test_case_df["underlier_price"], test_case_df["price"] - test_case_df["model_price"],
                         width=0.001,
                         color=colors[i], alpha=0.3, label=f"price error, expiry={t:0.2f}")
            ax[0][1].legend()

            input_data = tf.Variable(
                test_case_df[self.input_variables].values,
                dtype=tf.float64,
            )
            with tf.GradientTape() as tape:
                tape.watch(input_data)
                model_preds = self.model(input_data)
            test_case_greeks = tape.batch_jacobian(model_preds, input_data)
            for j, (variable, greek_dict) in enumerate(self.config.model.greeks.items()):
                for greek in greek_dict:
                    if greek["name"] not in test_case_df:
                        continue
                    test_case_df.loc[
                        :,
                        f"model_{greek['name']}_AD",
                    ] = test_case_greeks[:, 0, self.input_variables.index(variable)].numpy()
                    ax[j+1][0].plot(test_case_df["underlier_price"], test_case_df[greek["name"]],
                               label=greek['name'] + f", expiry={t:0.2f}",
                               color=colors[i])
                    ax[j+1][0].plot(test_case_df["underlier_price"], test_case_df[f"model_{greek['name']}_AD"],
                               label=greek['name'] + " from AD" + f", expiry={t:0.2f}",
                               color=colors[i], linestyle="--")
                    ax[j+1][0].set_title(greek['name'])
                    ax[j+1][0].set_xlabel("Moneyness")
                    ax[j+1][0].set_ylabel(greek['name'])
                    ax[j+1][0].legend()

                    ax[j+1][1].bar(test_case_df["underlier_price"], test_case_df[greek["name"]] - test_case_df[f"model_{greek['name']}_AD"],
                                   width=0.001,
                                   color=colors[i], alpha=0.3, label=f"{greek['name']} error, expiry={t:0.2f}")
                    ax[j+1][1].legend()

        plt.tight_layout()
        plt.savefig(f"figures/{self.pricing_model.__name__}_slicing_plot_{self.config.model.neuron_per_layer}_{self.config.model.layer_number}_{self.config.model.hidden_layer_activation}.png")
        return output_dict


if __name__ == "__main__":
    import time
    from config import pipeline_configs

    df_list = []
    for neuron_number, layer_number, activation_function, seed in [
        (32, 10, "tanh", 20250317),
        # (32, 4),
        ]:
        # run_config = pipeline_configs["bs_uo_call"]
        # run_config.train_data.seed = seed
        # run_config.model.hidden_layer_activation = activation_function
        # run_config.model.neuron_per_layer = neuron_number
        # run_config.model.layer_number = layer_number
        # s = time.perf_counter()
        # pipeline = Pipeline(run_config)
        # pipeline.train()
        # df_dict = pipeline.evaluate()

        # run_config = pipeline_configs["bs_put"]
        # run_config.train_data.seed = seed
        # run_config.model.hidden_layer_activation = activation_function
        # run_config.model.neuron_per_layer = neuron_number
        # run_config.model.layer_number = layer_number
        # s = time.perf_counter()
        # pipeline = Pipeline(run_config)
        # pipeline.train()
        # df_dict = pipeline.evaluate()
        #
        run_config = pipeline_configs["bs_call"]
        run_config.train_data.seed = seed
        run_config.model.hidden_layer_activation = activation_function
        run_config.model.neuron_per_layer = neuron_number
        run_config.model.layer_number = layer_number
        s = time.perf_counter()
        pipeline = Pipeline(run_config)
        pipeline.train()
        df_dict = pipeline.evaluate()
