import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch_model import PricerNetTorch, PricerDataset
from torch.utils.data import DataLoader
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
        self.greek_variables = config.model.greeks
        self.greek_relative_weight = config.model.greeks_relative_weight

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

                setattr(self, f"{df_type}_data", data)
                dataset = PricerDataset(data,
                                        self.input_variables,
                                        self.target_variables,
                                        list(self.greek_variables.keys()))
                setattr(self, f"{df_type}_dataset", DataLoader(dataset,
                                                               batch_size=config.model.batch_size,
                                                               shuffle=True)
                        )
            else:
                setattr(self, f"{df_type}_data", None)
                setattr(self, f"{df_type}_dataset", None)

    def build_model(self):
        config = self.config.model
        model_cls = config.model_class
        neuron_number = config.neuron_per_layer
        layer_number = config.layer_number
        hidden_layer_activation = config.hidden_layer_activation

        runtime_config_dict = {
            "input_size": len(self.input_variables),
            "output_size": len(self.target_variables),
            "layer_number": layer_number,
            "neuron_per_layer": neuron_number,
            "activation_function": hidden_layer_activation,
            "learning_rate": config.learning_rate,
            "Z_loss_weight": config.greeks_relative_weight
        }
        model = model_cls(runtime_config_dict)
        return model

    def create_datasets(self):
        for df_type in ["train", "validation", "test"]:
            if getattr(self, f"{df_type}_input_data") is None:
                # TODO: include seed in naming
                data_generator = self.config.__dict__[f"{df_type}_data"]
                data = data_generator.get_data()
                data.to_parquet(getattr(self, f"{df_type}_save_path"), index=False)

                setattr(self, f"{df_type}_data", data)
                dataset = PricerDataset(data,
                                        self.input_variables,
                                        self.target_variables,
                                        list(self.greek_variables.keys()))

                setattr(self, f"{df_type}_dataset", DataLoader(dataset,
                                                               batch_size=self.config.model.batch_size,
                                                               shuffle=True)
                        )

    def train(self):
        self.create_datasets()
        if not self.model:
            self.model = self.build_model()
            self.model.train_model(self.train_dataset, self.config.model.epochs,
                                   pd_metadata=self.greek_variables)

    def evaluate(self):
        self.train_data.loc[:, "in_sample"] = True
        self.test_data.loc[:, "in_sample"] = False
        concat_data = pd.concat([self.train_data, self.test_data], axis=0).reset_index(drop=True)

        calc_batch_size = self.config.model.jacobian_batch_size
        for i in range(concat_data.shape[0] // calc_batch_size + 1):
            input_data = torch.tensor(
                concat_data[self.config.model.input_variables][i * calc_batch_size : (i + 1) * calc_batch_size].values,
                dtype=torch.float32
            )
            input_data.requires_grad = True
            model_price = self.model(input_data)
            concat_data.loc[
            i * calc_batch_size: (i + 1) * calc_batch_size - 1,
            [f"model_{variable}" for variable in self.target_variables],
            ] = model_price.detach().numpy()

            gradients = torch.autograd.grad(outputs=model_price, inputs=input_data,
                                            grad_outputs=torch.ones_like(model_price),
                                            create_graph=True)[0]

            for greek_name, (ad_i, _) in self.greek_variables.items():
                concat_data.loc[
                        i * calc_batch_size : (i + 1) * calc_batch_size - 1,
                        f"model_{greek_name}_AD",
                    ] = gradients[:, ad_i].detach().numpy()

        output_dict = {"neuron_per_layer": self.config.model.neuron_per_layer,
                       "layer_number": self.config.model.layer_number}
        in_sample_mask = concat_data["in_sample"]
        out_sample_mask = ~concat_data["in_sample"]
        for variable in self.config.model.target_variables + list(self.greek_variables.keys()):
            col_name =  f"model_{variable + '_AD' if variable != 'price' else variable}"
            output_dict[f"in_sample_{variable}_mse"] = np.mean(
                (concat_data.loc[in_sample_mask, variable].values - concat_data.loc[in_sample_mask, col_name]) ** 2
            )
            output_dict[f"out_sample_{variable}_mse"] = np.mean(
                (concat_data.loc[out_sample_mask, variable].values - concat_data.loc[out_sample_mask, col_name]) ** 2
            )

        for variable in self.config.model.target_variables + list(self.greek_variables.keys()):
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
            np.linspace(self.test_data["underlier_price"].min(), self.test_data["underlier_price"].max(), 1001),
            columns=["underlier_price"]
        )
        test_case_df["interest_rate"] = self.train_data["interest_rate"].mean()
        test_case_df["volatility"] = self.train_data["volatility"].mean()
        test_case_df["strike"] = self.train_data["strike"].mean()
        if "barrier" in self.config.pricing_model.input_names:
            test_case_df["underlier_price"] = np.linspace(self.test_data["underlier_price"].min(), self.train_data["barrier"].mean(), 1001)
            test_case_df["underlier_price_grid"] = np.linspace(0.0, self.train_data["barrier"].mean() * 1.1, 1001)
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

            test_case_df["model_price"] = self.model(
                torch.tensor(test_case_df[self.input_variables].values, dtype=torch.float32)
            ).detach().numpy()

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

            input_data = torch.tensor(
                test_case_df[self.input_variables].values, dtype=torch.float32
            )
            input_data.requires_grad = True
            test_case_price = self.model(input_data)
            test_case_gradients = torch.autograd.grad(outputs=test_case_price, inputs=input_data,
                                            grad_outputs=torch.ones_like(test_case_price),
                                            create_graph=True)[0]

            for j, (greek_name, (ad_i, _)) in enumerate(self.greek_variables.items()):
                test_case_df.loc[
                    :,
                    f"model_{greek_name}_AD",
                ] = test_case_gradients[:, ad_i].detach().numpy()
                ax[j+1][0].plot(test_case_df["underlier_price"], test_case_df[greek_name],
                           label=greek_name + f", expiry={t:0.2f}",
                           color=colors[i])
                ax[j+1][0].plot(test_case_df["underlier_price"], test_case_df[f"model_{greek_name}_AD"],
                           label=greek_name + " from AD" + f", expiry={t:0.2f}",
                           color=colors[i], linestyle="--")
                ax[j+1][0].set_title(greek_name)
                ax[j+1][0].set_xlabel("Moneyness")
                ax[j+1][0].set_ylabel(greek_name)
                ax[j+1][0].legend()

                ax[j+1][1].bar(test_case_df["underlier_price"],
                               test_case_df[greek_name] - test_case_df[f"model_{greek_name}_AD"],
                               width=0.001,
                               color=colors[i], alpha=0.3, label=f"{greek_name} error, expiry={t:0.2f}")
                ax[j+1][1].legend()

        plt.tight_layout()
        plt.savefig(f"figures/{self.pricing_model.__name__}_slicing_plot_{self.config.model.neuron_per_layer}_{self.config.model.layer_number}_{self.config.model.hidden_layer_activation}.png")
        return output_dict


if __name__ == "__main__":
    import time
    from config import pipeline_configs

    df_list = []
    for neuron_number, layer_number, activation_function, seed in [
        (100, 10, "leaky_relu", 20250320),
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
