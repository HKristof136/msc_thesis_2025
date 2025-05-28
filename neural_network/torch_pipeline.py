import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score

from neural_network.torch_model import PricerDataset
from neural_network.config_base import PipeLineConfig
from get_logger import get_logger

logger = get_logger()

plt.style.use('seaborn-v0_8-bright')


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
        logger.info("Iniatializing Pipeline")
        self.config = config
        self.regenerate_data = config.regenerate_data
        self.model = None
        self.pricing_model = config.pricing_model
        self.input_variables = config.model.input_variables
        self.target_variables = config.model.target_variables
        self.greek_variables = config.model.greeks
        self.lamba_param = config.model.lambda_param

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
                logger.info(f"Loading {df_type} data from {save_path}")
                data = pd.read_parquet(save_path).reset_index(drop=True)

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
        logger.info(f"Building torch neural network, from class {self.config.model.model_class.__name__}")
        config = self.config.model
        model_cls = config.model_class
        neuron_number = config.neuron_per_layer
        layer_number = config.layer_number
        hidden_layer_activation = config.hidden_layer_activation
        learning_rate = config.learning_rate
        lambda_param = config.lambda_param

        runtime_config_dict = {
            "input_variables": self.input_variables,
            "target_variables": self.target_variables,
            "layer_number": layer_number,
            "neuron_per_layer": neuron_number,
            "activation_function": hidden_layer_activation,
            "learning_rate": learning_rate,
            "lambda_param": lambda_param,
            "calc_greek_regularization": config.calc_greek_regularization,
            "greek_weighting": config.greek_weighting,
        }
        logger.info(f"Using run time config items for {config.model_class.__name__}: {runtime_config_dict}")
        model = model_cls(runtime_config_dict)
        return model

    def create_datasets(self):
        for df_type in ["train", "validation", "test"]:
            if getattr(self, f"{df_type}_data") is None:
                if os.path.exists(getattr(self, f"{df_type}_save_path")):
                    data = pd.read_parquet(getattr(self, f"{df_type}_save_path")).reset_index(drop=True)
                else:
                    logger.info(f"Loading {df_type} data not found at {getattr(self, f'{df_type}_save_path')}")
                    data_generator = self.config.__dict__[f"{df_type}_data"]
                    data = data_generator.get_data()
                    logger.info(f"Saving {df_type} data to {getattr(self, f'{df_type}_save_path')}")
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

    def train(self, retrain=False):
        self.create_datasets()
        if not self.model or retrain:
            self.model = self.build_model()

            logger.info(f"Starting model training: batch_size {self.config.model.batch_size}, number of epochs: {self.config.model.epochs}")

            # self.model.train_model(self.train_dataset, self.config.model.epochs,
            #                        pd_metadata=self.greek_variables)
            self.model.train_model(self.train_data, self.config.model.epochs,
                                   pd_metadata=self.greek_variables)
            model_name = f"{self.config.pricing_model.__name__}_{self.config.model.neuron_per_layer}_{self.config.model.layer_number}_{self.config.model.hidden_layer_activation}{'_greek_reg' if self.config.model.calc_greek_regularization else ''}{'_weighted' if self.config.model.greek_weighting and False else ''}{'_lambda_' + str(self.lamba_param) if self.lamba_param else ''}"
            logger.info(f"Saving information with nameing: {model_name}")

            os.makedirs(os.path.join("saved_models", str(self.config.train_data.seed)), exist_ok=True)
            torch.save(self.model.state_dict(), f"saved_models/{self.config.train_data.seed}/{model_name}.model")
            logger.info(f"Model parameters saved to: {f'saved_models/{self.config.train_data.seed}/{model_name}.model'}")

            os.makedirs(os.path.join("model_memory", str(self.config.train_data.seed)), exist_ok=True)
            df = pd.concat([pd.DataFrame(
                pd.DataFrame({key: val for key, val in self.model.train_memory[i].items() if val}).mean(axis=0)).T for i
                            in self.model.train_memory]).reset_index(drop=True)
            df.to_csv(os.path.join(os.getcwd(), "model_memory", str(self.config.train_data.seed), f"{model_name}.csv"), index=False)
            logger.info(f"Model training memory per epoch saved to: {os.path.join(os.getcwd(), 'model_memory', str(self.config.train_data.seed), f'{model_name}.csv')}")

            df_list = [pd.DataFrame({key: val for key, val in self.model.train_memory[i].items() if val}) for i in
                 self.model.train_memory]
            for i, epoch_df in enumerate(df_list):
                epoch_df["epoch"] = i+1
            df_full = pd.concat(
                df_list).reset_index(drop=True)
            df_full.to_csv(os.path.join(os.getcwd(), "model_memory", str(self.config.train_data.seed), f"{model_name}_full.csv"), index=False)
            logger.info(f"Model training memory per batch saved to: {os.path.join(os.getcwd(), 'model_memory', str(self.config.train_data.seed), f'{model_name}_full.csv')}")


    def evaluate(self, model_path, create_plots=True):
        logger.info(f"Starting trained model evaluation")
        if not self.model:
            self.model = self.build_model()
            if os.path.exists(model_path):
                logger.info(f"Loading state_dict from {model_path}")
                self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device("cpu")))
        self.train_data.loc[:, "in_sample"] = True
        self.validation_data.loc[:, "in_sample"] = False
        concat_data = pd.concat([self.train_data, self.validation_data], axis=0).reset_index(drop=True)

        calc_batch_size = self.config.model.jacobian_batch_size
        logger.info(f"Getting model outputs for train and validation data")
        for i in (pbar := tqdm(range(concat_data.shape[0] // calc_batch_size + 1))):
            pbar.set_description("Batch processing model output")
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

            if self.greek_variables:
                gradients = torch.autograd.grad(outputs=model_price, inputs=input_data,
                                                grad_outputs=torch.ones_like(model_price),
                                                create_graph=True)[0]

                for greek_name, (ad_i, _) in self.greek_variables.items():
                    concat_data.loc[
                            i * calc_batch_size : (i + 1) * calc_batch_size - 1,
                            f"model_{greek_name}_AD",
                        ] = gradients[:, ad_i].detach().numpy()

        logger.info(f"Getting model outputs for test data")
        for i in (pbar := tqdm(range(self.test_data.shape[0] // calc_batch_size + 1))):
            pbar.set_description("Batch processing model output")
            test_tensor = torch.tensor(
                self.test_data[self.config.model.input_variables][i * calc_batch_size : (i + 1) * calc_batch_size].values,
                dtype=torch.float32
            )
            test_tensor.requires_grad = True
            test_price = self.model(test_tensor)
            self.test_data.loc[
                i * calc_batch_size: (i + 1) * calc_batch_size - 1,
                [f"model_{variable}" for variable in self.target_variables],
            ] = test_price.detach().numpy()
            if self.greek_variables:
                test_gradients = torch.autograd.grad(outputs=test_price, inputs=test_tensor,
                                                     grad_outputs=torch.ones_like(test_price),
                                                     create_graph=True)[0]

                for greek_name, (ad_i, _) in self.greek_variables.items():
                    self.test_data.loc[
                        i * calc_batch_size: (i + 1) * calc_batch_size - 1,
                        f"model_{greek_name}_AD",
                    ] = test_gradients[:, ad_i].detach().numpy()

        output_dict = {"neuron_per_layer": self.config.model.neuron_per_layer,
                       "layer_number": self.config.model.layer_number,
                       "hidden_layer_activation": self.config.model.hidden_layer_activation,
                       "learning_rate": self.config.model.learning_rate,
                       "lambda_param": self.config.model.lambda_param,
        }
        in_sample_mask = concat_data["in_sample"]
        out_sample_mask = ~concat_data["in_sample"]
        for variable in self.config.model.target_variables + list(self.greek_variables.keys()):
            col_name =  f"model_{variable + '_AD' if variable not in self.config.model.target_variables else variable}"
            output_dict[f"in_sample_{variable}_mse"] = np.mean(
                (concat_data.loc[in_sample_mask, variable].values - concat_data.loc[in_sample_mask, col_name]) ** 2
            )
            output_dict[f"out_sample_{variable}_mse"] = np.mean(
                (concat_data.loc[out_sample_mask, variable].values - concat_data.loc[out_sample_mask, col_name]) ** 2
            )
            output_dict[f"test_{variable}_mse"] = np.mean(
                (self.test_data[variable].values - self.test_data[col_name]) ** 2
            )

        if create_plots:
            self.create_histogram_plot(concat_data)
            self.create_slicing_plot()
        # model_name = f"{self.config.pricing_model.__name__}_{self.config.model.neuron_per_layer}_{self.config.model.layer_number}_{self.config.model.hidden_layer_activation}{'_greek_reg' if self.config.model.calc_greek_regularization else ''}{'_weighted' if self.config.model.greek_weighting else ''}"
        # with open(f"model_memory/{self.config.train_data.seed}/{model_name}.pickle", "wb+") as f:
        #     pickle.dump(output_dict, f)
        return output_dict

    def create_histogram_plot(self, df):
        model_name = f"{self.config.pricing_model.__name__}_{self.config.model.neuron_per_layer}_{self.config.model.layer_number}_{self.config.model.hidden_layer_activation}{'_greek_reg' if self.config.model.calc_greek_regularization else ''}{'_weighted' if self.config.model.greek_weighting else ''}"

        in_sample_mask = df["in_sample"]
        out_sample_mask = ~df["in_sample"]

        for variable in self.target_variables + list(self.greek_variables.keys()):
            if "partial" in variable:
                continue
            logger.info(f"Creating histogram and y=X plots for variable: {variable}")
            model_variable_name = f"model_{variable + '_AD' if variable not in self.target_variables else variable}"

            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            [x.grid(True) for x in ax.flatten()]
            colors = [x["color"] for x in list(plt.rcParams['axes.prop_cycle'])]

            ax[0][0].scatter(
                df.loc[in_sample_mask, variable],
                df.loc[
                    in_sample_mask,
                    model_variable_name],
                    s=0.1,
                    color=colors[0]
            )
            ax[0][0].plot(
                [ax[0][0].get_xlim()[0], ax[0][0].get_xlim()[1]],
                [ax[0][0].get_xlim()[0], ax[0][0].get_xlim()[1]],
                color="red"
            )
            latex_dict = {"delta": r"$\Delta$", "vega": r"$\nu$",
                          "theta": r"$\theta_\tau$", "rho": r"$\rho$",
                          "price": "ár", "volatility": r"$\sigma_{IV}$",
                          "modified_vega": r"$\hat{P}/\nu$"}
            ax[0][0].set_title(f"Tanuló adathalmaz, {latex_dict[variable]}", fontsize=14)
            ax[0][0].set_xlabel(f"valódi {latex_dict[variable]} érték")
            ax[0][0].set_ylabel(f"neurális háló által prediktált {latex_dict[variable]} érték")

            rsquared = r2_score(
                df.loc[in_sample_mask, variable].values,
                df.loc[
                    in_sample_mask,
                    model_variable_name
                ].values)
            ax[0][0].text(abs(ax[0][0].get_xlim()[0]) * 1.25, ax[0][0].get_ylim()[1] * 0.9,
                          fr"$R^2={rsquared:.4f}$")

            residuals = df.loc[in_sample_mask, variable] - df.loc[
                in_sample_mask,
                model_variable_name
            ]

            ax[0][1].hist(residuals, bins=50)
            ax[0][1].set_title(f"Tanuló adathalmaz, {latex_dict[variable]} hiba hisztogram", fontsize=14)
            ax[0][1].set_xlabel(f"hiba")
            ax[0][1].set_ylabel(f"megfigyelések száma")

            ax[1][0].scatter(
                df.loc[out_sample_mask, variable],
                df.loc[
                    out_sample_mask,
                    model_variable_name],
                    s=0.1,
                    color=colors[0]
            )
            ax[1][0].plot(
                [ax[1][0].get_xlim()[0], ax[1][0].get_xlim()[1]],
                [ax[1][0].get_xlim()[0], ax[1][0].get_xlim()[1]],
                color="red")
            ax[1][0].set_title(f"Teszt adathalmaz, {latex_dict[variable]}", fontsize=14)
            ax[1][0].set_xlabel(f"valódi {latex_dict[variable]} érték")
            ax[1][0].set_ylabel(f"neurális háló által prediktált {latex_dict[variable]} érték")

            rsquared_test = r2_score(
                df.loc[out_sample_mask, variable].values,
                df.loc[
                    out_sample_mask,
                    model_variable_name
                ].values)
            ax[1][0].text(abs(ax[1][0].get_xlim()[0]) * 1.25, ax[1][0].get_ylim()[1] * 0.9,
                          fr"$R^2={rsquared_test:.4f}$")

            residuals = df.loc[out_sample_mask, variable] - df.loc[
                out_sample_mask,
                model_variable_name
            ]
            ax[1][1].hist(residuals, bins=50)
            ax[1][1].set_title(f"Teszt adathalmaz, {latex_dict[variable]} hiba hisztogram", fontsize=14)
            ax[1][1].set_xlabel(f"hiba")
            ax[1][1].set_ylabel(f"megfigyelések száma")

            plt.tight_layout()
            plt.savefig(f"figures/{model_name}_{variable}.png")
            plt.close(fig)


    def create_slicing_plot(self, slice_config=None):
        model_name = f"{self.config.pricing_model.__name__}_{self.config.model.neuron_per_layer}_{self.config.model.layer_number}_{self.config.model.hidden_layer_activation}{'_greek_reg' if self.config.model.calc_greek_regularization else ''}{'_weighted' if self.config.model.greek_weighting else ''}"

        if slice_config is None:
            slice_config = dict()

        n_rows = len(self.target_variables + list(self.greek_variables.keys()))
        fig, ax = plt.subplots(
            n_rows,
            2,
            figsize=(20, 5 * n_rows)
        )
        [x.grid(True) for x in ax.flatten()]

        logger.info(r"Creating train average slicing plot")

        if self.pricing_model.__name__ in ["BlackScholesCall", "BlackScholesPut"]:
            n = slice_config.get("n", 501)
            test_case_df = pd.DataFrame(
                slice_config.get(
                    "underlier_price",
                    np.linspace(
                        self.train_data["underlier_price"].quantile(0.1),
                        self.train_data["underlier_price"].quantile(0.9),
                        n
                    )
                ),
                columns=["underlier_price"]
            )
            test_case_df["interest_rate"] = slice_config.get("interest_rate", self.train_data["interest_rate"].mean())
            test_case_df["strike"] = slice_config.get("strike", self.train_data["strike"].mean())
            test_case_df["volatility"] = slice_config.get("volatility", self.train_data["volatility"].mean())

            colors = [x["color"] for x in list(plt.rcParams['axes.prop_cycle'])]
            xlim1, xlim2 = test_case_df["underlier_price"].min(), test_case_df["underlier_price"].max()
            ylim1, ylim2 = -0.001, 0.001
            for i, (t, expiry_str) in enumerate(
                    zip(
                        [30/365, 91/365, 182/365, 273/365],
                        ["1M", "3M", "6M", "9M"]
            )):
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

                test_case_df["model_price"] = self.model(
                    torch.tensor(test_case_df[self.input_variables].values, dtype=torch.float32)
                ).detach().numpy()

                ax[0][0].plot(
                    test_case_df["underlier_price"], test_case_df["price"],
                    label=f"opcióár, lejárat={expiry_str}", color=colors[i])
                ax[0][0].plot(
                    test_case_df["underlier_price"], test_case_df["model_price"],
                    label=f"model ár, lejárat={expiry_str}", color=colors[i], linestyle="--")
                ax[0][0].set_title("Opcióár", fontsize=14)
                ax[0][0].set_xlabel(r"$S_0/K$")
                ax[0][0].set_ylabel("opcióár")
                ax[0][0].legend()

                ax[0][1].bar(test_case_df["underlier_price"], (test_case_df["price"] - test_case_df["model_price"]).abs(),
                             width=0.003,
                             color=colors[i], alpha=0.3, label=f"ár L1 hiba, lejárat={expiry_str}")
                ax[0][1].legend()

                ylim1, ylim2 = min(ylim1, ax[0][1].get_ylim()[0]), max(ylim2, ax[0][1].get_ylim()[1])

                input_data = torch.tensor(
                    test_case_df[self.input_variables].values, dtype=torch.float32
                )
                input_data.requires_grad = True
                test_case_price = self.model(input_data)
                test_case_gradients = torch.autograd.grad(outputs=test_case_price, inputs=input_data,
                                                          grad_outputs=torch.ones_like(test_case_price),
                                                          create_graph=True)[0]

                latex_dict = {"delta": r"$\Delta$", "vega": r"$\nu$",
                              "theta": r"$\theta_\tau$", "rho": r"$\rho$"}
                for j, (greek_name, (ad_i, _)) in enumerate(self.greek_variables.items()):
                    test_case_df.loc[:, f"model_{greek_name}_AD", ] = test_case_gradients[:, ad_i].detach().numpy()
                    ax[j + 1][0].plot(
                        test_case_df["underlier_price"], test_case_df[greek_name],
                        label=f"{latex_dict[greek_name]}, lejárat={expiry_str}", color=colors[i])
                    ax[j + 1][0].plot(
                        test_case_df["underlier_price"], test_case_df[f"model_{greek_name}_AD"],
                        label= f"$model_{'{' + latex_dict[greek_name].replace('$', '') + '}'}$, lejárat={expiry_str}", color=colors[i], linestyle="--")
                    ax[j + 1][0].set_title(latex_dict[greek_name], fontsize=14)
                    ax[j + 1][0].set_xlabel(r"$S_0/K$")
                    ax[j + 1][0].set_ylabel(latex_dict[greek_name])
                    ax[j + 1][0].legend()

                    ax[j + 1][1].bar(
                        test_case_df["underlier_price"],
                        (test_case_df[greek_name] - test_case_df[f"model_{greek_name}_AD"]).abs(),
                        width=0.003, color=colors[i], alpha=0.3, label=f"{latex_dict[greek_name]} L1 hiba, lejárat={expiry_str}")
                    ax[j + 1][1].legend()

                    ylim1, ylim2 = min(ylim1, ax[j + 1][1].get_ylim()[0]), max(ylim2, ax[j + 1][1].get_ylim()[1])

            [x.set_ylim(ylim1, ylim2) for x in ax[:, 1].flatten()]
            [x.hlines(y=0.0, xmin=xlim1, xmax=xlim2, color="black") for x in ax[:, 0].flatten()]
            [x.set_xlim(xlim1, xlim2) for x in ax[:, 0].flatten()]

        elif self.pricing_model.__name__ in [
            "BarrierUpAndOutCallPDE",
            "BarrierUpAndInPutPDE",
            "BarrierDownAndOutPutPDE",
            "BarrierDownAndInCallPDE"
        ]:
            n = slice_config.get("n", 301)
            test_case_df = pd.DataFrame(
                slice_config.get(
                    "underlier_price",
                    np.linspace(
                        self.train_data["underlier_price"].quantile(0.1),
                        self.train_data["underlier_price"].quantile(0.9),
                        n
                    )
                ),
                columns=["underlier_price"]
            )
            test_case_df["interest_rate"] = slice_config.get("interest_rate", self.train_data["interest_rate"].mean())
            test_case_df["strike"] = slice_config.get("strike", self.train_data["strike"].mean())
            test_case_df["volatility"] = slice_config.get("volatility", self.train_data["volatility"].mean())

            if "Up" in self.config.pricing_model.__name__:
                test_case_df["underlier_price"] = np.linspace(
                    self.train_data["underlier_price"].quantile(0.15),
                    slice_config.get("barrier", self.train_data["barrier"].mean()) - 0.02,
                    n
                )
                test_case_df["underlier_price_grid"] = np.linspace(
                    0.01,
                    slice_config.get("barrier", self.train_data["barrier"].mean()) + 0.02,
                    n
                )
            else:
                test_case_df["underlier_price"] = np.linspace(
                    slice_config.get("barrier", self.train_data["barrier"].mean()) + 0.01,
                    self.train_data["underlier_price"].quantile(0.85),
                    n
                )

            test_case_df["barrier"] = slice_config.get("barrier", self.train_data["barrier"].mean())

            colors = [x["color"] for x in list(plt.rcParams['axes.prop_cycle'])]
            ylim1, ylim2 = -0.001, 0.001
            xlim1, xlim2 = test_case_df["underlier_price"].min(), test_case_df["underlier_price"].max()
            for i, (t, expiry_str) in enumerate(
                    zip(
                        [30/365, 91/365, 182/365, 273/365],
                        ["1M", "3M", "6M", "9M"]
            )):
                if expiry_str == "1M":
                    continue
                test_case_df["expiry"] = t
                test_case_df["expiry"] = t
                pricing_instance_config = {
                    "time_grid": np.linspace(0, t + 0.01, 101),
                    "interest_rate": test_case_df["interest_rate"].values[0],
                    "volatility": test_case_df["volatility"].values[0],
                    "strike": test_case_df["strike"].values[0],
                    "barrier": test_case_df["barrier"].values[0],
                }
                if "Down" in self.config.pricing_model.__name__:
                    ds = np.linspace(
                        0.01,
                        slice_config.get("barrier", self.train_data["barrier"].mean()) + 0.01,
                        n
                    )
                    ds = ds[1] - ds[0]
                    pricing_instance_config["underlier_price_grid"] = np.arange(
                        0.01,
                        slice_config.get("strike", self.train_data["strike"].mean()) * 3,
                        ds
                    )
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

                ax[0][0].plot(
                    test_case_df["underlier_price"], test_case_df["price"],
                    label=f"opcióár, lejárat={expiry_str}", color=colors[i])
                ax[0][0].plot(
                    test_case_df["underlier_price"], test_case_df["model_price"],
                    label=f"model ár, lejárat={expiry_str}", color=colors[i], linestyle="--")
                ax[0][0].set_title("Opcióár", fontsize=14)
                ax[0][0].set_xlabel(r"$S_0/K$")
                ax[0][0].set_ylabel("opcióár")
                ax[0][0].legend()

                ax[0][1].bar(test_case_df["underlier_price"], (test_case_df["price"] - test_case_df["model_price"]).abs(),
                             width=0.003,
                             color=colors[i], alpha=0.3, label=f"ár L1 hiba, lejárat={expiry_str}")
                ax[0][1].legend()

                ylim1, ylim2 = min(ylim1, ax[0][1].get_ylim()[0]), max(ylim2, ax[0][1].get_ylim()[1])

                input_data = torch.tensor(
                    test_case_df[self.input_variables].values, dtype=torch.float32
                )
                input_data.requires_grad = True
                test_case_price = self.model(input_data)
                test_case_gradients = torch.autograd.grad(outputs=test_case_price, inputs=input_data,
                                                          grad_outputs=torch.ones_like(test_case_price),
                                                          create_graph=True)[0]

                latex_dict = {"delta": r"$\Delta$", "vega": r"$\nu$",
                              "theta": r"$\theta_\tau$", "rho": r"$\rho$"}
                for j, (greek_name, (ad_i, _)) in enumerate(self.greek_variables.items()):
                    test_case_df.loc[:, f"model_{greek_name}_AD", ] = test_case_gradients[:, ad_i].detach().numpy()
                    ax[j + 1][0].plot(
                        test_case_df["underlier_price"], test_case_df[greek_name],
                        label=f"{latex_dict[greek_name]}, lejárat={expiry_str}", color=colors[i])
                    ax[j + 1][0].plot(
                        test_case_df["underlier_price"], test_case_df[f"model_{greek_name}_AD"],
                        label= f"$model_{'{' + latex_dict[greek_name].replace('$', '') + '}'}$, lejárat={expiry_str}", color=colors[i], linestyle="--")
                    ax[j + 1][0].set_title(latex_dict[greek_name], fontsize=14)
                    ax[j + 1][0].set_xlabel(r"$S_0/K$")
                    ax[j + 1][0].set_ylabel(latex_dict[greek_name])
                    ax[j + 1][0].legend()

                    ax[j + 1][1].bar(
                        test_case_df["underlier_price"],
                        (test_case_df[greek_name] - test_case_df[f"model_{greek_name}_AD"]).abs(),
                        width=0.003, color=colors[i], alpha=0.3, label=f"{latex_dict[greek_name]} L1 hiba, lejárat={expiry_str}")
                    ax[j + 1][1].legend()

                    ylim1, ylim2 = min(ylim1, ax[j + 1][1].get_ylim()[0]), max(ylim2, ax[j + 1][1].get_ylim()[1])

            [x.set_ylim(ylim1, ylim2) for x in ax[:, 1].flatten()]
            [x.hlines(y=0.0, xmin=xlim1, xmax=xlim2, color="black") for x in ax[:, 0].flatten()]
            [x.set_xlim(xlim1, xlim2) for x in ax[:, 0].flatten()]

        elif self.pricing_model.__name__ in [
            "ADICallPDE",
            "ADIBarrierUpAndOutCallPDE"
        ]:
            n = slice_config.get("n", 301)
            test_case_df = pd.DataFrame(
                slice_config.get(
                    "underlier_price",
                    np.linspace(
                        self.train_data["underlier_price"].quantile(0.1),
                        self.train_data["underlier_price"].quantile(0.9),
                        n
                    )
                ),
                columns=["underlier_price"]
            )
            test_case_df["interest_rate"] = slice_config.get("interest_rate", self.train_data["interest_rate"].mean())
            test_case_df["strike"] = slice_config.get("strike", self.train_data["strike"].mean())

            if "Barrier" in self.config.pricing_model.__name__:
                test_case_df["underlier_price"] = np.linspace(
                    self.train_data["underlier_price"].quantile(0.15),
                    slice_config.get("barrier", self.train_data["barrier"].mean()) - 0.01,
                    n
                )
                test_case_df["barrier"] = slice_config.get("barrier", self.train_data["barrier"].mean())
                test_case_df["underlier_price"] = np.linspace(
                    self.train_data["underlier_price"].quantile(0.1),
                    self.train_data["barrier"].mean() - 0.005,
                    n
                )

            test_case_df["kappa"] = slice_config.get("kappa", self.train_data["kappa"].mean())
            test_case_df["variance_theta"] = slice_config.get("variance_theta", self.train_data["variance_theta"].mean())
            test_case_df["sigma"] = slice_config.get("sigma", self.train_data["sigma"].mean())
            test_case_df["corr"] = slice_config.get("corr", self.train_data["corr"].mean())
            test_case_df["initial_variance"] = slice_config.get("initial_variance",
                                                                self.train_data["initial_variance"].mean()
                                                                )

            colors = [x["color"] for x in list(plt.rcParams['axes.prop_cycle'])]
            ylim1, ylim2 = -0.001, 0.001
            xlim1, xlim2 = test_case_df["underlier_price"].min(), test_case_df["underlier_price"].max()
            for i, (t, expiry_str) in enumerate(
                    zip(
                        [30/365, 91/365, 182/365, 273/365],
                        ["1M", "3M", "6M", "9M"]
            )):
                if "Barrier" in self.config.pricing_model.__name__:
                    if t < 60/365:
                        continue
                test_case_df["expiry"] = t
                pricing_instance_config = {
                    "underlier_price_grid": np.array([]),
                    "time_grid": np.linspace(0, t + 0.01, 101),
                    "interest_rate": test_case_df["interest_rate"].values[0],
                    "strike": test_case_df["strike"].values[0],
                    "barrier": test_case_df["barrier"].values[0] if "Barrier" in self.pricing_model.__name__ else None,
                    "kappa": test_case_df["kappa"].values[0],
                    "variance_theta": test_case_df["variance_theta"].values[0],
                    "sigma": test_case_df["sigma"].values[0],
                    "corr": test_case_df["corr"].values[0],
                    "n": 101,
                    "m": 501,
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
                points = np.zeros(shape=(test_case_df.shape[0], 3))
                points[:, 0] = test_case_df["initial_variance"].values[0]
                points[:, 1] = t
                points[:, 2] = test_case_df["underlier_price"]

                test_case_df["price"] = pricing_instance.price(points)
                test_case_df["delta"] = pricing_instance.delta(points)
                test_case_df["vega"] = pricing_instance.vega(points)
                test_case_df["theta"] = pricing_instance.theta(points)
                test_case_df["rho"] = pricing_instance.rho(points)

                test_case_df["model_price"] = self.model(
                    torch.tensor(test_case_df[self.input_variables].values, dtype=torch.float32)
                ).detach().numpy()

                ax[0][0].plot(
                    test_case_df["underlier_price"], test_case_df["price"],
                    label=f"opcióár, lejárat={expiry_str}", color=colors[i])
                ax[0][0].plot(
                    test_case_df["underlier_price"], test_case_df["model_price"],
                    label=f"model ár, lejárat={expiry_str}", color=colors[i], linestyle="--")
                ax[0][0].set_title("Opcióár", fontsize=14)
                ax[0][0].set_xlabel(r"$S_0/K$")
                ax[0][0].set_ylabel("opcióár")
                ax[0][0].legend()

                ax[0][1].bar(test_case_df["underlier_price"], (test_case_df["price"] - test_case_df["model_price"]).abs(),
                             width=0.003,
                             color=colors[i], alpha=0.3, label=f"ár L1 hiba, lejárat={expiry_str}")
                ax[0][1].legend()

                ylim1, ylim2 = min(ylim1, ax[0][1].get_ylim()[0]), max(ylim2, ax[0][1].get_ylim()[1])

                input_data = torch.tensor(
                    test_case_df[self.input_variables].values, dtype=torch.float32
                )
                input_data.requires_grad = True
                test_case_price = self.model(input_data)
                test_case_gradients = torch.autograd.grad(outputs=test_case_price, inputs=input_data,
                                                          grad_outputs=torch.ones_like(test_case_price),
                                                          create_graph=True)[0]

                latex_dict = {"delta": r"$\Delta$", "vega": r"$\nu$",
                              "theta": r"$\theta_\tau$", "rho": r"$\rho$"}

                for j, (greek_name, (ad_i, _)) in enumerate(self.greek_variables.items()):
                    if "partial" in greek_name:
                        continue
                    test_case_df.loc[:, f"model_{greek_name}_AD", ] = test_case_gradients[:, ad_i].detach().numpy()
                    ax[j + 1][0].plot(
                        test_case_df["underlier_price"], test_case_df[greek_name],
                        label=f"{latex_dict[greek_name]}, lejárat={expiry_str}", color=colors[i])
                    ax[j + 1][0].plot(
                        test_case_df["underlier_price"], test_case_df[f"model_{greek_name}_AD"],
                        label= f"$model_{'{' + latex_dict[greek_name].replace('$', '') + '}'}$, lejárat={expiry_str}", color=colors[i], linestyle="--")
                    ax[j + 1][0].set_title(latex_dict[greek_name], fontsize=14)
                    ax[j + 1][0].set_xlabel(r"$S_0/K$")
                    ax[j + 1][0].set_ylabel(latex_dict[greek_name])
                    ax[j + 1][0].legend()

                    ax[j + 1][1].bar(
                        test_case_df["underlier_price"],
                        (test_case_df[greek_name] - test_case_df[f"model_{greek_name}_AD"]).abs(),
                        width=0.003, color=colors[i], alpha=0.3, label=f"{latex_dict[greek_name]} L1 hiba, lejárat={expiry_str}")
                    ax[j + 1][1].legend()

                    ylim1, ylim2 = min(ylim1, ax[j + 1][1].get_ylim()[0]), max(ylim2, ax[j + 1][1].get_ylim()[1])

            [x.set_ylim(ylim1, ylim2) for x in ax[:, 1].flatten()]
            [x.hlines(y=0.0, xmin=xlim1, xmax=xlim2, color="black") for x in ax[:, 0].flatten()]
            [x.set_xlim(xlim1, xlim2) for x in ax[:, 0].flatten()]

        elif self.pricing_model.__name__ in [
            "ImpliedVol"
        ]:
            n = slice_config.get("n", 501)
            test_case_df = pd.DataFrame(
                slice_config.get(
                    "underlier_price",
                    np.linspace(
                        self.train_data["underlier_price"].quantile(0.1),
                        self.train_data["underlier_price"].quantile(0.9),
                        n
                    )
                ),
                columns=["underlier_price"]
            )
            test_case_df["interest_rate"] = slice_config.get("interest_rate", self.train_data["interest_rate"].mean())
            test_case_df["strike"] = slice_config.get("strike", self.train_data["strike"].mean())
            test_case_df["volatility"] = slice_config.get("volatility", self.train_data["volatility"].mean())

            colors = [x["color"] for x in list(plt.rcParams['axes.prop_cycle'])]
            ylim1, ylim2 = -0.001, 0.001
            xlim1, xlim2 = test_case_df["underlier_price"].min(), test_case_df["underlier_price"].max()
            for i, (t, expiry_str) in enumerate(
                    zip(
                        [30/365, 91/365, 182/365, 273/365],
                        ["1M", "3M", "6M", "9M"]
            )):
                test_case_df["expiry"] = t
                pricing_instance = self.config.train_data.config.pricer(
                    self.config.train_data.config.pricer_config(
                        **{v: test_case_df.loc[:, v].values for v in self.config.train_data.config.pricer.input_names}
                    )
                )
                test_case_df["price"] = pricing_instance.price()
                test_case_df["vega"] = pricing_instance.vega()
                test_case_df["time_value"] = test_case_df["price"] - np.maximum(0, test_case_df["underlier_price"] - np.exp(-test_case_df["interest_rate"] * test_case_df["expiry"]))
                test_case_df = test_case_df.loc[test_case_df["time_value"] > 1e-4]
                test_case_df["log_time_value"] = np.log(test_case_df["time_value"])

                test_case_df["modified_vega"] = (1 / test_case_df["vega"]) * test_case_df["time_value"]

                test_case_df["model_volatility"] = self.model(
                    torch.tensor(test_case_df[self.input_variables].values, dtype=torch.float32)
                ).detach().numpy()

                if expiry_str == "1M":
                    ax[0][0].plot(
                        test_case_df["underlier_price"], test_case_df["volatility"],
                        label=r"$\sigma_{IV}$", color="black")
                ax[0][0].plot(
                    test_case_df["underlier_price"], test_case_df["model_volatility"],
                    label=r"model $\sigma_{'IV'}$" + f", lejárat={expiry_str}", color=colors[i], linestyle="--")
                ax[0][0].set_title("Implikált volatilitás", fontsize=14)
                ax[0][0].set_xlabel(r"$S_0/K$")
                ax[0][0].set_ylabel(r"$\sigma_{IV}$")
                ax[0][0].legend()
                ylim_1_iv, ylim_2_iv = ax[0][0].get_ylim()
                ax[0][0].set_ylim(ylim_1_iv - 0.002, ylim_2_iv + 0.002)

                ax[0][1].bar(test_case_df["underlier_price"], (test_case_df["volatility"] - test_case_df["model_volatility"]).abs(),
                             width=0.003,
                             color=colors[i], alpha=0.3, label=r"$\sigma_{IV}$" + f", lejárat={expiry_str}")
                ax[0][1].legend()
                ylim1, ylim2 = min(ylim1, ax[0][1].get_ylim()[0]), max(ylim2, ax[0][1].get_ylim()[1])

                input_data = torch.tensor(
                    test_case_df[self.input_variables].values, dtype=torch.float32
                )
                input_data.requires_grad = True
                test_case_price = self.model(input_data)
                test_case_gradients = torch.autograd.grad(outputs=test_case_price, inputs=input_data,
                                                          grad_outputs=torch.ones_like(test_case_price),
                                                          create_graph=True)[0]
                test_case_df.loc[:, "model_price_deriv_AD"] = test_case_gradients[:, 0].detach().numpy()
                ax[1][0].plot(
                    test_case_df["underlier_price"], test_case_df["modified_vega"],
                    label=r"valódi $\hat{P}/\nu$, lejárat={expiry_str}", color=colors[i])
                ax[1][0].plot(
                    test_case_df["underlier_price"], test_case_df["model_price_deriv_AD"],
                    label=r"model $\hat{'P'}/\nu$" + f", lejárat={expiry_str}", color=colors[i], linestyle="--")
                ax[1][0].set_title("opcióár szerinti derivált", fontsize=14)
                ax[1][0].set_xlabel(r"$S_0/K$")
                ax[1][0].set_ylabel("opcióár szerinti derivált")
                ax[1][0].legend()

                ax[1][1].bar(
                    test_case_df["underlier_price"], test_case_df["modified_vega"] - test_case_df["model_price_deriv_AD"],
                    width=0.003, color=colors[i], alpha=0.3, label=f"opcióár szerinti derivált hiba, lejárat={expiry_str}")
                ax[1][1].legend()
                ylim1, ylim2 = min(ylim1, ax[1][1].get_ylim()[0]), max(ylim2, ax[1][1].get_ylim()[1])

                [x.set_ylim(ylim1, ylim2) for x in ax[:, 1].flatten()]
                # [x.hlines(y=0.0, xmin=xlim1, xmax=xlim2, color="black") for x in ax[:, 0].flatten()]
                [x.set_xlim(xlim1, xlim2) for x in ax[:, 0].flatten()]

            plt.tight_layout()
            plt.savefig(f"figures/{model_name}_slicing_plot.png")
            plt.close(fig)
            return None

        plt.tight_layout()
        plt.savefig(f"figures/{model_name}_slicing_plot.png")
        plt.close(fig)


if __name__ == "__main__":
    from config import pipeline_configs

    seed = 20250420

    for key, run_config in pipeline_configs.items():

        for neuron_num, layer_num, activ_func, greek_reg, lambda_param in [
            (64, 4, "tanh", True, 0.01),
        ]:
            run_config.train_data.seed = seed
            run_config.model.hidden_layer_activation = activ_func
            run_config.model.neuron_per_layer = neuron_num
            run_config.model.layer_number = layer_num
            run_config.model.calc_greek_regularization = greek_reg
            run_config.model.greek_weighting = False # greek_weighting
            run_config.model.lambda_param = lambda_param

            pipeline = Pipeline(run_config)
            pipeline.train(retrain=True)

            model_name = f"{run_config.pricing_model.__name__}_{run_config.model.neuron_per_layer}_{run_config.model.layer_number}_{run_config.model.hidden_layer_activation}{'_greek_reg' if run_config.model.calc_greek_regularization else ''}{'_weighted' if run_config.model.greek_weighting and False else ''}{'_lambda_' + str(run_config.model.lambda_param) if run_config.model.lambda_param else ''}"
            pipeline.evaluate(f"saved_models/{seed}/{model_name}.model")
