import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class PricerDataset(torch.utils.data.Dataset):
    def __init__(self, df, X_cols, y_cols, Z_cols):
        self.X = torch.tensor(df[X_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[y_cols].values, dtype=torch.float32)
        if Z_cols:
            self.Z = torch.tensor(df[Z_cols].values, dtype=torch.float32)
        else:
            self.Z = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.Z is None:
            return self.X[idx], self.y[idx], torch.empty(0, dtype=torch.float32)
        return self.X[idx], self.y[idx], self.Z[idx]


class PricerNetTorch(nn.Module):
    def __init__(self, config):
        super(PricerNetTorch, self).__init__()
        self.config = config
        self.input_variables = config["input_variables"]
        self.target_variables = config["target_variables"]

        layers = []
        input_size = len(config['input_variables'])
        for _ in range(config['layer_number']):
            layers.append(nn.Linear(input_size, config['neuron_per_layer']))
            if config['activation_function'] == 'relu':
                layers.append(nn.ReLU())
            elif config['activation_function'] == 'tanh':
                layers.append(nn.Tanh())
            elif config['activation_function'] == 'leaky_relu':
                layers.append(nn.LeakyReLU(negative_slope=0.1))
            input_size = config['neuron_per_layer']
        layers.append(nn.Linear(input_size, len(config['target_variables'])))
        if config['activation_function'] != 'leaky_relu':
            layers.append(nn.Softplus(beta=1.0))
        self.model = nn.Sequential(*layers)
        self.model.apply(self._initialize_weights)


        # self.optimizer = optim.SGD(self.model.parameters(), lr=config['learning_rate'], momentum=0.7)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'],)
        self.loss_fn = nn.MSELoss()

        self.calc_greek_regularization = config['calc_greek_regularization']
        self.greek_weighting = config["greek_weighting"]
        self.lambda_param = config.get("lambda", 1.0)
        self.train_memory = {}
        self.device = config.get('device', 'cpu')

    def forward(self, x):
        return self.model(x)

    def train_model(self, df, epochs=10, batch_size=2**8, pd_metadata=None):
        self.model.train()
        for epoch in range(epochs):
            if epoch == 2 and False:
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
            max_norm = 1.0
            epoch_overall_losses = []
            epoch_losses = []
            epoch_pd_losses = []
            epoch_weighted_pd_weights = []
            epoch_weighting_factor = []
            epoch_grad_norms = []

            self.train_memory[epoch] = {
                "overall_loss": [],
                "price_loss": [],
            }
            for name in pd_metadata:
                self.train_memory[epoch][name + "_loss"] = []
                self.train_memory[epoch][name + "_weight"] = []

            data = PricerDataset(df.sample(10 ** 5),
                                 self.input_variables,
                                 self.target_variables,
                                 list(pd_metadata.keys())
            )
            data = DataLoader(data, batch_size=batch_size, shuffle=True)

            with tqdm(data, desc=f"Epoch {epoch + 1}/{epochs}", total=len(data)) as pbar:
                for X, y, Z in pbar:
                    self.optimizer.zero_grad()
                    output = self.forward(X)
                    X_loss = self.loss_fn(output, y)
                    X.requires_grad = True
                    output_grad = self.forward(X)
                    first_X_loss = self.loss_fn(output_grad, y)
                    self.train_memory[epoch]["price_loss"].append(first_X_loss.item())

                    if pd_metadata:
                        gradients = torch.autograd.grad(outputs=output_grad, inputs=X,
                                                        grad_outputs=torch.ones_like(output_grad),
                                                        create_graph=True)[0]
                    Z_loss = 0
                    pd_losses = {}
                    pd_weights = {}
                    weighted_pd_losses = {}
                    weighted_pd_weights = {}
                    for name, (ad_i, Z_i) in pd_metadata.items():
                        pd_loss = self.loss_fn(gradients[:, ad_i], Z[:, Z_i])
                        pd_losses[name] = pd_loss
                        self.train_memory[epoch][name + "_loss"].append(pd_loss.item())
                        self.train_memory[epoch][name + "_weight"].append(1.0)
                        pd_weights[name] = 1.0
                        Z_loss += pd_loss

                    if self.calc_greek_regularization:
                        if self.greek_weighting:
                            epoch_weighting_factor.append(self.lambda_param)

                            weighted_Z_loss = 0
                            for name, pd_loss in pd_losses.items():
                                weight = (pd_loss / Z_loss)
                                weighted_pd_loss = weight * pd_loss
                                weighted_pd_weights[name] = weight
                                self.train_memory[epoch][name + "_weight"][-1] = weight.item()
                                weighted_pd_losses[name] = weighted_pd_loss
                                weighted_Z_loss += weighted_pd_loss

                            epoch_weighted_pd_weights.append(
                                {name: weight.item() for name, weight in weighted_pd_weights.items()})

                            # weighted_Z_loss.backward()
                            loss = X_loss + self.lambda_param * weighted_Z_loss
                            loss.backward()

                            grads = [
                                param.grad.detach().flatten()
                                for param in self.model.parameters()
                                if param.grad is not None
                            ]
                            grad_norm = torch.cat(grads).norm()
                            epoch_grad_norms.append(grad_norm.item())

                        else:
                            # Z_loss.backward()
                            loss = X_loss + self.lambda_param * Z_loss
                            loss.backward()

                            grads = [
                                param.grad.detach().flatten()
                                for param in self.model.parameters()
                                if param.grad is not None
                            ]
                            grad_norm = torch.cat(grads).norm()
                            epoch_grad_norms.append(grad_norm.item())

                            epoch_weighting_factor.append(self.lambda_param)
                            epoch_weighted_pd_weights.append(
                                {name: weight for name, weight in pd_weights.items()})

                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                        self.optimizer.step()

                        overall_loss = X_loss + Z_loss
                        epoch_pd_losses.append(
                            {name: loss.item() for name, loss in pd_losses.items()})

                    else:
                        overall_loss = X_loss + Z_loss
                        X_loss.backward()

                        grads = [
                            param.grad.detach().flatten()
                            for param in self.model.parameters()
                            if param.grad is not None
                        ]
                        grad_norm = torch.cat(grads).norm()
                        epoch_grad_norms.append(grad_norm.item())

                        self.optimizer.step()

                        epoch_weighting_factor.append(0.0)
                        epoch_pd_losses.append(
                            {name: loss.item() for name, loss in pd_losses.items()})
                        epoch_weighted_pd_weights.append(
                            {name: weight for name, weight in pd_weights.items()})

                    self.train_memory[epoch]["overall_loss"].append(overall_loss.item())
                    epoch_overall_losses.append(overall_loss.item())
                    epoch_losses.append(X_loss.item())

                    pbar.set_postfix({
                        'L(x)': sum(epoch_overall_losses) / len(epoch_overall_losses),
                        'price_MSE': sum(epoch_losses) / len(epoch_losses),
                        **{f"{name}_MSE": sum([d[name] for d in epoch_pd_losses]) / len(
                            epoch_pd_losses) for name in pd_metadata.keys()},
                        **{f"{name}_weight": sum([d[name] for d in epoch_weighted_pd_weights]) / len(
                            epoch_weighted_pd_weights) for name in pd_metadata.keys()},
                        'gradient_norm': sum(epoch_grad_norms) / len(epoch_grad_norms),
                    })

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.normal_(module.bias)

    @staticmethod
    def get_lr(optimizer):
        return optimizer.param_groups[-1]['lr']

if __name__ == "__main__":
    import pandas as pd

    config = {
        'input_size': 6,
        'output_size': 1,
        'layer_number': 5,
        'neuron_per_layer': 32,
        'activation_function': 'tanh',
        'learning_rate': 0.1,
        'Z_loss_weight': 0.5
    }

    df = pd.read_parquet("data/20250328/BarrierUpAndInPutPDE_train.parquet")
    X_cols = ['underlier_price', 'expiry', 'volatility', 'interest_rate', 'barrier', 'strike']
    y_cols = ['price']
    Z_cols = ['delta', 'theta', 'vega', 'rho']

    train_dataset = PricerDataset(df.sample(10**7, random_state=0), X_cols, y_cols, Z_cols)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2 ** 12, shuffle=True)

    pricer_net = PricerNetTorch(config)
    pricer_net.train_model(train_dataloader,
                           epochs=10,
                           pd_metadata={'delta': (0, 0), 'theta': (1, 1), 'vega': (2, 2), 'rho': (3, 3)}
                           )
