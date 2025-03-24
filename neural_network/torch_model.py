import torch
import torch.nn as nn
import torch.optim as optim
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

        layers = []
        input_size = config['input_size']
        for _ in range(config['layer_number']):
            layers.append(nn.Linear(input_size, config['neuron_per_layer']))
            if config['activation_function'] == 'relu':
                layers.append(nn.ReLU())
            elif config['activation_function'] == 'tanh':
                layers.append(nn.Tanh())
            elif config['activation_function'] == 'leaky_relu':
                layers.append(nn.LeakyReLU(negative_slope=0.1))
            input_size = config['neuron_per_layer']
        layers.append(nn.Linear(input_size, config['output_size']))
        layers.append(nn.Softplus(beta=2.0))
        self.model = nn.Sequential(*layers)
        self.model.apply(self._initialize_weights)

        # self.optimizer = optim.SGD(self.model.parameters(), lr=config['learning_rate'], momentum=0.75)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'] / 100)
        self.loss_fn = nn.MSELoss()

        self.Z_loss_weight = config["layer_number"] ** 2 # config["Z_loss_weight"]

    def forward(self, x):
        return self.model(x)

    def train_model(self, data, epochs=10, pd_metadata=None):
        self.model.train()
        lr = self.config['learning_rate']
        for epoch in range(epochs):
            # self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.75)
            max_norm = 0.25

            epoch_overall_losses = []
            epoch_losses = []
            epoch_weighted_pd_losses = []
            epoch_weighted_pd_weights = []
            with tqdm(data, desc=f"Epoch {epoch + 1}/{epochs}", total=len(data)) as pbar:
                for X, y, Z in pbar:
                    if Z.numel() > 0:
                        if epoch == 0:
                            self.optimizer.zero_grad()
                            output = self.forward(X)
                            X_loss = self.loss_fn(output, y)
                            self.optimizer.zero_grad()
                            X_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                            self.optimizer.step()

                            X.requires_grad = True
                            output = self.forward(X)
                            gradients = torch.autograd.grad(outputs=output, inputs=X,
                                                            grad_outputs=torch.ones_like(output),
                                                            create_graph=True)[0]
                            Z_loss = 0
                            pd_losses = {}
                            weighted_pd_losses = {}
                            weighted_pd_weights = {}
                            for name, (ad_i, Z_i) in pd_metadata.items():
                                pd_loss = self.loss_fn(gradients[:, ad_i], Z[:, Z_i])
                                pd_losses[name] = pd_loss
                                Z_loss += pd_loss

                            weighted_Z_loss = 0
                            for name, pd_loss in pd_losses.items():
                                weight = (pd_loss / Z_loss)
                                weighted_pd_loss = weight * pd_loss
                                weighted_pd_weights[name] = weight
                                weighted_pd_losses[name] = weighted_pd_loss
                                weighted_Z_loss += self.Z_loss_weight * weighted_pd_loss

                            overall_loss = X_loss + weighted_Z_loss / self.Z_loss_weight

                        else:
                            self.optimizer.zero_grad()
                            X.requires_grad = True
                            output = self.forward(X)
                            gradients = torch.autograd.grad(outputs=output, inputs=X,
                                                            grad_outputs=torch.ones_like(output),
                                                            create_graph=True)[0]
                            Z_loss = 0
                            pd_losses = {}
                            weighted_pd_losses = {}
                            weighted_pd_weights = {}
                            for name, (ad_i, Z_i) in pd_metadata.items():
                                pd_loss = self.loss_fn(gradients[:, ad_i], Z[:, Z_i])
                                pd_losses[name] = pd_loss
                                Z_loss += pd_loss

                            weighted_Z_loss = 0
                            for name, pd_loss in pd_losses.items():
                                weight = (pd_loss / Z_loss)
                                weighted_pd_loss = weight * pd_loss
                                weighted_pd_weights[name] = weight
                                weighted_pd_losses[name] = weighted_pd_loss
                                weighted_Z_loss += self.Z_loss_weight * weighted_pd_loss

                            weighted_Z_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                            self.optimizer.step()

                            self.optimizer.zero_grad()
                            output = self.forward(X)
                            X_loss = self.loss_fn(output, y)
                            X_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                            self.optimizer.step()

                            overall_loss = X_loss + weighted_Z_loss / self.Z_loss_weight

                        epoch_overall_losses.append(overall_loss.item())
                        epoch_losses.append(X_loss.item())
                        epoch_weighted_pd_losses.append(
                            {name: loss.item() for name, loss in weighted_pd_losses.items()})
                        epoch_weighted_pd_weights.append(
                            {name: weight.item() for name, weight in weighted_pd_weights.items()})
                        pbar.set_postfix({
                            'overall_loss': sum(epoch_overall_losses) / len(epoch_overall_losses),
                            'price_loss': sum(epoch_losses) / len(epoch_losses),
                            **{f"{name}_loss": sum([d[name] for d in epoch_weighted_pd_losses]) / len(
                                epoch_weighted_pd_losses) for name in pd_metadata.keys()},
                            **{f"{name}_weight": sum([d[name] for d in epoch_weighted_pd_weights]) / len(
                                epoch_weighted_pd_weights) for name in pd_metadata.keys()}
                        })
                    else:
                        self.optimizer.zero_grad()
                        output = self.forward(X)
                        loss = self.loss_fn(output, y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                        self.optimizer.step()
                        epoch_losses.append(loss.item())
                        pbar.set_postfix({
                            'price_loss': sum(epoch_losses) / len(epoch_losses),
                        })
                    
    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

if __name__ == "__main__":
    import pandas as pd

    config = {
        'input_size': 4,
        'output_size': 1,
        'layer_number': 5,
        'neuron_per_layer': 32,
        'activation_function': 'tanh',
        'learning_rate': 0.03,
        'Z_loss_weight': 0.5
    }

    df = pd.read_parquet("data/20250317/BlackScholesCall_train.parquet")
    X_cols = ['underlier_price', 'expiry', 'volatility', 'interest_rate']
    y_cols = ['price']
    Z_cols = ['delta', 'theta', 'vega', 'rho']

    train_dataset = PricerDataset(df.sample(10**6, random_state=0), X_cols, y_cols, Z_cols)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2 ** 11, shuffle=True)

    pricer_net = PricerNetTorch(config)
    pricer_net.train_model(train_dataloader,
                           epochs=10,
                           pd_metadata={'delta': (0, 0), 'theta': (1, 1), 'vega': (2, 2), 'rho': (3, 3)}
                           )
