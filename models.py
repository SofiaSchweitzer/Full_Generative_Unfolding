import torch
import torch.nn as nn
import time
from torchdiffeq import odeint

class Regressor(nn.Module):
    def __init__(self, dims_in, params):
        super().__init__()
        self.dims_in = dims_in
        self.params = params
        self.init_network()

    def init_network(self):
        layers = []
        layers.append(nn.Linear(self.dims_in, self.params["internal_size"]))
        layers.append(nn.ReLU())
        for _ in range(self.params["hidden_layers"]):
            layers.append(nn.Linear(self.params["internal_size"], self.params["internal_size"]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.params["internal_size"], 1))
        self.network = nn.Sequential(*layers)

    def batch_loss(self, x, y):
        pred = self.network(x).squeeze()
        loss = torch.nn.MSELoss()(pred, y)
        return loss

    def train_regressor(self, data, value):
        dataset= torch.utils.data.TensorDataset(data, value)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.params["batch_size"],
                                             shuffle=True)
        n_epochs = self.params["n_epochs"]
        lr = self.params["lr"]
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        print(f"Training regressor for {n_epochs} epochs with lr {lr}")
        t0 = time.time()
        for epoch in range(n_epochs):
            losses = []
            for i, batch in enumerate(loader):
                x, y = batch
                optimizer.zero_grad()
                loss = self.batch_loss(x,y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if epoch % int(n_epochs / 5) == 0:
                print(
                    f"    Finished epoch {epoch} with average loss {torch.tensor(losses).mean()} after time {round(time.time() - t0, 1)}")
        print(
            f"    Finished epoch {epoch} with average loss {torch.tensor(losses).mean()} after time {round(time.time() - t0, 1)}")

    def evaluate(self, data):
        predictions = []
        with torch.no_grad():
            for batch in torch.split(data, self.params["batch_size_sample"]):
                pred = self.network(batch).squeeze().detach()
                predictions.append(pred)
        predictions = torch.cat(predictions)
        return predictions

class Classifier(nn.Module):
    def __init__(self, dims_in, params):
        super().__init__()
        self.dims_in = dims_in
        self.params = params
        self.init_network()

    def init_network(self):
        layers = []
        layers.append(nn.Linear(self.dims_in, self.params["internal_size"]))
        layers.append(nn.ReLU())
        for _ in range(self.params["hidden_layers"]):
            layers.append(nn.Linear(self.params["internal_size"], self.params["internal_size"]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.params["internal_size"], 1))
        self.network = nn.Sequential(*layers)

    def batch_loss(self, x, y, w):
        pred = self.network(x).squeeze()
        loss = torch.nn.BCEWithLogitsLoss(weight=w)(pred, y)
        return loss

    def train_classifier(self, data_true, data_false, weights_true=None, weights_false=None, balanced=True):
        if weights_true is None:
            weights_true = torch.ones((data_true.shape[0])).to(data_true.device, dtype=data_true.dtype)
        if weights_false is None:
            weights_false = torch.ones((data_false.shape[0])).to(data_true.device, dtype=data_true.dtype)

        dataset_true = torch.utils.data.TensorDataset(data_true, weights_true)
        loader_true = torch.utils.data.DataLoader(dataset_true, batch_size=self.params["batch_size"],
                                             shuffle=True)
        dataset_false = torch.utils.data.TensorDataset(data_false, weights_false)
        loader_false = torch.utils.data.DataLoader(dataset_false, batch_size=self.params["batch_size"],
                                                  shuffle=True)

        if not balanced:
            class_weight = len(data_true)/len(data_false)
            print(f"    Training with unbalanced training set with weight {class_weight}")
        else:
            class_weight = 1
        n_epochs = self.params["n_epochs"]* int(class_weight)
        lr = self.params["lr"]
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        print(f"Training classifier for {n_epochs} epochs with lr {lr}")
        t0 = time.time()
        for epoch in range(n_epochs):
            losses = []
            for i, (batch_true, batch_false) in enumerate(zip(loader_true, loader_false)):
                x_true, weight_true = batch_true
                x_false, weight_false = batch_false
                label_true = torch.ones((x_true.shape[0])).to(x_true.device)
                label_false = torch.zeros((x_false.shape[0])).to(x_false.device)
                optimizer.zero_grad()
                loss = self.batch_loss(x_true, label_true, weight_true)* class_weight
                loss += self.batch_loss(x_false, label_false, weight_false)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if epoch % int(n_epochs / 5) == 0:
                print(f"    Finished epoch {epoch} with average loss {torch.tensor(losses).mean()} after time {round(time.time() - t0, 1)}")
        print(f"    Finished epoch {epoch} with average loss {torch.tensor(losses).mean()} after time {round(time.time() - t0, 1)}")

    def evaluate(self, data, return_weights=True):
        predictions = []
        with torch.no_grad():
            for batch in torch.split(data, self.params["batch_size_sample"]):
                pred = self.network(batch).squeeze().detach()
                predictions.append(pred)
        predictions = torch.cat(predictions)
        return predictions.exp().clip(0, 30) if return_weights else torch.sigmoid(predictions)


class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def init_network(self):
        layers = []
        layers.append(nn.Linear(self.dims_in, self.params["internal_size"]))
        layers.append(nn.ReLU())
        for _ in range(self.params["hidden_layers"]):
            layers.append(nn.Linear(self.params["internal_size"], self.params["internal_size"]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.params["internal_size"], self.dims_x))
        self.network = nn.Sequential(*layers)

    def train_unfolder(self, data_x, data_c, weights=None):
        if weights is None:
            weights = torch.ones((data_x.shape[0]))
        dataset = torch.utils.data.TensorDataset(data_x, data_c, weights)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.params["batch_size"],
                                             shuffle=True)
        n_epochs = self.params["n_epochs"]
        lr = self.params["lr"]
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(loader) * n_epochs)
        print(f"Training CFM for {n_epochs} epochs with lr {lr}")
        t0 = time.time()
        for epoch in range(n_epochs):
            losses = []
            for i, batch in enumerate(loader):
                x_hard, x_reco, weight = batch
                optimizer.zero_grad()
                loss = self.batch_loss(x_hard, x_reco, weight)
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
            if epoch % int(n_epochs / 5) == 0:
                print(
                    f"    Finished epoch {epoch} with average loss {torch.tensor(losses).mean()} after time {round(time.time() - t0, 1)}")
        print(
            f"    Finished epoch {epoch} with average loss {torch.tensor(losses).mean()} after time {round(time.time() - t0, 1)}")

    def evaluate(self, data_c):
        predictions = []
        with torch.no_grad():
            for batch in torch.split(data_c, self.params["batch_size_sample"]):
                unfold_cfm = self.sample(batch).detach()
                predictions.append(unfold_cfm)
        predictions = torch.cat(predictions)
        return predictions


class CFM(Model):
    def __init__(self, dims_x, dims_c, params):
        super().__init__()
        self.dims_x = dims_x
        self.dims_c = dims_c
        self.params = params
        self.dims_in = self.dims_x + self.dims_c + 1
        self.init_network()

    def sample(self, c):
        batch_size = c.size(0)
        dtype = c.dtype
        device = c.device

        def net_wrapper(t, x_t):
            t = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device)
            v = self.network(torch.cat([t, x_t, c], dim=-1))
            return v

        x_0 = torch.randn((batch_size, self.dims_x)).to(device, dtype=dtype)
        x_t = odeint(func=net_wrapper, y0=x_0, t=torch.tensor([0., 1.]).to(device, dtype=dtype))
        return x_t[-1]

    def batch_loss(self, x, c, weight):
        x_0 = torch.randn((x.size(0), self.dims_x)).to(x.device)
        t = torch.rand((x.size(0), 1)).to(x.device)
        x_t = (1 - t) * x_0 + t * x
        x_t_dot = x - x_0
        v_pred = self.network(torch.cat([t, x_t, c], dim=-1))
        cfm_loss = ((v_pred - x_t_dot) ** 2 * weight.unsqueeze(-1)).mean()
        return cfm_loss
