import torch
import torch.nn as nn
import time
from torchdiffeq import odeint
import math
import normflows as nf
from pytorch_optimizer import Lion
from torch.utils.data import DataLoader, TensorDataset, random_split

import logging

import os
class Regressor(nn.Module):
    def __init__(self, dims_in, params):
        super().__init__()
        self.dims_in = dims_in
        self.params = params
        self.init_network()

    def init_network(self):
        layers = []
        layers.append(nn.Linear(self.dims_in, self.params["internal_size"]))
        layers.append(nn.SiLU())
        for _ in range(self.params["hidden_layers"]):
            layers.append(nn.Linear(self.params["internal_size"], self.params["internal_size"]))
            layers.append(nn.SiLU())
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
        logging.info(f"Training regressor for {n_epochs} epochs with lr {lr}")
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
                logging.info(
                    f"    Finished epoch {epoch} with average loss {torch.tensor(losses).mean()} after time {round(time.time() - t0, 1)}")
        logging.info(
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
    def __init__(self, dims_in, params, model_name, logger=None):
        super().__init__()
        self.dims_in = dims_in
        self.params = params
        self.init_network()
        self.logger = logger
        self.model_name = model_name
    def init_network(self):
        layers = []
        layers.append(nn.Linear(self.dims_in, self.params["internal_size"]))
        layers.append(nn.GELU())
        for _ in range(self.params["hidden_layers"]):
            layers.append(nn.Linear(self.params["internal_size"], self.params["internal_size"]))
            layers.append(nn.GELU())
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
        self.network = self.network.to(data_true.device)

        dataset_true = torch.utils.data.TensorDataset(data_true, weights_true)
        loader_true = torch.utils.data.DataLoader(dataset_true, batch_size=self.params["batch_size"],
                                             shuffle=True)
        dataset_false = torch.utils.data.TensorDataset(data_false, weights_false)
        loader_false = torch.utils.data.DataLoader(dataset_false, batch_size=self.params["batch_size"],
                                                  shuffle=True)

        if not balanced:
            class_weight = len(data_true)/len(data_false)
            logging.info(f"    Training with unbalanced training set with weight {class_weight}")
        else:
            class_weight = 1
        n_epochs = self.params["n_epochs"]* int(class_weight)
        lr = self.params["lr"]
        #optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        optimizer = Lion(self.network.parameters(), lr=lr)
        n_batches = min(len(loader_true), len(loader_false))
        logging.info(f"Training classifier for {n_epochs} epochs with lr {lr}")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.params.get("max_lr", 3 * self.params["lr"]),
            epochs=n_epochs,
            steps_per_epoch=n_batches)

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
                scheduler.step()
                losses.append(loss.item())
                if self.logger is not None:
                    self.logger.add_scalar(f"{self.model_name}/train_losses", losses[-1], epoch * len(loader_true) + i)
                    self.logger.add_scalar(f"{self.model_name}/learning_rate", scheduler.get_last_lr()[0],
                                           len(loader_true) * epoch + i)
            if epoch % int(n_epochs / 5) == 0:
                logging.info(f"    Finished epoch {epoch} with average loss {torch.tensor(losses).mean()} after time {round(time.time() - t0, 1)}")

            if self.logger is not None:
                self.logger.add_scalar(f"{self.model_name}/train_losses_epoch", torch.tensor(losses).mean(), epoch)
        logging.info(f"    Finished epoch {epoch} with average loss {torch.tensor(losses).mean()} after time {round(time.time() - t0, 1)}")

    def evaluate(self, data, return_weights=True):
        predictions = []
        with torch.no_grad():
            self.network = self.network.to(data.device)
            for batch in torch.split(data, self.params["batch_size_sample"]):
                pred = self.network(batch).squeeze().detach()
                predictions.append(pred)
        predictions = torch.cat(predictions)
        return predictions.exp().clip(0, 30) if return_weights else torch.sigmoid(predictions)

    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.epochs = 0
        self.patience = 30
        self.test_frac = 0.2
    def init_network(self):
        layers = []
    
        # Input layer
        layers.append(nn.Linear(self.dims_in, self.params["internal_size"]))
        layers.append(nn.GELU())        
        for _ in range(self.params["hidden_layers"]):
            layers.append(nn.Linear(self.params["internal_size"], self.params["internal_size"]))
            layers.append(nn.GELU())        

        # Output layer
        layers.append(nn.Linear(self.params["internal_size"], self.dims_x))

        self.network = nn.Sequential(*layers)
    def train(self, data_x, data_c=None, weights=None, test_frac = 0.2):
        if weights is None:
            weights = torch.ones((data_x.shape[0]))
        if data_c is not None:
            print("using conditional model")
            dataset = torch.utils.data.TensorDataset(data_x, weights, data_c)
        else:
            dataset = torch.utils.data.TensorDataset(data_x, weights)


        total = len(dataset)
        test_len = int(total * self.test_frac)
        train_len = total - test_len
        train_set, test_set = random_split(dataset, [train_len, test_len])
        # DataLoaders
        batch_size = self.params.get("batch_size", 32)
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.params.get("num_workers", 0),
            pin_memory=self.params.get("pin_memory", False),
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        self.network = self.network.to(data_x.device)        
        n_epochs = self.params["n_epochs"]
        lr = self.params["lr"]
        #optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        optimizer = Lion(self.network.parameters(), lr=lr, wd = 0.01)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * n_epochs)
        logging.info(f"Training CFM for {n_epochs} epochs with lr {lr}")

        best_loss = float('inf')
        epochs_no_improve = 0
        best_model_wts = {k: v.clone() for k, v in self.network.state_dict().items()}
                
        t0 = time.time()                
        for epoch in range(n_epochs):
            self.network.train()
            train_losses = []
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                loss = self.batch_loss(*batch)
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_losses.append(loss.item())

            self.network.eval()
            test_losses = []
            with torch.no_grad():
                for batch in test_loader:
                    loss = self.batch_loss(*batch)
                    test_losses.append(loss.item())

            avg_test_loss = sum(test_losses) / len(test_losses)
            logging.info(f"Epoch {epoch + 1}: train_loss={sum(train_losses)/len(train_losses):.4f}, test_loss={avg_test_loss:.4f}")

            # Early stopping check
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                epochs_no_improve = 0
                best_model_wts = {k: v.clone() for k, v in self.network.state_dict().items()}

            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logging.info(f"No improvement for {self.patience} epochs. Stopping early at epoch {epoch + 1}.")
                    break

            self.epochs += 1
            
        self.network.load_state_dict(best_model_wts)                    
        logging.info(f"Training complete. Total epochs run: {self.epochs}. Elapsed time: {time.time() - t0:.1f}s")
                
    def evaluate(self, data_c):
        predictions = []
        with torch.no_grad():
            for batch in torch.split(data_c, self.params["batch_size_sample"]):
                unfold_cfm = self.sample(batch).detach()
                predictions.append(unfold_cfm)
        predictions = torch.cat(predictions)
        return predictions


    
class CFM(Model):
    def __init__(self, dims_x, dims_c, params, logger = None):
        super().__init__()
        self.dims_x = dims_x
        self.dims_c = dims_c
        self.params = params
        self.dims_in = self.dims_x + self.dims_c + 1
        self.init_network()
        self.logger = logger
        
    def sample(self, c = None, num_evts = 0, device = None, dtype = None):
        if c is not None:
            batch_size = c.size(0)            
            self.network = self.network.to(c.device)
            device = device if device is not None else c.device
            dtype = dtype if dtype is not None else c.dtype
        else:
            batch_size = num_evts

        def net_wrapper(t, x_t):
            t = t * torch.ones_like(x_t[:, [0]], device=device, dtype=dtype)
            if c is not None:
                x_t = torch.cat([x_t,c],-1)
            v = self.network(torch.cat([t, x_t],-1))
            return v

        x_0 = torch.randn((batch_size, self.dims_x)).to(device, dtype=dtype)
        x_t = odeint(func=net_wrapper, y0=x_0, t=torch.tensor([0., 1.]).to(device, dtype=dtype))
        return x_t[-1]

    def batch_loss(self, x, weight, c=None):
        x_0 = torch.randn((x.size(0), self.dims_x)).to(x.device)
        t = torch.rand((x.size(0), 1)).to(x.device)
        x_t = (1 - t) * x_0 + t * x
        x_t_dot = x - x_0
        if c is not None:
            x_t = torch.cat([x_t,c],-1)
        
        v_pred = self.network(torch.cat([t, x_t],-1))
        cfm_loss = ((v_pred - x_t_dot) ** 2 * weight.unsqueeze(-1)).mean()
        return cfm_loss

    def evaluate(self, data_c=None, num_evts = 0, device = None, dtype = None):
        predictions = []
        with torch.no_grad():
            if data_c is not None:
                for batch in torch.split(data_c, self.params["batch_size_sample"]):
                    predictions.append(self.sample(batch).detach())
            else:
                num_batches = num_evts//self.params["batch_size_sample"]
                for _ in range(num_batches):
                    predictions.append(self.sample(
                        num_evts = self.params["batch_size_sample"],
                        device = device, dtype = dtype).detach())
                #Last batch
                residual = num_evts % self.params["batch_size_sample"]
                if residual > 0:
                    predictions.append(self.sample(num_evts = residual,device = device, dtype = dtype).detach())
        predictions = torch.cat(predictions)
        return predictions

