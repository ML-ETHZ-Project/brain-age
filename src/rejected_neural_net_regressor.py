"""Feedforward neural-network regressor, as an alternative to GradientBoostingRegressor.

A small multi-layer perceptron trained with Adam + dropout + weight decay.
Age (y) is standardized internally for training stability and inverse-
transformed back at predict time; this is invisible to callers (fit/predict
still take/return raw age values).
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_sizes=(64, 32), dropout=0.3, weight_decay=1e-3,
                 lr=1e-3, n_epochs=300, batch_size=32, lr_schedule=False, random_state=42):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr_schedule = lr_schedule
        self.random_state = random_state

    def _build_model(self, n_in):
        layers = []
        in_dim = n_in
        for h in self.hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(self.dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.y_mean_ = y.mean()
        self.y_std_ = y.std()
        y_scaled = (y - self.y_mean_) / self.y_std_

        self.model_ = self._build_model(X.shape[1])
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)
            if self.lr_schedule else None
        )

        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y_scaled))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )

        self.model_.train()
        for _ in range(self.n_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model_(xb)
                loss = nn.functional.mse_loss(pred, yb)
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        self.model_.eval()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            pred_scaled = self.model_(torch.from_numpy(X)).numpy().ravel()
        return pred_scaled * self.y_std_ + self.y_mean_


class EnsembleMLPRegressor(BaseEstimator, RegressorMixin):
    """Averages predictions from n_models TorchMLPRegressors, one per random seed.

    Same architecture/training hyperparameters for every member; only the
    weight initialization and minibatch shuffling differ. Averaging over
    seeds trades n_models x the training cost for lower prediction variance,
    which single from-scratch NNs are prone to at this dataset size.
    """

    def __init__(self, n_models=5, hidden_sizes=(64, 32), dropout=0.3, weight_decay=1e-3,
                 lr=1e-3, n_epochs=300, batch_size=32, lr_schedule=False, random_state=42):
        self.n_models = n_models
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr_schedule = lr_schedule
        self.random_state = random_state

    def fit(self, X, y):
        self.models_ = []
        for i in range(self.n_models):
            model = TorchMLPRegressor(
                hidden_sizes=self.hidden_sizes, dropout=self.dropout,
                weight_decay=self.weight_decay, lr=self.lr, n_epochs=self.n_epochs,
                batch_size=self.batch_size, lr_schedule=self.lr_schedule,
                random_state=self.random_state + i,
            )
            model.fit(X, y)
            self.models_.append(model)
        return self

    def predict(self, X):
        preds = np.stack([m.predict(X) for m in self.models_], axis=0)
        return preds.mean(axis=0)
