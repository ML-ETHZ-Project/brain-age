"""Unsupervised feature extraction via a sparse autoencoder's bottleneck.

Unlike CorrelationPrunedKBest (src/feature_selection.py), which ranks and
prunes the original 832 features by their correlation with age, this learns
a new, smaller feature set by training a single-hidden-layer autoencoder to
reconstruct the (imputed, scaled) input -- entirely unsupervised, no y used.
An L1 penalty on the hidden-unit activations encourages a sparse code (most
hidden units near zero for a given row), rather than every latent dimension
carrying dense, entangled signal.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin


class _SparseAutoencoderModule(nn.Module):
    def __init__(self, n_in, n_hidden):
        super().__init__()
        self.encoder = nn.Linear(n_in, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_in)

    def forward(self, x):
        h = torch.relu(self.encoder(x))
        x_hat = self.decoder(h)
        return x_hat, h


class SparseAutoencoderFeatures(BaseEstimator, TransformerMixin):
    """Trains a sparse autoencoder on X, transforms X into its hidden layer.

    Loss = reconstruction MSE + sparsity_weight * mean(|hidden activation|),
    optimized with Adam (weight_decay for L2 on the network weights).
    """

    def __init__(self, n_hidden=64, sparsity_weight=1e-3, weight_decay=1e-5,
                 lr=1e-3, n_epochs=200, batch_size=64, random_state=42):
        self.n_hidden = n_hidden
        self.sparsity_weight = sparsity_weight
        self.weight_decay = weight_decay
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        torch.manual_seed(self.random_state)

        self.model_ = _SparseAutoencoderModule(X.shape[1], self.n_hidden)
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )

        X_t = torch.from_numpy(X)
        dataset = torch.utils.data.TensorDataset(X_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )

        self.model_.train()
        for _ in range(self.n_epochs):
            for (batch,) in loader:
                optimizer.zero_grad()
                x_hat, h = self.model_(batch)
                recon_loss = nn.functional.mse_loss(x_hat, batch)
                sparsity_loss = h.abs().mean()
                loss = recon_loss + self.sparsity_weight * sparsity_loss
                loss.backward()
                optimizer.step()

        self.model_.eval()
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            _, h = self.model_(torch.from_numpy(X))
        return h.numpy()
