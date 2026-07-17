"""Denoising autoencoder (DAE) as a noise-robustness preprocessing step.

Unlike src/sparse_autoencoder.py (an unsupervised bottleneck meant to
*replace* feature selection, which lost badly to CorrelationPrunedKBest --
see README), this DAE keeps the original feature count and role: it sits
between VarianceThreshold and CorrelationPrunedKBest, denoising each row
before the existing supervised selection/model see it.

Trained a la Vincent et al.'s stacked denoising autoencoders: each training
batch is corrupted by zeroing a random fraction of its entries, and the
network has to reconstruct the ORIGINAL, uncorrupted batch from that
corrupted input. Forcing reconstruction from partial/corrupted information
(rather than just copying input to output, which a plain autoencoder can
trivially learn) pushes the bottleneck to model the correlations between
features -- which real anatomical measurements should have, and the
injected noise columns (per CLAUDE.md) shouldn't -- so cleaning through
that learned structure should suppress noise, not just replicate it.
At inference, clean (uncorrupted) input is passed through once and its
reconstruction -- the "denoised" version, same dimensionality as the
input -- is what gets passed downstream.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin


class _DAEModule(nn.Module):
    def __init__(self, n_features, hidden_size):
        super().__init__()
        self.encoder = nn.Linear(n_features, hidden_size)
        self.decoder = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        h = torch.relu(self.encoder(x))
        return self.decoder(h)


class DenoisingAutoencoderFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, hidden_size=256, corruption_frac=0.3, weight_decay=1e-5,
                 lr=1e-3, n_epochs=200, batch_size=64, random_state=42):
        self.hidden_size = hidden_size
        self.corruption_frac = corruption_frac
        self.weight_decay = weight_decay
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y=None):
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)

        self.model_ = _DAEModule(X.shape[1], self.hidden_size)
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        corruption_rng = torch.Generator().manual_seed(self.random_state)

        X_t = torch.from_numpy(X)
        dataset = torch.utils.data.TensorDataset(X_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )

        self.model_.train()
        for _ in range(self.n_epochs):
            for (batch,) in loader:
                keep_mask = (torch.rand(batch.shape, generator=corruption_rng)
                             > self.corruption_frac).float()
                corrupted = batch * keep_mask

                optimizer.zero_grad()
                recon = self.model_(corrupted)
                loss = nn.functional.mse_loss(recon, batch)  # reconstruct the CLEAN batch
                loss.backward()
                optimizer.step()

        self.model_.eval()
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            recon = self.model_(torch.from_numpy(X))
        return recon.numpy()
