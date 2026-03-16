# src/models_deep.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE
from src.models_classical import build_missingness_indicator


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class MLPClassifierWrapper:
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        dropout=0.2,
        lr=1e-3,
        batch_size=32,
        epochs=20,
        device=None,
        random_state=RANDOM_STATE
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = SimpleMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(self.device)

        self.loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with Logits
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X_train, y_train):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)

        dataset = TensorDataset(
            torch.tensor(X_train),
            torch.tensor(y_train)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
                loss.backward()
                self.optimizer.step()

        return self

    def predict_proba(self, X):
        '''
        Predict class probabilities for the input samples.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns:
        probs : array-like of shape (n_samples, 2)
            Predicted class probabilities.
        '''
        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.tensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

        return np.column_stack([1 - probs, probs])

    def predict(self, X, threshold=0.5):
        '''
        Predict class labels for the input samples.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            Input samples.
        threshold : float, default=0.5
            Threshold for converting predicted probabilities to class labels.

        Returns:
        labels : array-like of shape (n_samples,)
            Predicted class labels.
        '''
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def get_params(self, deep=True):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "device": self.device,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def fit_predict_mlp(X_train, y_train, X_test):
    '''
    fit an MLP model and return predicted probabilities for the positive class.
    '''
    # Build missingness indicators
    X_train = build_missingness_indicator(X_train)
    X_test = build_missingness_indicator(X_test)

    # Impute nans and scale features
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    input_dim = X_train_scaled.shape[1]
    model = MLPClassifierWrapper(input_dim=input_dim)
    model.fit(X_train_scaled, y_train)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    return y_prob
