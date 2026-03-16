# src/models_deep.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from tabicl import TabICLClassifier

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


class MLPClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        dropout=0.2,
        lr=1e-4,  # reduced learning rate for better convergence
        batch_size=32,
        epochs=10,  # reduced epochs to prevent overfitting
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
        self.device = device

    def _build_model(self):
        self.device_ = self.device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = SimpleMLP(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        ).to(self.device_)

        self.loss_fn_ = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with Logits
        self.optimizer_ = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr)

    def fit(self, X_train, y_train):
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        np.random.seed(self.random_state)

        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)

        self._build_model()

        dataset = TensorDataset(
            torch.tensor(X_train),
            torch.tensor(y_train)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for _ in range(self.epochs):
            for batch_idx, (xb, yb) in enumerate(loader):
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)

                # Check for NaN values in the batch during training
                if torch.isnan(xb).any():
                    raise ValueError(
                        f"NaN found in xb at epoch {_}, batch {batch_idx} during MLP training.")
                if torch.isnan(yb).any():
                    raise ValueError(
                        f"NaN found in yb at epoch {_}, batch {batch_idx} during MLP training.")

                self.optimizer_.zero_grad()
                logits = self.model_(xb)
                loss = self.loss_fn_(logits, yb)
                loss.backward()
                # add gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model_.parameters(), max_norm=1.0)
                self.optimizer_.step()
        self.classes_ = np.array([0, 1])
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
        X_tensor = torch.tensor(X).to(self.device_)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        '''
        Predict class labels for the input samples.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns:
        labels : array-like of shape (n_samples,)
            Predicted class labels.
        '''
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

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

    # # Impute nans and scale features
    # imputer = SimpleImputer(strategy="median")
    # scaler = StandardScaler()

    # X_train_imp = imputer.fit_transform(X_train)
    # X_test_imp = imputer.transform(X_test)

    # X_train_scaled = scaler.fit_transform(X_train_imp)
    # X_test_scaled = scaler.transform(X_test_imp)

    # assert np.isfinite(X_train_scaled).all(
    # ), "X_train_scaled contains non-finite values"
    # assert np.isfinite(X_test_scaled).all(
    # ), "X_test_scaled contains non-finite values"

    input_dim = X_train.shape[1]
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", MLPClassifierWrapper(input_dim=input_dim))
    ])

    param_grid = {
        'model__hidden_dim': [32, 64],
        'model__lr': [1e-4, 3e-4],
        'model__dropout': [0.1, 0.2]
    }

    inner_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='roc_auc',
        refit=True,
        n_jobs=1
    )

    # # Debug
    # print("Before model.fit")
    # print("X_train_scaled NaNs:", np.isnan(X_train_scaled).sum())
    # print("X_test_scaled NaNs:", np.isnan(X_test_scaled).sum())
    # print("X_train_scaled shape:", X_train_scaled.shape)
    # print("About to pass into model.fit:", "X_train_scaled")

    grid_search.fit(X_train, y_train)
    y_prob = grid_search.predict_proba(X_test)[:, 1]
    best_params = grid_search.best_params_
    return y_prob, best_params


def fit_predict_tabicl(X_train, y_train, X_test):
    '''
    fit a TabICL model and return predicted probabilities for the positive class.
    '''
    model = TabICLClassifier()
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    return y_prob
