# src/models_classical.py

# 1. Logistic Regression model


from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
from interpret import show

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.config import HIGH_MISSING_COLS, RANDOM_STATE


def build_missingness_indicator(X):
    ''''
    For each column in HIGH_MISSING_COLS, create a new binary column indicating whether the value is missing.
    This can help the model learn patterns related to missingness.
    '''
    X = X.copy()
    for col in HIGH_MISSING_COLS:
        X[col + '_missing'] = X[col].isna().astype(int)
    return X


def fit_predict_logistic_regression(X_train, y_train, X_test):
    '''
    Fit a logistic regression model and return predicted probabilities for the positive class.
    '''
    # Build missingness indicators
    X_train = build_missingness_indicator(X_train)
    X_test = build_missingness_indicator(X_test)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000,
         solver="liblinear", class_weight='balanced'))
    ])

    param_grid = {
        'model__C': [0.01, 0.1, 1, 10],
        'model__penalty': ['l1', 'l2']
    }

    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='roc_auc',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    y_prob = grid_search.predict_proba(X_test)[:, 1]
    best_params = grid_search.best_params_

    return y_prob, best_params


def fit_predict_xgboost(X_train, y_train, X_test):
    '''
    Fit an XGBoost model and return predicted probabilities for the positive class.
    '''
    model = XGBClassifier(
        random_state=RANDOM_STATE,
        objective='binary:logistic',
        eval_metric='auc',
        tree_method='hist',
        n_jobs=-1

    )

    param_grid = {
        # number of trees, in combination with learning_rate
        # (smaller learning_rate requires more trees)
        'n_estimators': [100, 300],
        # model complexity, higher depth can capture more complex patterns but may overfit
        'max_depth': [3, 5],
        # step size shrinkage to prevent overfitting
        # lower values make the model more robust but require more trees
        'learning_rate': [0.01, 0.1],
        # fraction of rows per tree, lower values can prevent overfitting but may underfit
        'subsample': [0.8, 1.0],
        # Fraction of features used per tree
        'colsample_bytree': [0.8, 1.0],
        # Minimum sum of instance weight needed in a child, higher values can prevent overfitting
        'min_child_weight': [1, 5]
    }

    inner_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='roc_auc',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    y_prob = grid_search.predict_proba(X_test)[:, 1]
    best_params = grid_search.best_params_

    return y_prob, best_params


def fit_predict_ebm(X_train, y_train, X_test):
    '''
    Fit an Explainable Boosting Machine (EBM) model.
    '''

    model = ExplainableBoostingClassifier(random_state=RANDOM_STATE)

    param_grid = {
        "max_leaves": [2, 3],
        "interactions": ["3x", "4x"],
        "learning_rate": [0.005, 0.01],
    }

    inner_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='roc_auc',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    y_prob = grid_search.predict_proba(X_test)[:, 1]
    best_params = grid_search.best_params_

    return y_prob, best_params
