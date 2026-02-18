from __future__ import annotations

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def get_models(random_state: int = 42) -> dict:
    """
    Returns a registry of model name -> sklearn estimator.
    Keep this as the single source of truth for which models you compare.
    """
    return {
        "linear_regression": LinearRegression(),

        # Regularized linear models
        "ridge_a10": Ridge(alpha=10.0, random_state=random_state),
        "lasso_a0p01": Lasso(alpha=0.01, random_state=random_state, max_iter=50000),

        # Tree-based models
        "decision_tree": DecisionTreeRegressor(
            random_state=random_state,
            max_depth=20,
            min_samples_leaf=10
        ),

        "random_forest_50": RandomForestRegressor(
            n_estimators=50,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=10,
            max_depth=30
        ),
        "gbr": GradientBoostingRegressor(random_state=random_state),
    }
