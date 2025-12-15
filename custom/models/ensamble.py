from sklearn.linear_model import Ridge
from sklearn.exceptions import NotFittedError
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from typing import Self
import pandas as pd
import numpy as np

class EnsambleRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, models: list[RegressorMixin], alpha: float = 1.0, random_state: int | None = None) -> None:
        self.models = models
        self.alpha = alpha
        self.random_state = random_state

    def _get_meta_features(self, X:  pd.DataFrame | np.ndarray) -> np.ndarray:
        preds = [model.predict(X).ravel() for model in self.models]
        return np.column_stack(preds)

    def fit(self, X:  pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> Self:
        X, y = check_X_y(X, y)

        for i, model in enumerate(self.models):
            try:
                check_is_fitted(model)
            except NotFittedError:
                raise NotFittedError(
                    f"Model {model.__class__.__name__} at index {i} is not fitted. "
                    "EnsembleRegressor expects pre-fitted base models."
                )

        meta_features = self._get_meta_features(X)

        self.meta_learner_ = Ridge(alpha=self.alpha, random_state=self.random_state).fit(meta_features, y)
        
        return self
    
    def predict(self, X:  pd.DataFrame | np.ndarray) -> np.ndarray:
        X = check_array(X)

        meta_features = self._get_meta_features(X)

        return self.meta_learner_.predict(meta_features)
