from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.linear_model import Ridge, LogisticRegression
import pandas as pd
import numpy as np

def fit_ensemble_model(X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, base_regrssion_models: list[tuple[str, RegressorMixin]] | None = None,
                       base_classification_models: list[tuple[str, ClassifierMixin]]| None = None, classification_threshold: float = 0.5, 
                       verbose: int = 0) -> RegressorMixin | ClassifierMixin | tuple[RegressorMixin, ClassifierMixin]:
    """
    Fits stacked ensemble models for regression, classification, or both.

    Uses a StackingRegressor with a Ridge meta-learner and/or a 
    StackingClassifier with a LogisticRegression meta-learner.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vectors, where `n_samples` is the number of samples and
        `n_features` is the number of features.
    y : array-like of shape (n_samples,)
        Target values. Note that for classification, y must be categorical 
        or binary, if using for both regression and classification it is 
        converted via y > classification_threshold.
    base_regrssion_models : list of (str, estimator) tuples, optional
        Base regression estimators to be stacked.
    base_classification_models : list of (str, estimator) tuples, optional
        Base classification estimators to be stacked.
    classification_threshold : float, default=0.5
        Threshold to convert regression targets to binary for classification.
    verbose : int, default=0
        Verbosity level passed to the stacking estimators.

    Returns
    -------
    reg_ensemble : StackingRegressor
        Returned if only `base_regrssion_models` is provided.
    class_ensemble : StackingClassifier
        Returned if only `base_classification_models` is provided.
    (reg_ensemble, class_ensemble) : tuple
        Returned if both model lists are provided.

    Raises
    ------
    ValueError
        If neither regression nor classification base models are provided.

    Notes
    -----
    The regression final estimator is Ridge(alpha=1.0). 
    The classification final estimator is LogisticRegression(max_iter=1000).
    """
    if not (np.issubdtype(y, np.integer) or np.issubdtype(y, np.bool_)):
        y_binary = (y > classification_threshold).astype(int)
    else:
        y_binary = y
    
    if not (base_regrssion_models or base_classification_models):
        raise ValueError("At least one of base_regrssion_models or base_classification_models must be provided.")
    
    if base_regrssion_models and base_classification_models:
        reg_ensemble = StackingRegressor(estimators=base_regrssion_models, final_estimator=Ridge(alpha=1.0), n_jobs=-1, verbose=verbose).fit(X[y > classification_threshold], y[y > classification_threshold])
        class_ensemble = StackingClassifier(estimators=base_classification_models, final_estimator=LogisticRegression(max_iter=1000), n_jobs=-1, verbose=verbose).fit(X, y_binary)
        return reg_ensemble, class_ensemble
    if base_regrssion_models:
        if verbose:
            print("Fitting regression ensemble...")
        reg_ensemble = StackingRegressor(estimators=base_regrssion_models, final_estimator=Ridge(alpha=1.0), n_jobs=-1, verbose=verbose).fit(X, y)
    if base_classification_models:
        if verbose:
            print("Fitting classification ensemble...")
        class_ensemble = StackingClassifier(estimators=base_classification_models, final_estimator=LogisticRegression(max_iter=1000), n_jobs=-1, verbose=verbose).fit(X, y_binary)
    elif base_regrssion_models:
        return reg_ensemble
    else:
        return class_ensemble
    
def classify_then_regress(X: pd.DataFrame | np.ndarray,
                        classification_model: ClassifierMixin, regression_model: RegressorMixin,
                        threshold: float = 0.5) -> np.ndarray:
    """
    Implements a two-stage Hurdle Model prediction logic.

    First predicts the probability of a positive outcome. If the probability 
    is below the threshold, the prediction is set to 0.0. Otherwise, the 
    regression model is used to estimate the value.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Samples to be predicted.
    classification_model : ClassifierMixin
        A trained classifier that supports `predict_proba`.
    regression_model : RegressorMixin
        A trained regressor used to estimate values for high-risk samples.
    threshold : float
        The probability threshold (between 0 and 1). Samples with 
        P(claim) < threshold are zeroed out.

    Returns
    -------
    final_preds : ndarray of shape (n_samples,)
        The combined predictions, where low-risk samples are 0.0 and 
        high-risk samples are the output of the regression model.
    """
    class_preds = classification_model.predict_proba(X)[:, 1]
    high_risk_mask = class_preds >= threshold
    low_risk_mask = ~high_risk_mask
    final_preds = np.zeros_like(class_preds)
    if np.any(high_risk_mask):
        final_preds[high_risk_mask] = regression_model.predict(X[high_risk_mask])
    if np.any(low_risk_mask):
        final_preds[low_risk_mask] = 0.0
    return final_preds