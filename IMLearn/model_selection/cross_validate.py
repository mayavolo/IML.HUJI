from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_score = 0
    validation_score = 0
    division = np.remainder(np.arange(X.size), cv)
    for k in range(cv):
        train_k_x = X[division != k].to_numpy().ravel()
        train_k_y = y[division != k].to_numpy().ravel()
        val_k_x = X[division == k].to_numpy().ravel()
        val_k_y = y[division == k].to_numpy().ravel()

        estimator.fit(train_k_x, train_k_y)
        loss_train_d = scoring(estimator.predict(train_k_x), train_k_y, 0)
        loss_validation_d = scoring(estimator.predict(val_k_x), val_k_y, 0)

        train_score += np.array(loss_train_d) / cv
        validation_score += np.array(loss_validation_d) / cv

    return train_score, validation_score
