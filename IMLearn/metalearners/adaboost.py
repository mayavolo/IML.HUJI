import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        D = np.ones(y.shape[0]) / y.shape[0]
        self.D_ = [D]
        self.weights_ = np.zeros(self.iterations_)
        self.models_ = []
        for t in range(self.iterations_):
            # Find the best weak classifier
            best_stump = self.wl_()
            best_stump.fit(X, y * D)
            self.models_.append(best_stump)
            # Compute Error rate
            prediction = best_stump.predict(X)
            error = np.sum(D * (np.abs(prediction - y) / 2))
            # Assigning classifier weight
            self.weights_[t] = 0.5 * np.log((1.0 / error) - 1)
            # Update sample weight
            D = D * np.exp((-1) * y * self.weights_[t] * prediction)
            # Normalize sample weight
            D = D / np.sum(D)
            self.D_.append(D)

    def _predict(self, X):  # TODO GOOD :)
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        prediction = np.zeros(X.shape[0])
        for i in range(self.iterations_):
            prediction += (self.weights_[i] * self.models_[i].predict(X))
        return np.sign(prediction)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:  # TODO GOOD :)
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        labels = self._predict(X)
        return np.sum(np.not_equal(y, labels)) / y.shape[0]

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:  # TODO GOOD :)
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        prediction = np.zeros(X.shape[0])
        for t in range(T):
            prediction += self.weights_[t] * self.models_[t].predict(X)
        return np.sign(prediction)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        labels = self.partial_predict(X, T)
        return np.sum(np.not_equal(y, labels)) / y.shape[0]
