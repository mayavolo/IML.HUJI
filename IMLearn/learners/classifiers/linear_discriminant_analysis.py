from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from ...metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.cov_ = np.zeros((X.shape[1], X.shape[1]))
        self.mu_ = np.zeros((self.classes_.shape[0], X.shape[1]))
        for i in range(self.classes_.shape[0]):
            indices = (y == self.classes_[i]).nonzero()
            mean = np.mean(X[indices], axis=0)
            self.mu_[i] = mean

        const = 1 / (X.shape[0] - self.classes_.shape[0])
        for i in range(self.classes_.shape[0]):
            indices = (y == self.classes_[i]).nonzero()
            for sample in X[indices]:
                self.cov_ += np.matmul(np.reshape(sample - self.mu_[i], (X.shape[1], 1)),
                                       np.transpose(np.reshape(sample - self.mu_[i], (X.shape[1], 1))))
        self.cov_ *= const

        self._cov_inv = np.linalg.inv(self.cov_)
        # raise NotImplementedError()

    def _predict(self, X: np.ndarray) -> np.ndarray:
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
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            max_p = -np.inf
            max_k = -1
            for c in range(self.classes_.shape[0]):
                a_k = np.matmul(self._cov_inv, self.mu_[c].transpose())
                b_k = -0.5 * (self.mu_[c] @ self._cov_inv @ self.mu_[c].transpose())
                res = np.matmul(a_k.transpose(), X[i]) + b_k
                if res > max_p:
                    max_p = res
                    max_k = c
            y[i] = max_k
        return y

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        # TODO CHECK
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        constant = 1 / (np.sqrt(np.power(2 * np.pi, X.shape[0]) * np.linalg.det(self.cov_)))
        # exp_function_part = np.exp(
        #     -0.5 * np.matmul(np.transpose(X - self.mu_), np.matmul(np.linalg.inv(self.cov_), (X - self.mu_))))
        exp_function_part = np.exp(-0.5 * np.transpose(X - self.mu_) @ np.linalg.inv(self.cov_) @ (X - self.mu_))
        return np.product(constant * exp_function_part)
        # raise NotImplementedError()

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
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
        # from ...metrics import misclassification_error
        # raise NotImplementedError()
        return misclassification_error(y_pred=self.predict(X), y_true=y)
