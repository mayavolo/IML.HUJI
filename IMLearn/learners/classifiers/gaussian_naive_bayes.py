from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)

        self.mu_ = np.zeros((self.classes_.shape[0], X.shape[1]))
        for i in range(self.classes_.shape[0]):
            indices = (y == self.classes_[i]).nonzero()
            mean = np.mean(X[indices], axis=0)
            self.mu_[i] = mean

        self.vars_ = np.zeros((self.classes_.shape[0], X.shape[1]))
        for i in range(self.classes_.shape[0]):
            indices = (y == self.classes_[i]).nonzero()
            var = np.var(X[indices], axis=0)
            self.vars_[i] = var

        self.pi_ = np.zeros((self.classes_.shape[0]))
        for i in range(self.classes_.shape[0]):
            indices = (y == self.classes_[i]).nonzero()[0].shape[0]
            self.pi_[i] = indices / X.shape[0]

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
                numerator = np.exp((-1 / 2) * ((X[i] - self.mu_[c]) ** 2) / (2 * self.vars_[c]))
                denominator = np.sqrt(2 * np.pi * self.vars_[c])
                likelihood = numerator / denominator
                prior = self.pi_[c]
                res = np.product(prior * likelihood)
                if res > max_p:
                    max_p = res
                    max_k = c
            y[i] = max_k
        return y

        # raise NotImplementedError()

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
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        raise NotImplementedError()

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
