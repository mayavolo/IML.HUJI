from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import matplotlib.pyplot as plt
import os
from IMLearn.metrics import accuracy


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    os.chdir('..')
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        samples, labels = load_dataset(os.path.join(os.path.join(os.getcwd(), 'datasets'), f))

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        p = Perceptron()
        p.fit(samples, labels)
        losses = p.losses

        # Plot figure of loss as function of fitting iteration
        iterations = np.arange(len(losses))
        plot = plt.figure()
        plt.plot(iterations, losses, color='navy')
        plt.xlabel('Iterations')
        plt.ylabel('Losses')
        plt.title(f'Losses during Perceptron fitting to the data set {n}')
        plt.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    # return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")
    # plt.scatter(mu[0] + xs, mu[1] + ys)
    plt.plot(mu[0] + xs, mu[1] + ys, '-o', color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """

    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        samples, labels = load_dataset(os.path.join(os.path.join(os.getcwd(), 'datasets'), f))

        # Fit models and predict over training set
        g = GaussianNaiveBayes()
        lda = LDA()
        g.fit(samples, labels)
        lda.fit(samples, labels)
        g_prediction = g.predict(samples)
        lda_prediction = lda.predict(samples)

        unique_classes = np.unique(labels)
        samples_by_class = []
        for uc in unique_classes:
            indices = (labels == uc).nonzero()[0]
            samples_by_class.append(samples[indices])

        samples_by_g_pred = []
        for uc in unique_classes:
            indices = (g_prediction == uc).nonzero()[0]
            samples_by_g_pred.append(samples[indices])

        samples_by_lda_pred = []
        for uc in unique_classes:
            indices = (lda_prediction == uc).nonzero()[0]
            samples_by_lda_pred.append(samples[indices])
        # Plot a figure with two subplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots

        plot = plt.figure(f)
        plt.suptitle(f)
        plt.subplot(1, 2, 1)
        m = ["d", 'p', '*', 'H', "^"]
        for i in range(len(samples_by_class)):
            plt.scatter(samples_by_class[i][:, 0], samples_by_class[i][:, 1], marker=m[i],
                       cmap='cool', linewidths=3, label='samples')

        plt.title(f'Gaussian Naive Bayes classifier {accuracy(labels, g_prediction)} ')
        plt.subplot(1, 2, 2)
        for i in range(len(samples_by_class)):
            plt.scatter(samples_by_class[i][:, 0], samples_by_class[i][:, 1], marker=m[i],
                        cmap='cool', linewidths=3,label='samples')

        plt.title(f'LDA classifier {accuracy(labels, lda_prediction)}')

        # Add ellipses depicting the covariances of the fitted Gaussians
        plt.subplot(1, 2, 1)
        for i in range(len(samples_by_g_pred)):
            mu = np.mean(samples_by_g_pred[i], axis=0)
            plt.scatter(mu[0], mu[1], marker='x', color='k', linewidths=2)
            cov = np.matmul(
                np.reshape(samples_by_g_pred[i] - mu, (samples_by_g_pred[i].shape[1], samples_by_g_pred[i].shape[0])),
                np.transpose(np.reshape(samples_by_g_pred[i] - mu,
                                        (samples_by_g_pred[i].shape[1], samples_by_g_pred[i].shape[0])))) / \
                  samples_by_g_pred[i].shape[0]

            get_ellipse(mu, cov)

        plt.subplot(1, 2, 2)
        for i in range(len(samples_by_lda_pred)):
            mu = np.mean(samples_by_lda_pred[i], axis=0)
            plt.scatter(mu[0], mu[1], marker='x', color='k', linewidths=2)
            cov = np.matmul(
                np.reshape(samples_by_lda_pred[i] - mu,
                           (samples_by_lda_pred[i].shape[1], samples_by_lda_pred[i].shape[0])),
                np.transpose(np.reshape(samples_by_lda_pred[i] - mu,
                                        (samples_by_lda_pred[i].shape[1], samples_by_lda_pred[i].shape[0])))) / \
                  samples_by_lda_pred[i].shape[0]

            get_ellipse(mu, cov)
        plt.show()
        # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
