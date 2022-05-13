import numpy as np
from typing import Tuple
from IMLearn.metalearners import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from IMLearn.metrics import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train - and test errors of AdaBoost in noiseless case
    ada_boost = AdaBoost(DecisionStump, iterations=n_learners)
    ada_boost.fit(train_X, train_y)
    test_losses = _q1(ada_boost, n_learners, test_X, test_y, train_X, train_y)

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    sct = _q2(T, lims, noise, test_X, test_y, ada_boost)

    # Question 3: Decision surface of best performing ensemble
    _q3(ada_boost, lims, noise, sct, test_X, test_losses, test_y)

    # Question 4: Decision surface with weighted samples
    _q4(ada_boost, lims, noise, train_X, train_y)


def _q1(model, n_learners, test_X, test_y, train_X, train_y):
    training_losses = []
    test_losses = []
    for i in range(1, n_learners):
        training_losses.append(model.partial_loss(train_X, train_y, i))
        test_losses.append(model.partial_loss(test_X, test_y, i))
    plt.figure(1)
    plt.plot(np.arange(1, n_learners), training_losses, color='c', label='training error')
    plt.plot(np.arange(1, n_learners), test_losses, color='m', label='test error')
    plt.xlabel("Number of fitted learners")
    plt.ylabel("Error")
    plt.title("Training and test errors as a function of the number of fitted learners")
    plt.legend()
    plt.show()
    return test_losses


def _q2(T, lims, noise, test_X, test_y, model):
    figure = make_subplots(rows=2, cols=2, subplot_titles=[f"{i} learners" for i in T], horizontal_spacing=0.05,
                           vertical_spacing=0.05)
    sct = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False, name="1",
                     marker=dict(color=(test_y == 1).astype(int), symbol=class_symbols[test_y.astype(int)],
                                 colorscale=[custom[0], custom[-1]], line=dict(color="black", width=1)))
    for i, t in enumerate(T):
        figure.add_traces(
            [decision_surface(lambda x: model.partial_predict(x, t), lims[0], lims[1], showscale=False), sct],
            rows=(i // 2) + 1, cols=(i % 2) + 1)
    figure.update_layout(
        width=1000, height=900,
        title=rf"$\textbf{{AdaBoost Decision Boundaries with noise={noise}}}$",
        margin=dict(t=100)).update_xaxes(matches='x', range=[-1, 1], constrain="domain").update_yaxes(
        matches='y', range=[-1, 1], constrain="domain", scaleanchor="x", scaleratio=1)
    # figure.write_image(f'AdaBoost_DB_Noise_{noise}.png')
    figure.show()
    return sct


def _q3(model, lims, noise, sct, test_X, test_losses, test_y):
    i = np.argmin(test_losses)
    figure = go.Figure(
        data=[
            decision_surface(lambda x: model.partial_predict(x, i + 1), lims[0], lims[1], showscale=False), sct],
        layout=go.Layout(width=600, height=500,
                         title=f"AdaBoost Best Performing Ensemble. Learners={i + 1} <br> Noise={noise}. Accuracy={accuracy(test_y, model.partial_predict(test_X, i + 1)):.3f}",
                         margin=dict(t=100)))
    figure.update_xaxes(range=[-1, 1], constrain="domain").update_yaxes(
        range=[-1, 1], constrain="domain", scaleanchor="x", scaleratio=1)
    # figure.write_image(f'AdaBoost_Best_performing_with_Noise_{noise}.png')
    figure.show()


def _q4(model, lims, noise, train_X, train_y):
    dots_size = model.D_ / np.max(model.D_) * 10

    figure = go.Figure(
        data=[decision_surface(model.predict, lims[0], lims[1], showscale=False),
              go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                         showlegend=False, marker=dict(color=(train_y == 1).astype(int),
                                                       size=dots_size,
                                                       symbol=class_symbols[train_y.astype(int)],
                                                       colorscale=[custom[0], custom[-1]],
                                                       line=dict(color="black", width=0.5)))],
        layout=go.Layout(width=600, height=600,
                         title=f"AdaBoost training set with a weighted point size, noise={noise}.", ))
    figure.update_xaxes(range=[-1, 1], constrain="domain").update_yaxes(range=[-1, 1], constrain="domain",
                                                                        scaleanchor="x", scaleratio=1)

    # figure.write_image(f'AdaBoost_trained_with_weights_Noise_{noise}.png')
    figure.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0, 250, 5000, 500)
    fit_and_evaluate_adaboost(0.4, 250, 5000, 500)
