from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    sigma = np.sqrt(noise)
    min_val = -1.2
    max_val = 2
    train_proportion = (2.0 / 3.0)
    x = np.linspace(min_val, max_val, n_samples)
    dataset_x = x
    dataset_y = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2) + np.random.randn(n_samples) * sigma
    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(dataset_x), pd.Series(dataset_y), train_proportion)
    # Plotting
    fig, ax = plt.subplots(1)

    ax.scatter(dataset_x, (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2), c='blueviolet', label='true noiseless set',
               alpha=0.7, edgecolors='none')
    ax.scatter(train_x, train_y, c='blue', label='train set', alpha=0.8, edgecolors='none')
    ax.scatter(test_x, test_y, c='springgreen', label='test set', alpha=0.8, edgecolors='none')
    ax.legend()
    plt.title('Question 1')
    plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    max_degree = 10
    k_fold_number = 5
    # Creating polynomial fittings objects for every degree
    poly_fits = [PolynomialFitting(k) for k in range(max_degree + 1)]
    # Making 5-fold cross validation
    avg_train_err = np.zeros(max_degree + 1)
    avg_validation_err = np.zeros(max_degree + 1)
    division = np.remainder(np.arange(train_x.size), k_fold_number)
    for k in range(k_fold_number):
        train_k_x = train_x[division != k].to_numpy().ravel()
        train_k_y = train_y[division != k].to_numpy().ravel()
        val_k_x = train_x[division == k].to_numpy().ravel()
        val_k_y = train_y[division == k].to_numpy().ravel()
        for d in range(max_degree + 1):
            poly_fits[d].fit(train_k_x, train_k_y)
        loss_train_d = [poly_fits[d].loss(train_k_x, train_k_y) for d in range(max_degree + 1)]
        loss_validation_d = [poly_fits[d].loss(val_k_x, val_k_y) for d in range(max_degree + 1)]
        avg_train_err += np.array(loss_train_d) / k_fold_number
        avg_validation_err += np.array(loss_validation_d) / k_fold_number
    # Plotting
    fig = plt.figure(2)
    degrees = np.arange(0, max_degree + 1, 1)
    plt.plot(degrees, avg_train_err, '-o', color='springgreen', label='Train Error')
    plt.plot(degrees, avg_validation_err, '-o', color='blue', label='Test Error')
    plt.legend()
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Average Error")
    plt.title('The Average Training and Test Errors for Different Polynomial Degrees')
    plt.show()
    # ------------------------------------------------------------------------------------------------------------------
    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_degree = int(np.argmin(avg_validation_err))
    print(f'The best degree is {best_degree}')
    best_poly_fit = PolynomialFitting(best_degree)
    best_poly_fit.fit(train_x.to_numpy().ravel(), train_y.to_numpy().ravel())
    # Validation Error
    loss_best = best_poly_fit.loss(test_x.to_numpy().ravel(), test_y.to_numpy().ravel())
    print('The test loss using the best value is:', loss_best)
    # ------------------------------------------------------------------------------------------------------------------


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    train_size = n_samples
    k_fold = 5
    X, y = datasets.load_diabetes(return_X_y=True)
    train_x = X[:train_size, :]
    train_y = y[:train_size]
    test_x = X[train_size:, :]
    test_y = y[train_size:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions

    # Ridge Regression Cross-Validation
    lambda_range = np.linspace(0, 1, num=n_evaluations)

    # Creating ridge regression objects for every degree
    ridge_models = [RidgeRegression(lam) for lam in lambda_range]

    ridge_avg_train_error = np.zeros(n_evaluations)
    ridge_avg_validation_error = np.zeros(n_evaluations)
    division = np.remainder(np.arange(train_y.size), k_fold)
    for k in range(k_fold):
        train_k_x = train_x[division != k]
        train_k_y = train_y[division != k]
        val_k_x = train_x[division == k]
        val_k_y = train_y[division == k]
        for i in range(lambda_range.size):
            ridge_models[i].fit(train_k_x, train_k_y)

        loss_train_d = [ridge_model.loss(train_k_x, train_k_y) for ridge_model in ridge_models]
        loss_validation_d = [ridge_model.loss(val_k_x, val_k_y) for ridge_model in ridge_models]
        ridge_avg_train_error += np.array(loss_train_d) / k_fold
        ridge_avg_validation_error += np.array(loss_validation_d) / k_fold

    # Lasso Regression Cross-Validation
    lasso_models = [Lasso(i) for i in range(lambda_range.size)]

    lasso_avg_train_error = np.zeros(n_evaluations)
    lasso_avg_validation_error = np.zeros(n_evaluations)
    division = np.remainder(np.arange(train_y.size), k_fold)

    for k in range(k_fold):
        train_k_x = train_x[division != k]
        train_k_y = train_y[division != k]
        val_k_x = train_x[division == k]
        val_k_y = train_y[division == k]
        for i in range(lambda_range.size):
            lasso_models[i].fit(train_k_x, train_k_y)
        loss_train_d = [np.mean((l_model.predict(train_k_x) - train_k_y) ** 2) for l_model in lasso_models]
        loss_validation_d = [np.mean((l_model.predict(val_k_x) - val_k_y) ** 2) for l_model in lasso_models]

        lasso_avg_train_error += np.array(loss_train_d) / k_fold
        lasso_avg_validation_error += np.array(loss_validation_d) / k_fold

    # Plotting
    fig = plt.figure(3)

    plt.plot(lambda_range, ridge_avg_train_error, color='springgreen', label='train error - ridge')
    plt.plot(lambda_range, ridge_avg_validation_error, color='blue', label='validation error - ridge')
    plt.plot(lambda_range, lasso_avg_train_error, color='mediumturquoise', label='train error - lasso')
    plt.plot(lambda_range, lasso_avg_validation_error, color='magenta', label='validation error - lasso')
    plt.legend()
    plt.title('The Average Training and Test Errors as a Function \n of the Tested Regularization Parameter Value')
    plt.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lambda_ridge = lambda_range[np.argmin(ridge_avg_validation_error)]
    best_lambda_lasso = lambda_range[np.argmin(lasso_avg_validation_error)]
    print('The best lambda for ridge is', best_lambda_ridge)
    print('The best lambda for lasso is', best_lambda_lasso)

    # Fitting models with the best lambda
    best_ridge = RidgeRegression(best_lambda_ridge)
    best_lasso = Lasso(best_lambda_lasso)
    best_least_squares = LinearRegression()
    best_ridge.fit(train_x, train_y)
    best_lasso.fit(train_x, train_y)
    best_least_squares.fit(train_x, train_y)
    print(f'Error for Least Squares:{best_least_squares.loss(test_x, test_y)}')
    print(f'Error for Ridge:{best_ridge.loss(test_x, test_y)}')
    print(f'Error for Lasso:{np.mean((best_lasso.predict(test_x) - test_y) ** 2)}')


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(n_samples=100, noise=5)
    select_polynomial_degree(n_samples=100, noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter(n_samples=50, n_evaluations=500)
