import os
import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """

    # Loading the dataset
    data = pd.read_csv(filename, parse_dates=["Date"])

    # Removing samples where temp is too small to be true
    data = data[data["Temp"] > -70]

    # Getting labels column and removing it from dataset
    labels = data.loc[:, "Temp"]
    data = data.drop(columns=["Temp"])

    # Removing City and Day columns (unnecessary)
    data = data.drop(columns=["City", "Day"])

    # Getting dummies of categorical features in dataset
    data = pd.get_dummies(data, columns=["Country"])

    # Adding a day of year column based on date column
    dayofyear_column = data["Date"].dt.dayofyear
    data.insert(0, "Day Of Year", dayofyear_column)
    data = data.drop(columns=["Date"])

    # Returning results
    return data, labels


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    os.chdir('..')
    X, y = load_data(os.path.join(os.path.join(os.getcwd(), 'datasets'), 'City_Temperature.csv'))

    # Question 2 - Exploring data for specific country
    plt.figure(1)
    israel_samples = X.loc[X['Country_Israel'] == 1]
    plt.scatter(x=israel_samples["Day Of Year"], y=y.loc[israel_samples.index], c=israel_samples["Year"])
    plt.xlabel("Day Of Year")
    plt.ylabel("Temp")
    plt.title("Temp as function of day of year")

    fig = plt.figure(2)
    israel_labels_grouped_by_month = pd.concat([y.loc[israel_samples.index], israel_samples["Month"]], axis=1).groupby(
        "Month").agg("std")
    months = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    plt.bar(months, np.ravel(israel_labels_grouped_by_month.to_numpy()))
    plt.xlabel("Month")
    plt.ylabel("std of the daily temperatures")
    plt.title("For each month the std of the daily temperatures")
    plt.show()

    # Question 3 - Exploring differences between countries
    raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()
