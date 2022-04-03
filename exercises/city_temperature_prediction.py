import os
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test
import numpy as np
import pandas as pd
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
    data = pd.concat([pd.get_dummies(data, columns=["Country"]), data["Country"]], axis=1)

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
    plt.ylabel("std of daily temperatures")
    plt.title("For each month the std of daily temperatures")

    # Question 3 - Exploring differences between countries
    israel_mean = pd.concat([y.loc[israel_samples.index], israel_samples["Month"]], axis=1).groupby("Month").agg("mean")
    israel_std = pd.concat([y.loc[israel_samples.index], israel_samples["Month"]], axis=1).groupby("Month").agg("std")

    the_netherlands_samples = X.loc[X['Country_The Netherlands'] == 1]
    the_netherlands_mean = pd.concat([y.loc[the_netherlands_samples.index], the_netherlands_samples["Month"]],
                                     axis=1).groupby("Month").agg("mean")
    the_netherlands_std = pd.concat([y.loc[the_netherlands_samples.index], the_netherlands_samples["Month"]],
                                    axis=1).groupby("Month").agg("std")

    south_africa_samples = X.loc[X['Country_South Africa'] == 1]
    south_africa_mean = pd.concat([y.loc[south_africa_samples.index], south_africa_samples["Month"]], axis=1).groupby(
        "Month").agg("mean")
    south_africa_std = pd.concat([y.loc[south_africa_samples.index], south_africa_samples["Month"]], axis=1).groupby(
        "Month").agg("std")

    jordan_samples = X.loc[X['Country_Jordan'] == 1]
    jordan_mean = pd.concat([y.loc[jordan_samples.index], jordan_samples["Month"]], axis=1).groupby("Month").agg("mean")
    jordan_std = pd.concat([y.loc[jordan_samples.index], jordan_samples["Month"]], axis=1).groupby("Month").agg("std")

    plt.figure(3)
    plt.errorbar(months, np.ravel(israel_mean.to_numpy()), yerr=np.ravel(israel_std.to_numpy()), label="Israel")
    plt.errorbar(months, np.ravel(the_netherlands_mean.to_numpy()), yerr=np.ravel(the_netherlands_std.to_numpy()),
                 label="The Netherlands")
    plt.errorbar(months, np.ravel(south_africa_mean.to_numpy()), yerr=np.ravel(south_africa_std.to_numpy()),
                 label="South Africa")
    plt.errorbar(months, np.ravel(jordan_mean.to_numpy()), yerr=np.ravel(jordan_std.to_numpy()), label="Jordan")
    plt.xlabel("Month")
    plt.ylabel("average of monthly temperatures")
    plt.title("For each month the average of monthly temperatures with std error bars")
    plt.legend(loc='upper left')

    # Question 4 - Fitting model for different values of `k`
    israel_samples = israel_samples.drop(columns=["Country"])
    x_train, y_train, x_test, y_test = split_train_test(israel_samples, y.loc[israel_samples.index], 0.75)
    x_train = x_train["Day Of Year"].to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test["Day Of Year"].to_numpy()
    y_test = y_test.to_numpy()

    losses = []
    for k in np.arange(1, 11):
        poly = PolynomialFitting(k)
        poly.fit(np.array([x_train]).ravel(), y_train)
        losses.append(poly.loss(np.array([x_test]).ravel(), y_test))
    print(losses)
    # Plot the losses
    plot = plt.figure(4)
    plt.xlabel('polynomial degree')
    plt.ylabel('loss')
    plt.title("The log loss as a function of the polynomial degree")
    plt.bar(np.arange(1, 11), np.log(np.array(losses)))

    # Question 5 - Evaluating fitted model on different countries
    k = 3  # best polynomial degree
    poly = PolynomialFitting(k)
    poly.fit(np.array([israel_samples["Day Of Year"].to_numpy()]).ravel(),
             y.loc[israel_samples["Day Of Year"].index].to_numpy())
    losses = [poly.loss(np.array([jordan_samples["Day Of Year"].to_numpy()]).ravel(),
                        y.loc[jordan_samples.index].to_numpy()),
              poly.loss(np.array([the_netherlands_samples["Day Of Year"].to_numpy()]).ravel(),
                        y.loc[the_netherlands_samples["Day Of Year"].index].to_numpy()),
              poly.loss(np.array([south_africa_samples["Day Of Year"].to_numpy()]).ravel(),
                        y.loc[south_africa_samples["Day Of Year"].index].to_numpy())]
    plt.figure(5)
    plt.bar(["Jordan", "The Netherlands", "South Africa"], losses)
    plt.xlabel('Countries')
    plt.ylabel('losses')
    plt.title("Losses of countries when training on Israel samples")

    plt.show()
