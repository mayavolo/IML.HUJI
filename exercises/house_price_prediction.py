import os

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt

pio.templates.default = "simple_white"

def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    # Loading the data
    data = pd.read_csv(filename)

    # labels column
    labels = data.loc[:, "price"]

    # Removing the unnecessary columns
    relevant_features = data.drop(columns=["id", "date", "price", "zipcode", "lat", "long"])

    # Rounding bathrooms column
    feature = data.loc[:, "bathrooms"].round()
    relevant_features = relevant_features.drop(columns=["bathrooms"])
    relevant_features.insert(4, "bathrooms", feature)

    # Rounding floors column
    feature = data.loc[:, "floors"].round()
    relevant_features = relevant_features.drop(columns=["floors"])
    relevant_features.insert(7, "floors", feature)

    # Making renovation column binary
    feature = data.loc[:, "yr_renovated"]
    feature = feature.where(feature == 0, 1)
    relevant_features = relevant_features.drop(columns=["yr_renovated"])
    relevant_features.insert(1, "yr_renovated", feature)

    return relevant_features, labels


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # Iteration over every feature in the data set
    for (columnName, columnData) in X.iteritems():
        col_arr = columnData.to_numpy()
        # Calculating the Pearson Correlation
        pearson_correlation = (np.cov(col_arr, y.to_numpy()) / (np.std(col_arr) * np.std(y)))[0][0]

        # Plotting the scatter
        plot = plt.figure()
        plt.scatter(col_arr, y.to_numpy())
        plt.xlabel(str(columnName))
        plt.ylabel('response')
        plt.title(
            'the Pearson Correlation between ' + str(columnName) + ' and the house pricing is: ' + str(
                pearson_correlation))
        # plt.show()
        plt.savefig(output_path + 'correspondence_with_' + str(columnName) + '.png')
        plt.close(plot)

    # plt.show()


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    os.chdir('..')
    X, y = load_data(os.path.join(os.path.join(os.getcwd(), 'datasets'), 'house_prices.csv'))

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(data_frame, labels) #TODO Uncomment

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    percent_range = np.arange(10, 100, 1)
    lini = LinearRegression()
    mean_loss_array = []
    for percent in percent_range:
        mean_loss_sum = 0
        number_of_samples = np.ceil(train_x.shape[0] * (percent / 100)).astype(int)
        train_x_sampled_rows = train_x.sample(n=number_of_samples)
        train_y_samples_labels = train_y.loc[train_x_sampled_rows.index]
        for i in range(10):
            lini.fit(train_x_sampled_rows.to_numpy(), train_y_samples_labels.to_numpy())
            # summation of the mean losses
            mean_loss_sum += lini.loss(train_x_sampled_rows.to_numpy(), train_y_samples_labels.to_numpy())
        mean_loss_array.append(mean_loss_sum / 10.0)  # average mean loss

