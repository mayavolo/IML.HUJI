from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mean = 10
    std = 1
    sample_size = 1000
    samples = np.random.normal(mean, std, sample_size)
    u_gaussian = UnivariateGaussian()
    u_gaussian.fit(samples)
    print("(" + str(u_gaussian.mu_) + ", " + str(u_gaussian.var_) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    sample_sizes = np.arange(10, 1010, 10)
    absolute_distances = np.empty(sample_sizes.shape[0])
    for i in range(sample_sizes.shape[0]):
        new_samples = np.random.normal(mean, std, sample_sizes[i])
        u_gaussian = UnivariateGaussian()
        u_gaussian.fit(new_samples)
        absolute_distances[i] = np.abs(u_gaussian.mu_ - mean)

    plot = plt.figure()
    plt.plot(sample_sizes, absolute_distances)
    plt.xlabel('Sample size')
    plt.ylabel('Absolute distance between the estimated and true value of the expectation')
    plt.title(
        'Absolute distance between the estimated and true value of the expectation as a function of the sample size')
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_array = u_gaussian.pdf(samples)
    plot = plt.figure()
    plt.scatter(samples, pdf_array)
    plt.xlabel('Sample value')
    plt.ylabel('Sample value PDF')
    plt.title(
        'Sample value and their PDF\'s')
    plt.show()
    print("Because the model is fitted by the same sample that we calculate the PDF on, I expect to see the PDF"
          " of the model with its given parameters")
    # end


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean = np.transpose(np.array([0, 0, 4, 0]))
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    sample_size = 1000
    samples = np.random.multivariate_normal(mean, cov, sample_size)
    m_gaussian = MultivariateGaussian()
    m_gaussian.fit(samples)
    print("estimated expectation:\n" + str(np.reshape(m_gaussian.mu_, (4, 1))) + "\ncovariance matrix:\n" + str(
        m_gaussian.cov_))

    # Question 5 - Likelihood evaluation
    grid_size = 200
    f1 = np.linspace(-10, 10, grid_size)
    f3 = np.linspace(-10, 10, grid_size)
    log_likelihoods = np.zeros((grid_size, grid_size))
    indices = [0, 0]
    max = None
    for i in range(len(f1)):
        for j in range(len(f3)):
            current_mu = np.reshape([f1[i], 0, f3[j], 0], (4, 1))
            log_likelihoods[i][j] = m_gaussian.log_likelihood(current_mu, cov, np.transpose(samples))
            if not max:
                max = log_likelihoods[i][j]
                indices[0] = i
                indices[1] = j
            if log_likelihoods[i][j] > max:
                max = log_likelihoods[i][j]
                indices[0] = i
                indices[1] = j

    fig, ax = plt.subplots()
    im = ax.imshow(log_likelihoods)
    # Show all ticks and label them with the respective list entries
    f3_int = f3.astype(int)
    f1_int = f1.astype(int)
    ax.set_xticks(np.arange(grid_size), labels=f3_int)
    ax.set_yticks(np.arange(grid_size), labels=f1_int)
    plt.xlabel('f3 values')
    plt.ylabel('f1 values')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_title("Log likelihood heatmap of f1 values as rows and f3 values as columns")
    plt.show()

    # Question 6 - Maximum likelihood
    print("f1: " + str(f1[indices[0]].round(3)) + ", f3:" + str(f3[indices[1]].round(3)))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
