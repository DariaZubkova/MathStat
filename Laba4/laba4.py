import seaborn as sns
from pylab import *
from scipy.stats import *
from numpy import random
import matplotlib.pyplot as plt


buf = [20, 60, 100]

def cdf_normal(x, i):
    y = norm.cdf(x)
    plt.subplot(1, len(buf), i + 1)
    plt.plot(x, y, 'b')

def cdf_laplace(x, i):
    y = laplace.cdf(x)
    plt.subplot(1, len(buf), i + 1)
    plt.plot(x, y, 'b')

def cdf_cauchy(x, i):
    y = cauchy.cdf(x)
    plt.subplot(1, len(buf), i + 1)
    plt.plot(x, y, 'b')

def cdf_poisson(x, i):
    y = poisson.cdf(x, 10)
    plt.subplot(1, len(buf), i + 1)
    plt.plot(x, y, 'b')

def cdf_uniform(x, i):
    y = uniform.cdf(x, -sqrt(3), 2 *sqrt(3))
    plt.subplot(1, len(buf), i + 1)
    plt.plot(x, y, 'b')

def emp_dist(num, x, start, end):
    x.sort()
    y = linspace(0, 1, num)
    y.sort()

    for j in range(num):
        if x[j] < start:
            x[j] = start
        elif x[j] > end:
            x[j] = end

    numDouble = num * 2
    xx = empty(numDouble)
    xx[0] = start
    for j in range(num - 1):
        xx[j * 2 + 1] = x[j]
        xx[j * 2 + 2] = x[j]
    xx[num * 2 - 1] = end

    yy = empty(numDouble)
    yy[0] = 0
    yy[1] = y[0]
    for j in range(num - 1):
        yy[j * 2 + 2] = y[j + 1]
        yy[j * 2 + 3] = y[j + 1]

    plt.plot(xx, yy, 'r')


cdf_dist = {
    'normal': cdf_normal,
    'laplace': cdf_laplace,
    'cauchy': cdf_cauchy,
    'poisson': cdf_poisson,
    'uniform': cdf_uniform
}

def rvs_normal(num):
    return norm.rvs(scale=1, size=num)

def rvs_laplace(num):
    return laplace.rvs(scale=1 / sqrt(2), size=num)

def rvs_cauchy(num):
    return cauchy.rvs(size = num)

def rvs_poisson(num):
    return poisson.rvs(10, size = num)

def rvs_uniform(num):
    return random.uniform(-sqrt(3), sqrt(3), num)

rvs_dist = {
    'normal': rvs_normal,
    'laplace': rvs_laplace,
    'cauchy': rvs_cauchy,
    'poisson': rvs_poisson,
    'uniform': rvs_uniform
}

def pdf_normal(x):
    return norm.pdf(x, 0, 1)

def pdf_laplace(x):
    return laplace.pdf(x)

def pdf_cauchy(x):
    return cauchy.pdf(x)

def pdf_poisson(x):
    return poisson.pmf(x, 10)

def pdf_uniform(x):
    return uniform.pdf(x, -sqrt(3), 2 * sqrt(3))

pdf_dist = {
    'normal': pdf_normal,
    'laplace': pdf_laplace,
    'cauchy': pdf_cauchy,
    'poisson': pdf_poisson,
    'uniform': pdf_uniform
}

def emp_research(type_dist):
    fig, ax = plt.subplots(1, 3)
    for i in range(len(buf)):
        num = buf[i]
        if type_dist != 'poisson':
            start = -4
            end = 4
            x = linspace(start, end, 1000)
        else:
            start = 6
            end = 14
            x = linspace(start, end, 1000)
        cdf_dist[type_dist](x, i)
        xx = rvs_dist[type_dist](num)
        emp_dist(num, xx, start, end)
        if i == 1:
            plt.title('Empirical ' + type_dist + ', n = 20, 60, 100')
    plt.show()

def ker_research(type_dist):
    for i in range(len(buf)):
        fig, ax = plt.subplots(1, 3)
        num = buf[i]
        if type_dist != 'poisson':
            start = -4
            end = 4
            x = linspace(start, end, 1000)
        else:
            start = 6
            end = 14
            x = linspace(start - 1, end + 1, 11)

        y = pdf_dist[type_dist](x)

        ax[0].plot(x, y, 'b')
        ax[1].plot(x, y, 'b')
        ax[2].plot(x, y, 'b')

        xx = rvs_dist[type_dist](num)

        xx = xx[xx <= end]
        xx = xx[xx >= start]
        xx.sort()

        kde = gaussian_kde(xx)
        kde.set_bandwidth(bw_method='silverman')
        h_n = kde.factor

        sns.kdeplot(xx, ax=ax[0], bw=h_n / 2, color="r")
        sns.kdeplot(xx, ax=ax[1], bw=h_n, color="r")
        sns.kdeplot(xx, ax=ax[2], bw=2 * h_n, color="r")

        ax[0].set_title('h = h_n / 2')
        ax[1].set_title('h = h_n')
        ax[2].set_title('h = 2 * h_n')

        plt.suptitle('Kernel ' + type_dist + ', n=' + str(num))
        plt.show()
    print()

type_dist = ['normal', 'laplace', 'cauchy', 'poisson', 'uniform']
for i in range(5):
    emp_research(type_dist[i])
    ker_research(type_dist[i])
