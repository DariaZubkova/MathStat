import numpy as np
import scipy.stats as stats

def m_interval(dist):
    m = np.mean(dist)
    s = np.std(dist)
    n = len(dist)
    interval = s * stats.t.ppf((1 + gamma) / 2, n - 1) / (n - 1) ** 0.5
    return np.around(m - interval, decimals=2), np.around(m + interval, decimals=2)


def var_interval(dist):
    s = np.std(dist)
    n = len(dist)
    low = s * (n / stats.chi2.ppf((1 + gamma) / 2, n - 1)) ** 0.5
    high = s * (n / stats.chi2.ppf((1 - gamma) / 2, n - 1)) ** 0.5
    return np.around(low, decimals=2), np.around(high, decimals=2)


def m_asimpt(dist):
    m = np.mean(dist)
    s = np.std(dist)
    n = len(dist)
    u = stats.norm.ppf((1 + gamma) / 2)
    interval = s * u / (n ** 0.5)
    return np.around(m - interval, decimals=2), np.around(m + interval, decimals=2)


def var_asimpt(dist):
    s = np.std(dist)
    n = len(dist)
    m_4 = stats.moment(dist, 4)
    e = m_4 / s**4 - 3
    u = stats.norm.ppf((1 + gamma) / 2)
    U = u * (((e + 2) / n) ** 0.5)
    low = s * (1 + 0.5 * U) ** (-0.5)
    high = s * (1 - 0.5 * U) ** (-0.5)
    return np.around(low, decimals=2), np.around(high, decimals=2)


buf = [20, 100]
gamma = 0.95
for num in buf:
    dist = np.random.normal(0, 1, size=num)
    print('size = ' + str(num))
    print('mean', m_interval(dist))
    print('variance', var_interval(dist))
    print('asimpt_mean', m_asimpt(dist))
    print('asimpt_variance', var_asimpt(dist))
