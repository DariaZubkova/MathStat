import numpy as np
import matplotlib.pyplot as plt

N = [10, 50, 1000]
for i in range(3):
    y = np.random.standard_normal(N[i])
    plt.hist(y, 20, density=True)
    Y = np.linspace(np.min(y), np.max(y), 100)
    plt.plot(Y, (1 / np.sqrt(2 * np.pi)) * np.exp(-Y * Y / 2))
    plt.show()

for i in range(3):
    y = np.random.laplace(0, 1/np.sqrt(3), N[i])
    plt.hist(y, 20, density=True)
    Y = np.linspace(np.min(y), np.max(y), 100)
    plt.plot(Y, (1 / np.sqrt(2)) * np.exp(-np.sqrt(2) * np.abs(Y)))
    plt.show()

for i in range(3):
    y = np.random.standard_cauchy(N[i])
    plt.hist(y, 20, density=True)
    Y = np.linspace(np.min(y), np.max(y), 100)
    plt.plot(Y, 1 / (np.pi * (1 + Y * Y)))
    plt.show()

#lambda = 7
for i in range(3):
    y = np.random.poisson(7, N[i])
    plt.hist(y, 20, density=True)
    Y = np.linspace(np.min(y), np.max(y), 100)
    plt.plot(Y, (np.power(Y, 7) / np.math.factorial(7)) * np.exp(-Y))
    plt.show()

for i in range(3):
    y = np.random.uniform(-np.sqrt(3), np.sqrt(3), N[i])
    plt.hist(y, 20, density=True)
    Y = np.linspace(np.min(y), np.max(y), 100)
    plt.plot(Y, 1 / (2 * np.sqrt(3)) * (Y <= np.sqrt(3)))
    plt.show()