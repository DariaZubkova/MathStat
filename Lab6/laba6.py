import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def generate(a, b, h):
    n = int((b - a) / h) + 2
    x = np.linspace(-1.8, 2, n)
    e_i = np.random.standard_normal(n)
    y = 2 + 2*x + e_i
    return x, y

def addDisturbance(y):
    res = []
    for i in range(0, len(y)):
        res.append(y[i])
    res[0] = y[0] + 10
    res[-1] = y[-1] - 10
    return res

def MNK(x, y):
    beta_1 = (np.mean(x*y) - np.mean(x) * np.mean(y))/(np.mean(x*x) - np.mean(x)**2)
    beta_0 = np.mean(y) - beta_1 * np.mean(x)
    return beta_1, beta_0

def _fun_for_minim(params, x, y):
    a_1, a_2 = params
    res = 0
    for i in range(len(x)):
        res += abs(a_1 * x[i] + a_2 - y[i])
    return res

def MNM(x, y, beta_0, beta_1):
    result = minimize(_fun_for_minim, [beta_0, beta_1], args=(x, y), method='SLSQP')
    coefs = result.x
    a_0, a_1 = coefs[0], coefs[1]
    return a_0, a_1

if __name__ == '__main__':
    l = -1.8
    r = 2
    h = 0.2
    a = 2
    b = 2

    plt.figure()
    #plt.subplot(121)
    plt.title("Without disturbance")
    print("\t\t\tWithout disturbance")
    x, y = generate(l, r, h)
    plt.plot(x, x * (a * np.ones(len(x))) + b * np.ones(len(x)), label='Model')
    plt.scatter(x, y)
    beta_1, beta_0 = MNK(x, y)
    print('МНК')
    print('a = ' + str(np.around(beta_0, decimals=2)))
    print('b = ' + str(np.around(beta_1, decimals=2)))
    print("%12s:\ta = %lf, b = %lf" % ("МНК", beta_0, beta_1))
    plt.plot(x, x * (beta_1 * np.ones(len(x))) + beta_0 * np.ones(len(x)), label='МHK')
    a_0, a_1 = MNM(x, y, beta_0, beta_1)
    print('МНМ')
    print('a = ' + str(np.around(a_0, decimals=2)))
    print('b = ' + str(np.around(a_1, decimals=2)))
    print("%12s:\ta = %lf, b = %lf" % ("МНМ", a_0, a_1))
    plt.plot(x, x * (a_1 * np.ones(len(x))) + a_0 * np.ones(len(x)), label='МHM')
    plt.xlim([-1.8, 2])
    plt.legend()
    plt.savefig('picture1.png', format='png')
    plt.show()

    print("\n")
    plt.figure()
    #plt.subplot(122)
    plt.title("With disturbance")
    print("\t\t\tWith disturbance")
    x, y = generate(l, r, h)
    y = addDisturbance(y)
    plt.plot(x, x * (a * np.ones(len(x))) + b * np.ones(len(x)), label='Model')
    plt.scatter(x, y)
    beta_1, beta_0 = MNK(x, y)
    print('МНК')
    print('a = ' + str(np.around(beta_0, decimals=2)))
    print('b = ' + str(np.around(beta_1, decimals=2)))
    print("%12s:\ta = %lf, b = %lf" % ("МНК", beta_0, beta_1))
    plt.plot(x, x * (beta_1 * np.ones(len(x))) + beta_0 * np.ones(len(x)), label='МHK')
    a_0, a_1 = MNM(x, y, beta_0, beta_1)
    print('МНМ')
    print('a = ' + str(np.around(a_0, decimals=2)))
    print('b = ' + str(np.around(a_1, decimals=2)))
    print("%12s:\ta = %lf, b = %lf" % ("МНМ", a_0, a_1))
    plt.plot(x, x * (a_1 * np.ones(len(x))) + a_0 * np.ones(len(x)), label='МHM')
    plt.xlim([-1.8, 2])
    plt.legend()
    plt.savefig('picture2.png', format='png')
    plt.show()