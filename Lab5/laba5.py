import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import scipy.stats as stats
from matplotlib.patches import Ellipse
from tabulate import tabulate
import statistics

buf = [20, 60, 100]
rho = [0, 0.5, 0.9]

def quadrant(x, y):
    N = len(x)
    xx = np.empty(N, dtype=float)
    xx.fill(np.median(x))
    xx = x - xx
    yy = np.empty(N, dtype=float)
    yy.fill(np.median(y))
    yy = y - yy
    n = [0] * 4
    for i in range(N):
        if xx[i] >= 0 and yy[i] >= 0:
            n[0] += 1
        if xx[i] < 0 and yy[i] > 0:
            n[1] += 1
        if xx[i] < 0 and yy[i] < 0:
            n[2] += 1
        if xx[i] > 0 and yy[i] < 0:
            n[3] += 1
    return ((n[0] + n[2]) - (n[1] + n[3])) / N

def coef(num, rho):
    mean = [0, 0]
    cov = [[1.0, rho], [rho, 1.0]]
    p_coef = np.empty(1000, dtype=float)
    s_coef = np.empty(1000, dtype=float)
    q_coef = np.empty(1000, dtype=float)
    for i in range(1000):
        rv = stats.multivariate_normal.rvs(mean, cov, size=num)
        x = rv[:, 0]
        y = rv[:, 1]
        p_coef[i], t = stats.pearsonr(x, y)
        s_coef[i], t = stats.spearmanr(x, y)
        q_coef[i] = quadrant(x, y)
    return p_coef, s_coef, q_coef

def calculateValue(p_coef, s_coef, q_coef):
    p1 = np.median(p_coef)
    s1 = np.median(s_coef)
    q1 = np.median(q_coef)
    p_coef_twice = [p_coef[k] ** 2 for k in range(1000)]
    p2 = np.median(p_coef_twice)
    s_coef_twice = [s_coef[k] ** 2 for k in range(1000)]
    s2 = np.median(s_coef_twice)
    q_coef_twice = [q_coef[k] ** 2 for k in range(1000)]
    q2 = np.median(q_coef_twice)
    p3 = statistics.variance(p_coef)
    s3 = statistics.variance(s_coef)
    q3 = statistics.variance(q_coef)
    p = [p1, p2, p3]
    s = [s1, s2, s3]
    q = [q1, q2, q3]
    return p, s, q

def createTable(num, rho, p, s, q):
    lines = []
    if rho != -1:
        lines.append(["rho = " + str(rho), 'r', 'r_{S}', 'r_{Q}'])
    else:
        lines.append(["n = " + str(num), 'r', 'r_{S}', 'r_{Q}'])
    lines.append(['E(z)', np.around(p[0], decimals=3), np.around(s[0], decimals=3), np.around(q[0], decimals=3)])
    lines.append(['E(z^2)', np.around(p[1], decimals=3), np.around(s[1], decimals=3), np.around(q[1], decimals=3)])
    lines.append(['D(z)', np.around(p[2], decimals=3), np.around(s[2], decimals=3), np.around(q[2], decimals=3)])
    print(tabulate(lines, [], tablefmt="latex"))
    print('\n')

def valueRho(num, rho):
    p_coef, s_coef, q_coef = coef(num, rho)
    p, s, q = calculateValue(p_coef, s_coef, q_coef)
    createTable(num, rho, p, s, q)

def coefMixture(num):
    p_coef = np.empty(1000, dtype=float)
    s_coef = np.empty(1000, dtype=float)
    q_coef = np.empty(1000, dtype=float)
    for k in range(1000):
        rv = []
        for l in range(2):
            N = 0.9 * stats.multivariate_normal.rvs([0, 0], [[1, 0.9], [0.9, 1]], num) + \
                0.1 * stats.multivariate_normal.rvs([0, 0], [[10, -0.9], [-0.9, 10]], num)
            rv += list(N)
        rv = np.array(rv)
        x = rv[:, 0]
        y = rv[:, 1]
        p_coef[k], t = stats.pearsonr(x, y)
        s_coef[k], t = stats.spearmanr(x, y)
        q_coef[k] = quadrant(x, y)
    return p_coef, s_coef, q_coef

def valueRhoMixture(num):
    p_coef, s_coef, q_coef = coefMixture(num)
    p, s, q = calculateValue(p_coef, s_coef, q_coef)
    createTable(num, - 1, p, s, q)

def createEllipse(x, y, ax, **kwargs):
    n = 3.0
    cov = np.cov(x, y)
    p = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    radiusX = np.sqrt(1 + p)
    radiusY = np.sqrt(1 - p)
    ellipse = Ellipse((0, 0), width=radiusX * 2, height=radiusY * 2, facecolor='none', **kwargs) #facecolor='none'
    scaleX = np.sqrt(cov[0, 0]) * n
    meanX = np.mean(x)
    scaleY = np.sqrt(cov[1, 1]) * n
    meanY = np.mean(y)
    t = transforms.Affine2D().rotate_deg(45).scale(scaleX, scaleY).translate(meanX, meanY)
    ellipse.set_transform(t + ax.transData)
    return ax.add_patch(ellipse)

def scatter(num):
    fig, ax = plt.subplots(1, 3)
    fig.suptitle("n = " + str(num))
    #titles = [r'$ \rho = 0$', r'$\rho = 0.5 $', r'$ \rho = 0.9$']
    n = 0
    mean = [0, 0]
    for r in rho:
        cov = [[1.0, r], [r, 1.0]]
        rv = stats.multivariate_normal.rvs(mean, cov, size=num)
        x = rv[:, 0]
        y = rv[:, 1]
        ax[n].scatter(x, y, s=3)
        createEllipse(x, y, ax[n], edgecolor='navy')
        ax[n].scatter(np.mean(x), np.mean(y), c='r', s=3)
        ax[n].set_title('rho = ' + str(r))
        n += 1
    plt.show()

for i in range(len(buf)):
    num = buf[i]
    for r in rho:
        valueRho(num, r)
    print('------------------------------------------------')
for i in range(len(buf)):
    num = buf[i]
    valueRhoMixture(num)
for i in range(len(buf)):
    num = buf[i]
    scatter(num)