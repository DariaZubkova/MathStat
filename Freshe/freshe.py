import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import similaritymeasures
import math
answer = -1
point = []

def checkPoint(d, i, j, pred):
    global answer, point
    if pred == 0:
        pair_pred = [i - 1, j]
    elif pred == 1:
        pair_pred = [i - 1, j - 1]
    else:
        pair_pred = [i, j - 1]
    if (answer == d):
        point.append([i, j])
    else:
        answer = d
        point.clear()
        point.append([i, j])
        point.append(pair_pred)
    return

def c(ca, i, j, P, Q):
    global point, answer
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = np.linalg.norm(P[0] - Q[0])
        answer = ca[0, 0]
        point.append([0, 0])
    elif i > 0 and j == 0:
        d = np.linalg.norm(P[i] - Q[0])
        ca[i, j] = max(c(ca, i - 1, 0, P, Q), d)
        if ca[i - 1, j] == d:
            checkPoint(d, i, j, 0)
    elif i == 0 and j > 0:
        d = np.linalg.norm(P[0] - Q[j])
        ca[i, j] = max(c(ca, 0, j - 1, P, Q), d)
        if ca[i, j - 1] == d:
            checkPoint(d, i, j, 2)
    elif i > 0 and j > 0:
        d = np.linalg.norm(P[i] - Q[j])
        ca[i, j] = max(min(c(ca, i - 1, j, P, Q), c(ca, i - 1, j - 1, P, Q), c(ca, i, j - 1, P, Q)), d)
        if ca[i - 1, j] == d:
            checkPoint(d, i, j, 0)
        elif ca[i - 1, j - 1] == d:
            checkPoint(d, i, j, 1)
        elif ca[i, j - 1] == d:
            checkPoint(d, i, j, 2)
    else:
        ca[i, j] = math.inf
    return ca[i, j]


def distFrechet(P, Q):
    P = np.array(P, np.float64)
    Q = np.array(Q, np.float64)
    len_p = len(P)
    len_q = len(Q)
    if len_p == 0 or len_q == 0:
        print("Incorrect curves")
        return
    ca = (np.ones((len_p, len_q), dtype=np.float64) * -1)
    indexP = len_p - 1
    indexQ = len_q - 1
    distF = c(ca, len_p - 1, len_q - 1, P, Q)
    while (True):
        if (indexP > 0 and indexQ > 0 and distF == ca[indexP - 1, indexQ - 1]):
            indexP = indexP - 1
            indexQ = indexQ - 1
        if (indexP > 0 and distF == ca[indexP - 1, indexQ]):
            indexP = indexP - 1
        elif (indexQ > 0 and distF == ca[indexP, indexQ - 1]):
            indexQ = indexQ - 1
        else:
            break
    return distF, indexP, indexQ

def drawDistFrechet(P, Q, ind1, ind2, point):
    l = mlines.Line2D([P[ind1][0], Q[ind2][0]], [P[ind1][1], Q[ind2][1]], color='cyan')
    ax.add_line(l)
    for pair in point:
        l = mlines.Line2D([P[pair[0]][0], Q[pair[1]][0]], [P[pair[0]][1], Q[pair[1]][1]], color='cyan')
        ax.add_line(l)

def funN(x, sigma):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(- ((x / sigma)**2) / 2) / sigma

def funU(x, dist):
    return (x < dist) * (x >= -dist) * 1 / (2 * dist)


n = 100
xx = np.random.standard_normal(n)
x = np.linspace(np.min(xx), np.max(xx), n)
sigma = 1
y = funN(x, sigma)

#sigma = 2
#Y = funN(x, sigma)


XX = np.random.uniform(-np.sqrt(3), np.sqrt(3), n)
X = np.linspace(np.min(XX), np.max(XX), n)
Y = funU(x, np.sqrt(3))


#Y = funU(x, np.sqrt(2))


P = [0] * n
for i in range(n):
    P[i] = [0] * 2

for i in range(n):
    P[i][0] = x[i]
    P[i][1] = y[i]

Q = [0] * n
for i in range(n):
    Q[i] = [0] * 2

for i in range(n):
    Q[i][0] = x[i]
    Q[i][1] = Y[i]


fig, ax = plt.subplots()
plt.plot(x, y, color='green')
plt.plot(x, Y, color='blue')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Frechet Dist')
ax.grid(True)
plt.legend(handles=[
mpatches.Patch(color='green', label='P'),
mpatches.Patch(color='blue', label='Q'),
mpatches.Patch(color='cyan', label='dist')])

distF, indexP, indexQ = distFrechet(P, Q)
if (distF != answer):
     point.clear()
print("My distance frechet = ", distF)
print("Indexes for distance frechet for two curves =", indexP, ",", indexQ)
print("Array repeat value distance frechet = ", point)

drawDistFrechet(P, Q, indexP, indexQ, point)

df = similaritymeasures.frechet_dist(P, Q)
print("Built-in function frechet_dist = ", df)

if distF == df:
    print("True: distFrechet = ", distF)
else:
    print("False: incorrect distFrechet")

plt.show()