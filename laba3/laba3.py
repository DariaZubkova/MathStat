import numpy
import seaborn as sns
import matplotlib.pyplot as mplot
import sys

def laplace(x):
    return numpy.random.laplace(0, 1/numpy.sqrt(2), x)

def uniform(x):
    return numpy.random.uniform(-numpy.sqrt(3), numpy.sqrt(3), x)

def poisson(x):
    return numpy.random.poisson(2, x)

type_dist_value = {
    'normal': numpy.random.standard_normal,
    'cauchy': numpy.random.standard_cauchy,
    'laplace': laplace,
    'uniform': uniform,
    'poisson': poisson
}

type_dist = [
    'normal',
    'cauchy',
    'laplace',
    'uniform',
    'poisson'
]

def E(z):
    return numpy.mean(z)

def IQR(x):
    return numpy.abs(numpy.quantile(x, 1 / 4) - numpy.quantile(x, 3 / 4))


def ejection(x):
    length = x.size
    count = 0
    left = numpy.quantile(x, 1 / 4) - 1.5 * IQR(x)
    right = numpy.quantile(x, 3 / 4) + 1.5 * IQR(x)
    for i in range(0, length):
        if(x[i] < left or x[i] > right):
            count += 1
    return count / length

f = open('out2.csv', 'w')
sys.stdout = f

def research(type):
   # print('-------------------------------------')
    print()
    print(type)

    data = []

    selection = [20, 100]
    selection = numpy.sort(selection)

    for num in selection:
        eject = []
        arr = numpy.sort(type_dist_value[type](num))
        data.append(arr)

        for i in range(0, 1000):
            arr = numpy.sort(type_dist_value[type](num))
            eject.append(ejection(arr))

        print("%-10s;" % ('n = %i' % num), end="")
        print("%-12f;" % E(eject), end="")
        print()

    mplot.figure(type)
    mplot.title(type)
    sns.set(style="whitegrid")
    ax = sns.boxplot(data=data, orient='h')
    mplot.yticks(numpy.arange(2), ('20', '100'))
    mplot.show()

for name in type_dist:
    research(name)
f.close()