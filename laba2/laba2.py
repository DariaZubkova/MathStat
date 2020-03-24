import numpy
import sys

"""def standart_normal(x):
    return (1 / numpy.sqrt(2*numpy.pi)) * numpy.exp(- x * x / 2)

def standart_cauchy(x):
    return 1 / (numpy.pi * (1 + x*x))

def laplace(x):
    return 1 / numpy.sqrt(2) * numpy.exp (-numpy.sqrt(2) * numpy.abs(x))

def uniform(x):
    flag2 = x <= numpy.sqrt(3)
    flag1 = x >= -numpy.sqrt(3)
    return 1 / (numpy.sqrt(3) + numpy.sqrt(3)) * flag1 * flag2

def poisson(x):
    k = 2
    return (numpy.power(x, k) / numpy.math.factorial(k)) * numpy.exp(-x)"""

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

def Sum(z):
    sum = 0
    for i in range(len(z)):
        sum += z[i]
    return sum

def X(x):
    return Sum(x) / len(x)

def Zr(x):
    return (numpy.amin(x) + numpy.amax(x))/2

def Zq(x):
    return (numpy.quantile(x, 1/4) + numpy.quantile(x, 3/4) ) /2

def Ztr(x):
    length = x.size
    r = (int)(length / 4)
    sum = 0
    for i in range(r, length - r):
        sum += x[i]
    return sum/(length - 2 * r)

name_variable_value = {
    'x': numpy.mean,
    'med(x)': numpy.median,
    'Zr': Zr,
    'Zq': Zq,
    'Ztr': Ztr
}

name_variable = [
    'x',
    'med(x)',
    'Zr',
    'Zq',
    'Ztr'
]

def E(z):
    #return Sum(z) / len(z)
    return numpy.mean(z)

def D(z):
    """# calculate mean
    m = Sum(z) / len(z)

    # calculate variance using a list comprehension
    var_res = sum((xi - m) ** 2 for xi in z) / len(z)
    return var_res"""
    return numpy.var(z)

f = open('out.csv', 'w')
sys.stdout = f

def research(type):
    print('-------------------------------------')
    print(type)
    arr = [10, 100, 1000]
    for num in arr:
        print_table = {
            'E(z)': [],
            'D(z)': []
        }

        for variable in name_variable:
            z = []
            for i in range(0, 1000):
                array = numpy.sort(type_dist_value[type](num))
                array.sort()
                z.append(name_variable_value[variable](array))
            print_table['E(z)'].append(E(z))
            print_table['D(z)'].append(D(z))

        print()
        print("%-10s;" %('n = %i' %num), end="")
        for variable in name_variable:
            print("%-12s;" % variable, end="")

        print()
        print("%-10s;" % ('E(z) ='), end="")
        for e in print_table['E(z)']:
            print("%-12f;" % e, end="")

        print()
        print("%-10s;" %('D(z) ='), end="")
        for d in print_table['D(z)']:
            print("%-12f;" % d, end="")
        print()

for name in type_dist:
    research(name)
