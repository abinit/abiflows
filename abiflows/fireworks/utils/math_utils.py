from __future__ import print_function, division, unicode_literals, absolute_import

from math import sqrt
from six.moves import reduce

##############################################################
### cartesian product of lists ##################################
##############################################################

def _append_es2sequences(sequences, es):
    result = []
    if not sequences:
        for e in es:
            result.append([e])
    else:
        for e in es:
            result += [seq+[e] for seq in sequences]
    return result


def _cartesian_product(lists):
    """
    given a list of lists,
    returns all the possible combinations taking one element from each list
    The list does not have to be of equal length
    """
    return reduce(_append_es2sequences, lists, [])


def prime_factors(n):
    """Lists prime factors of a given natural integer, from greatest to smallest
    :param n: Natural integer
    :rtype : list of all prime factors of the given natural n
    """
    i = 2
    while i <= sqrt(n):
        if n % i == 0:
            l = prime_factors(n/i)
            l.append(i)
            return l
        i += 1
    return [n]      # n is prime


def _factor_generator(n):
    """
    From a given natural integer, returns the prime factors and their multiplicity
    :param n: Natural integer
    :return:
    """
    p = prime_factors(n)
    factors = {}
    for p1 in p:
        try:
            factors[p1] += 1
        except KeyError:
            factors[p1] = 1
    return factors


def divisors(n):
    """
    From a given natural integer, returns the list of divisors in ascending order
    :param n: Natural integer
    :return: List of divisors of n in ascending order
    """
    factors = _factor_generator(n)
    _divisors = []
    listexponents = [[k**x for x in range(0, factors[k]+1)] for k in list(factors.keys())]
    listfactors = _cartesian_product(listexponents)
    for f in listfactors:
        _divisors.append(reduce(lambda x, y: x*y, f, 1))
    _divisors.sort()
    return _divisors