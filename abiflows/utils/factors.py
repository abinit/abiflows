# coding: utf-8
"""
Utilities to get factors of a given number or related things ...
"""
from __future__ import print_function, division, unicode_literals, absolute_import

import itertools
import math

def lowest_nn_gte_mm(mm, factors):
    if mm < 1:
        raise ValueError('Number mm should be >= 1')
    if not any([factor > 1 for factor in factors]):
        raise ValueError('At least one factor should be > 1')
    if len(set(factors)) != len(factors):
        raise ValueError('The factors should all be different')
    lowest = -1
    current_exponents = -1
    max_exponents_factors = [int(math.ceil(math.log(float(mm), factor))) for factor in factors]

    exponents_possibilities = [range(max_exponent+1) for max_exponent in max_exponents_factors]
    for exponents in itertools.product(*exponents_possibilities):
        nn = 1
        for ifactor, factor in enumerate(factors):
            nn *= factor ** exponents[ifactor]
        if nn < mm:
            continue
        elif nn == mm:
            lowest = nn
            current_exponents = exponents
            break
        else:
            if lowest == -1:
                lowest = nn
                current_exponents = exponents
            elif nn < lowest:
                lowest = nn
                current_exponents = exponents

    return lowest, current_exponents