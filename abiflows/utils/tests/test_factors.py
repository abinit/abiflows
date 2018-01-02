# coding: utf-8
from __future__ import print_function, division, unicode_literals, absolute_import

from abiflows.core.testing import AbiflowsTest
from abiflows.utils.factors import lowest_nn_gte_mm

class TestFactors(AbiflowsTest):
    def test_lowest_nn_gte_mm(self):
        ll, exponents = lowest_nn_gte_mm(101, [2, 3, 5])
        self.assertEqual(ll, 108)
        self.assertEqual(exponents, (2, 3, 0))

        ll, exponents = lowest_nn_gte_mm(100, [2, 3, 5])
        self.assertEqual(ll, 100)
        self.assertEqual(exponents, (2, 0, 2))

        ll, exponents = lowest_nn_gte_mm(100, [2, 3, 5])
        self.assertEqual(ll, 100)
        self.assertEqual(exponents, (2, 0, 2))

        ll, exponents = lowest_nn_gte_mm(101322142376, [7, 2, 13])
        self.assertEqual(ll, 101322142376)
        self.assertEqual(exponents, (8, 3, 3))

        ll, exponents = lowest_nn_gte_mm(101322142375, [7, 2, 13])
        self.assertEqual(ll, 101322142376)
        self.assertEqual(exponents, (8, 3, 3))

        ll, exponents = lowest_nn_gte_mm(101322142377, [7, 2, 13])
        self.assertEqual(ll, 102820990636)
        self.assertEqual(exponents, (11, 2, 1))