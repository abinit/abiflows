# coding: utf-8
from __future__ import print_function, division, unicode_literals, absolute_import

import os

from abiflows.core.testing import AbiflowsTest
from abiflows.fireworks.utils.time_utils import TimeReport, seconds_to_hms


class TestTimeReport(AbiflowsTest):
    def test_class(self):

        tr = TimeReport(120, 3, total_cpu_time=1200, contributed_cpu_time=3, total_run_time_per_tag={"tag": 10},
                 total_cpu_time_per_tag={"tag": 10}, contributed_cpu_time_per_tag={"tag": 3}, worker="worker name")

        self.assertMSONable(tr)

        str(tr)

        tr = TimeReport(120, 3)

        self.assertMSONable(tr)

        str(tr)


class TestFunctions(AbiflowsTest):

    def test_seconds_to_hms(self):
        time_string = seconds_to_hms(120)

        assert time_string == "0:02:00"
