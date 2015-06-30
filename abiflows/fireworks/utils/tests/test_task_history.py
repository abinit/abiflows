# coding: utf-8
from __future__ import unicode_literals, division, print_function

import os
from abiflows.fireworks.tasks.abinit_tasks import RestartInfo
from abiflows.fireworks.utils.task_history import TaskHistory
from abipy.core.testing import AbipyTest
from pymatgen.io.abinitio.events import Correction, DilatmxErrorHandler, DilatmxError

class TestTaskHistory(AbipyTest):

    def test_task_history_and_events(self):
        th = TaskHistory()
        th.log_autoparal({u'time': u'12:0:0', u'ntasks': 15, u'partition': 'defq', u'nodes': 1, u'mem_per_cpu': 3000})
        th.log_finalized()
        th.log_restart(RestartInfo(os.path.abspath('.'), reset=True, num_restarts=2))
        th.log_concluded()
        th.log_unconverged()
        th.log_corrections([Correction(DilatmxErrorHandler(), {}, DilatmxError('', '', '',), )])

        self.assertPMGSONable(th)

        for te in th:
            self.assertPMGSONable(te)