# coding: utf-8
from __future__ import unicode_literals, division, print_function

import os
import abipy.data as abidata
import abipy.abilab as abilab
from abiflows.fireworks.tasks.abinit_tasks import RestartInfo
from abiflows.fireworks.utils.task_history import TaskHistory, TaskEvent
from abiflows.core.testing import AbiflowsTest
from abipy.abio.factories import ion_ioncell_relax_input
from pymatgen.io.abinit.events import Correction, DilatmxErrorHandler, DilatmxError

class TestTaskHistory(AbiflowsTest):

    def test_task_history_and_events(self):
        th = TaskHistory()
        th.log_autoparal({u'time': u'12:0:0', u'ntasks': 15, u'partition': 'defq', u'nodes': 1, u'mem_per_cpu': 3000})
        th.log_finalized()
        th.log_restart(RestartInfo(os.path.abspath('.'), reset=True, num_restarts=2))
        th.log_unconverged()
        th.log_corrections([Correction(DilatmxErrorHandler(), {}, DilatmxError('', '', '',), )])
        th.log_abinit_stop(run_time=100)

        si = abilab.Structure.from_file(abidata.cif_file("si.cif"))
        si_relax_input = ion_ioncell_relax_input(si, abidata.pseudos("14si.pspnc"), ecut=2).split_datasets()[1]

        th.log_converge_params({'dilatmx': 1.03}, si_relax_input)

        self.assertMSONable(th)

        for te in th:
            self.assertMSONable(te)

    def test_get_events_by_types(self):
        th = TaskHistory()
        th.log_unconverged()
        th.log_unconverged()
        th.log_abinit_stop(run_time=100)
        th.log_finalized()

        events = th.get_events_by_types([TaskEvent.UNCONVERGED, TaskEvent.FINALIZED])

        self.assertEqual(len(th), 4)
        self.assertEqual(len(events), 3)

    def test_get_total_run_time(self):
        th = TaskHistory()
        th.log_abinit_stop(run_time=100)
        th.log_abinit_stop(run_time=100)
        th.log_abinit_stop(run_time=100)

        total_run_time = th.get_total_run_time()

        self.assertEqual(total_run_time, 300)


