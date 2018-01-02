# coding: utf-8
from __future__ import print_function, division, unicode_literals, absolute_import

import json
import os

from pymatgen.io.abinit.qadapters import QueueAdapter
from fireworks.core.firework import Firework
from abiflows.core.testing import AbiflowsTest
from abiflows.fireworks.tasks.handlers import WalltimeHandler
from abiflows.fireworks.tasks.utility_tasks import CheckTask


test_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                        "test_files")

class TestWalltimeHandler(AbiflowsTest):

    def test_walltime_handler(self):
        # Simple test from a queue_adapter
        f = open(os.path.join(test_dir, "fw_handlers", "walltime", "PBS_qadapter.json"))
        dd = json.load(f)
        f.close()
        qad = QueueAdapter.from_dict(dd)
        walltime_handler = WalltimeHandler(job_rundir=os.path.join(test_dir, "fw_handlers", "walltime"),
                                           qout_file='queue.qout', qerr_file='queue.qerr', queue_adapter=qad,
                                           max_timelimit=1200, timelimit_increase=600)
        self.assertTrue(walltime_handler.check())
        # Test with a given Firework
        walltime_handler = WalltimeHandler(job_rundir='.',
                                           qout_file='queue.qout', qerr_file='queue.qerr', queue_adapter=None,
                                           max_timelimit=1200, timelimit_increase=600)
        # Get the Firework from json file
        f = open(os.path.join(test_dir, 'fw_handlers', 'walltime', 'sleep_fw.json'))
        dd = json.load(f)
        f.close()
        fw_to_check = Firework.from_dict(dd)

        # Hack the launch_dir so that it points to the directory where the queue.qout is
        fw_to_check.launches[-1].launch_dir = os.path.join(test_dir, fw_to_check.launches[-1].launch_dir)
        walltime_handler.src_setup(fw_spec={}, fw_to_check=fw_to_check)

        self.assertTrue(walltime_handler.check())
        actions = walltime_handler.correct()
        self.assertEqual(actions, {'errors': ['WalltimeHandler'],
                                   'actions': [{'action': {'_set': {'timelimit': 660.0}},
                                                'object': {'source': 'fw_spec', 'key': 'qtk_queueadapter'},
                                                'action_type': 'modify_object'}]})