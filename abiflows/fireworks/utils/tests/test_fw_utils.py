# coding: utf-8
from __future__ import print_function, division, unicode_literals, absolute_import

import os
import shutil
import unittest
import yaml

from monty.tempfile import ScratchDir
from fireworks import Firework, LaunchPad
from fireworks.core.rocket_launcher import rapidfire
from fireworks.user_objects.firetasks.script_task import PyTask
from abiflows.core.testing import AbiflowsTest, has_mongodb, TESTDB_NAME
from abiflows.fireworks.utils.fw_utils import *
from abiflows.fireworks.utils.tests.tasks import LpTask


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

test_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                        "test_files", "fw_task_managers")



class TestFWTaskManager(AbiflowsTest):

    def test_abipy_manager_from_file(self):
        with open(os.path.join(test_dir, "fw_manager_ok.yaml"), "rt") as fh:
            conf = yaml.load(fh)
        conf['fw_policy']['abipy_manager'] = os.path.join(test_dir, "manager_ok.yml")
        ftm = FWTaskManager(**conf)

        self.assertEqual(ftm.fw_policy.max_restarts, 20)

        ftm.update_fw_policy({'max_restarts': 30})

        self.assertTrue(ftm.has_task_manager())
        self.assertTrue(ftm.fw_policy.rerun_same_dir)
        self.assertEqual(ftm.fw_policy.max_restarts, 30)
        self.assertTrue(ftm.fw_policy.autoparal)

    def test_ok(self):

        ftm = FWTaskManager.from_file(os.path.join(test_dir, "fw_manager_ok.yaml"))
        ftm.update_fw_policy({'max_restarts': 30})

        self.assertTrue(ftm.fw_policy.rerun_same_dir)
        self.assertEqual(ftm.fw_policy.max_restarts, 30)
        self.assertTrue(ftm.fw_policy.autoparal)

    def test_local_qadapter(self):

        ftm = FWTaskManager.from_file(os.path.join(test_dir, "fw_manager_local_qadapter.yaml"))
        ftm.update_fw_policy({'max_restarts': 30})

        self.assertTrue(ftm.fw_policy.rerun_same_dir)
        self.assertEqual(ftm.fw_policy.max_restarts, 30)
        self.assertTrue(ftm.fw_policy.autoparal)

    def test_unknown_keys(self):

        with self.assertRaises(RuntimeError):
            ftm = FWTaskManager.from_file(os.path.join(test_dir, "fw_manager_unknown_keys.yaml"))

    def test_from_user_config(self):

        # create also using the from_user_config classmethod. Copy the file in the current folder
        with ScratchDir("."):
            shutil.copy2(os.path.join(test_dir, "fw_manager_ok.yaml"),
                         os.path.join(os.getcwd(), FWTaskManager.YAML_FILE))
            ftm = FWTaskManager.from_user_config()
            ftm.update_fw_policy({'max_restarts': 30})

            self.assertTrue(ftm.fw_policy.rerun_same_dir)
            self.assertEqual(ftm.fw_policy.max_restarts, 30)
            self.assertTrue(ftm.fw_policy.autoparal)


class TestFunctions(AbiflowsTest):

    @classmethod
    def setUpClass(cls):
        cls.setup_fireworks()
        ftm_path = os.path.join(test_dir, "fw_manager_ok.yaml")
        with open(ftm_path, "rt") as fh:
            conf = yaml.load(fh)
        conf['fw_policy']['abipy_manager'] = os.path.join(test_dir, "manager_ok.yml")
        cls.ftm = FWTaskManager(**conf)

    @classmethod
    def tearDownClass(cls):
        cls.teardown_fireworks(module_dir=MODULE_DIR)

    def tearDown(self):
        if self.lp:
            self.lp.reset(password=None,require_password=False)

    def test_get_short_single_core_spec(self):
        spec = get_short_single_core_spec(self.ftm, timelimit=610)

        assert spec['ntasks'] == 1
        assert spec['time'] == '0-0:10:10'

    def test_set_short_single_core_to_spec(self):
        spec = {}
        spec = set_short_single_core_to_spec(spec, fw_manager=self.ftm)

        assert spec['_queueadapter']['ntasks'] == 1
        assert spec['mpi_ncpus'] == 1

    @unittest.skipUnless(has_mongodb(), "A local mongodb is required.")
    def test_get_time_report_for_wf(self):
        task = PyTask(func="time.sleep", args=[0.5])
        fw1 = Firework([task], spec={'wf_task_index': "test1_1", "nproc": 16}, fw_id=1)
        fw2 = Firework([task], spec={'wf_task_index': "test2_1", "nproc": 16}, fw_id=2)
        wf = Workflow([fw1,fw2])
        self.lp.add_wf(wf)

        rapidfire(self.lp, self.fworker, m_dir=MODULE_DIR)

        wf = self.lp.get_wf_by_fw_id(1)

        assert wf.state == "COMPLETED"

        tr = get_time_report_for_wf(wf)

        assert tr.n_fws == 2
        assert tr.total_run_time > 1

    @unittest.skipUnless(has_mongodb(), "A local mongodb is required.")
    def test_get_lp_and_fw_id_from_task(self):
        """
        Tests the get_lp_and_fw_id_from_task. This test relies on the fact that the LaunchPad loaded from auto_load
        will be different from what is defined in TESTDB_NAME. If this is not the case the test will be skipped.
        """
        lp = LaunchPad.auto_load()

        if not lp or lp.db.name == TESTDB_NAME:
            raise unittest.SkipTest("LaunchPad lp {} is not suitable for this test. Should be available and different"
                                    "from {}".format(lp, TESTDB_NAME))

        task = LpTask()
        # this will pass the lp
        fw1 = Firework([task], spec={'_add_launchpad_and_fw_id': True}, fw_id=1)
        # this will not have the lp and should fail
        fw2 = Firework([task], spec={}, fw_id=2, parents=[fw1])
        wf = Workflow([fw1, fw2])
        self.lp.add_wf(wf)

        rapidfire(self.lp, self.fworker, m_dir=MODULE_DIR, nlaunches=1)

        fw = self.lp.get_fw_by_id(1)

        assert fw.state == "COMPLETED"

        rapidfire(self.lp, self.fworker, m_dir=MODULE_DIR, nlaunches=1)

        fw = self.lp.get_fw_by_id(2)

        assert fw.state == "FIZZLED"