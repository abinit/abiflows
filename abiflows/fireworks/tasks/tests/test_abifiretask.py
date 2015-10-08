# coding: utf-8
from __future__ import unicode_literals, division, print_function

import os

import mock
import abipy.data as abidata
import abipy.abilab as abilab
import abiflows.fireworks.tasks.abinit_tasks as abinit_tasks
from abipy.abio.factories import *
from abipy.core.testing import AbipyTest
from abiflows.fireworks.tasks.tests import mock_objects
from pymatgen.io.abinit.events import Correction, DilatmxErrorHandler, DilatmxError
from fireworks import FWAction
import abiflows.fireworks.utils.fw_utils


class TestAbiFireTask(AbipyTest):

    def setUp(self):
        si = abilab.Structure.from_file(abidata.cif_file("si.cif"))
        self.si_scf_input = ebands_input(si, abidata.pseudos("14si.pspnc"), ecut=2, kppa=10).split_datasets()[0]

    def test_AbiFireTask(self):
        task = abinit_tasks.AbiFireTask(self.si_scf_input)
        task.to_dict()
        self.assertPMGSONable(self.si_scf_input)
        self.assertFwSerializable(task)

    def test_ScfFireTask(self):
        task = abinit_tasks.ScfFWTask(self.si_scf_input)
        task.to_dict()
        self.assertFwSerializable(task)


class TestTaskAnalysis(AbipyTest):
    def setUp(self):
        si = abilab.Structure.from_file(abidata.cif_file("si.cif"))
        self.si_scf_input = ebands_input(si, abidata.pseudos("14si.pspnc"), ecut=2, kppa=10).split_datasets()[0]

    @mock.patch.object(abinit_tasks.AbiFireTask, 'get_event_report')
    def test_scf_unconverged(self, report):
        scf_task = abinit_tasks.ScfFWTask(self.si_scf_input)
        scf_task.ftm = abiflows.fireworks.utils.fw_utils.FWTaskManager.from_user_config({})

        report.return_value = mock_objects.report_ScfConvergenceWarning()

        with mock.patch.object(abinit_tasks.AbiFireTask, 'prepare_restart',
                               return_value=(False, mock_objects.fake_fw, {})) as pr:
            fake_spec = {'test': 1}
            action = scf_task.task_analysis(fake_spec)
            pr.assert_called_once_with(fake_spec)
            self.assertIsInstance(action, FWAction)

            scf_task.restart_info = abinit_tasks.RestartInfo(
                previous_dir='.', num_restarts=abiflows.fireworks.utils.fw_utils.FWTaskManager.fw_policy_defaults['max_restarts'])
            with self.assertRaises(abinit_tasks.UnconvergedError):
                scf_task.task_analysis(fake_spec)

    @mock.patch.object(abinit_tasks.AbiFireTask, 'get_event_report')
    def test_generic_error(self, report):
        scf_task = abinit_tasks.ScfFWTask(self.si_scf_input)

        report.return_value = mock_objects.report_AbinitError()

        fake_spec = {'test': 1}
        with self.assertRaises(abinit_tasks.AbinitRuntimeError):
            # set the returncode to avoid logging problems
            scf_task.returncode = 10
            scf_task.task_analysis(fake_spec)

    @mock.patch.object(abinit_tasks.AbiFireTask, 'get_event_report')
    def test_no_report_no_err(self, report):
        scf_task = abinit_tasks.ScfFWTask(self.si_scf_input)
        scf_task.set_workdir(os.getcwd())

        report.return_value = None

        fake_spec = {'test': 1}
        with self.assertRaises(abinit_tasks.AbinitRuntimeError):
            # set the returncode to avoid logging problems
            scf_task.returncode = 10
            scf_task.task_analysis(fake_spec)


class TestFWTaskManager(AbipyTest):
    def tearDown(self):
        try:
            os.remove(abiflows.fireworks.utils.fw_utils.FWTaskManager.YAML_FILE)
        except OSError:
            pass

    def test_no_file(self):
        abiflows.fireworks.utils.fw_utils.FWTaskManager.from_user_config({})

    def test_ok(self):
        with open(os.path.abspath(abiflows.fireworks.utils.fw_utils.FWTaskManager.YAML_FILE), 'w') as f:
            f.write(mock_objects.MANAGER_OK)

        ftm = abiflows.fireworks.utils.fw_utils.FWTaskManager.from_user_config()
        ftm.update_fw_policy({'max_restarts': 30})

        self.assertTrue(ftm.fw_policy.rerun_same_dir)
        self.assertEqual(ftm.fw_policy.max_restarts, 30)
        self.assertTrue(ftm.fw_policy.autoparal)

    def test_no_qadapter(self):
        with open(os.path.abspath(abiflows.fireworks.utils.fw_utils.FWTaskManager.YAML_FILE), 'w') as f:
            f.write(mock_objects.MANAGER_NO_QADAPTERS)

        ftm = abiflows.fireworks.utils.fw_utils.FWTaskManager.from_user_config({})

        self.assertIsNone(ftm.task_manager)

    def test_unknown_keys(self):
        with open(os.path.abspath(abiflows.fireworks.utils.fw_utils.FWTaskManager.YAML_FILE), 'w') as f:
            f.write(mock_objects.MANAGER_UNKNOWN_KEYS)

        with self.assertRaises(RuntimeError):
            abiflows.fireworks.utils.fw_utils.FWTaskManager.from_user_config({})

