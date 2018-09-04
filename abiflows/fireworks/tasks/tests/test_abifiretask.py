# coding: utf-8
from __future__ import print_function, division, unicode_literals, absolute_import

import os
import abipy.data as abidata
import abipy.abilab as abilab
import abiflows.fireworks.tasks.abinit_tasks as abinit_tasks
import abiflows.fireworks.utils.fw_utils

from abipy.abio.factories import *
from abipy.abio.factories import ScfForPhononsFactory
from abipy.abio.inputs import AnaddbInput
from abiflows.core.testing import AbiflowsTest
from abiflows.fireworks.tasks.tests import mock_objects
from fireworks import FWAction


mock = AbiflowsTest.get_mock_module()


test_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                        "test_files", "fw_task_managers")


class TestAbiFireTask(AbiflowsTest):

    def setUp(self):
        self.si_structure = abilab.Structure.from_file(abidata.cif_file("si.cif"))
        self.si_scf_input = ebands_input(self.si_structure, abidata.pseudos("14si.pspnc"), ecut=2, kppa=10).split_datasets()[0]
        self.si_scf_factory = ScfForPhononsFactory(self.si_structure, abidata.pseudos("14si.pspnc"), ecut=2, kppa=10)

    def test_AbiFireTask_basic(self):
        task = abinit_tasks.AbiFireTask(self.si_scf_input)
        task.to_dict()
        self.assertMSONable(self.si_scf_input)
        self.assertFwSerializable(task)

    def test_AbiFireTask_methods(self):
        task = abinit_tasks.AbiFireTask(self.si_scf_input)

        assert task.get_fw_task_manager({'ftm_file': os.path.join(test_dir, "fw_manager_ok.yaml")})

        ftm_path = os.path.join(test_dir, "fw_manager_ok.yaml")
        ftm = abiflows.fireworks.utils.fw_utils.FWTaskManager.from_file(ftm_path)
        # remove the task manager to make the function fail
        ftm.task_manager = None
        with self.assertRaises(abinit_tasks.InitializationError):
            task.run_autoparal(self.si_scf_input, '.', ftm)

        ftm_path = os.path.join(test_dir, "fw_manager_walltime_command.yaml")
        task.is_autoparal = None
        task.setup_task({'ftm_file': ftm_path})
        self.assertEqual(task.walltime, 1000)


    def test_ScfFireTask(self):
        task = abinit_tasks.ScfFWTask(self.si_scf_input)
        task.to_dict()
        self.assertFwSerializable(task)

        task = abinit_tasks.ScfFWTask(self.si_scf_factory)
        task.to_dict()
        self.assertFwSerializable(task)

    def test_AnaDdbAbinitTask(self):
        ana_inp = AnaddbInput.phbands_and_dos(self.si_structure, ngqpt=[4,4,4], nqsmall=10)
        task = abinit_tasks.AnaDdbAbinitTask(ana_inp)
        task.to_dict()
        self.assertFwSerializable(task)


class TestTaskAnalysis(AbiflowsTest):
    def setUp(self):
        self.si_structure = abilab.Structure.from_file(abidata.cif_file("si.cif"))
        self.si_scf_input = ebands_input(self.si_structure, abidata.pseudos("14si.pspnc"), ecut=2, kppa=10).split_datasets()[0]

    @mock.patch.object(abinit_tasks.AbiFireTask, 'get_event_report')
    def test_scf_unconverged(self, report):
        scf_task = abinit_tasks.ScfFWTask(self.si_scf_input)
        ftm_path = os.path.join(test_dir, "fw_manager_ok.yaml")
        scf_task.ftm = abiflows.fireworks.utils.fw_utils.FWTaskManager.from_file(ftm_path)

        report.return_value = mock_objects.report_ScfConvergenceWarning()

        with mock.patch.object(abinit_tasks.AbiFireTask, 'prepare_restart',
                               return_value=(False, mock_objects.fake_fw, {})) as pr:
            fake_spec = {'test': 1}
            action = scf_task.task_analysis(fake_spec)
            pr.assert_called_once_with(fake_spec)
            self.assertIsInstance(action, FWAction)

            scf_task.restart_info = abinit_tasks.RestartInfo(
                previous_dir='.', num_restarts=scf_task.ftm.fw_policy.max_restarts)
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


class TestErrorClasses(AbiflowsTest):

    def setUp(self):
        self.si_structure = abilab.Structure.from_file(abidata.cif_file("si.cif"))
        self.si_scf_input = ebands_input(self.si_structure, abidata.pseudos("14si.pspnc"), ecut=2, kppa=10).split_datasets()[0]

    @mock.patch.object(abinit_tasks.AbiFireTask, 'get_event_report')
    def test_abinit_runtime_error(self, report):
        err = abinit_tasks.AbinitRuntimeError(msg="test error", num_errors=5)
        err.to_dict()
        new_err = abinit_tasks.AbinitRuntimeError.from_dict(err.as_dict())

        scf_task = abinit_tasks.ScfFWTask(self.si_scf_input)

        report.return_value = mock_objects.report_AbinitError()

        fake_spec = {'test': 1}

        try:
            # set the returncode to avoid logging problems
            scf_task.returncode = 10
            scf_task.task_analysis(fake_spec)
        except abinit_tasks.AbinitRuntimeError as e:
            e.to_dict()
            assert e.num_errors == 1


class TestRestartInfo(AbiflowsTest):

    def test_restart_info(self):
        ri = abinit_tasks.RestartInfo("/path/to/dir", reset=False, num_restarts=4)
        self.assertMSONable(ri)
        ri.prev_indir
        ri.prev_outdir