from __future__ import print_function, division, unicode_literals

import pytest
import os
import unittest

from abiflows.fireworks.workflows.abinit_workflows import PhononFullFWWorkflow, PhononFWWorkflow
from abiflows.fireworks.tasks.abinit_tasks import RelaxFWTask
from abiflows.fireworks.utils.fw_utils import get_fw_by_task_index, load_abitask
from fireworks.core.rocket_launcher import rapidfire
from abipy.dynamics.hist import HistFile
from pymatgen.io.abinit.utils import Directory
import abiflows.fireworks.tasks.abinit_tasks as abinit_tasks
from abipy.abio.factories import phonons_from_gsinput, PhononsFromGsFactory


ABINIT_VERSION = "8.6.1"

# pytestmark = [pytest.mark.skipif(not has_abinit(ABINIT_VERSION), reason="Abinit version {} is not in PATH".format(ABINIT_VERSION)),
#               pytest.mark.skipif(not has_fireworks(), reason="fireworks paackage is missing"),
#               pytest.mark.skipif(not has_mongodb(), reason="no connection to mongodb")]

# pytestmark = pytest.mark.usefixtures("cleandb")

class ItestPhonon():

    def itest_phonon(self, lp, fworker, tmpdir, input_scf_phonon_si_low, use_autoparal):
        """
        Tests the complete running of PhononFullFWWorkflow and PhononFWWorkflow
        """

        # test at gamma
        ph_fac = PhononsFromGsFactory(qpoints=[[0,0,0]], ph_tol = {"tolvrs": 1.0e-7}, ddk_tol = {"tolwfr": 1.0e-16},
                                      dde_tol = {"tolvrs": 1.0e-7}, wfq_tol = {"tolwfr": 1.0e-16})
        wf_gen = PhononFWWorkflow(input_scf_phonon_si_low, ph_fac, autoparal=use_autoparal)

        wf_gen.add_anaddb_ph_bs_fw(input_scf_phonon_si_low.structure, ph_ngqpt=[1,1,1], ndivsm=2, nqsmall=2)

        scf_id = wf_gen.scf_fw.fw_id
        ph_generation_fw_id = wf_gen.ph_generation_fw.fw_id
        old_new = wf_gen.add_to_db(lpad=lp)
        scf_id = old_new[scf_id]
        ph_generation_fw_id = old_new[ph_generation_fw_id]

        # run all the workflow
        rapidfire(lp, fworker, m_dir=str(tmpdir))

        wf_gen = lp.get_wf_by_fw_id(scf_id)

        assert wf_gen.state == "COMPLETED"

        wf_full = PhononFullFWWorkflow(input_scf_phonon_si_low, ph_fac, autoparal=use_autoparal)

        scf_id = wf_full.scf_fw.fw_id
        old_new = wf_full.add_to_db(lpad=lp)
        scf_id = old_new[scf_id]

        # run all the workflow
        rapidfire(lp, fworker, m_dir=str(tmpdir))

        wf_full = lp.get_wf_by_fw_id(scf_id)

        assert wf_full.state == "COMPLETED"

        # the full workflow doesn't contain the generation FW and the anaddb, but should have the same amount of
        # perturbations.
        if use_autoparal:
            diff = 1
        else:
            diff = 2
        assert len(wf_full.id_fw) + diff == len(wf_gen.id_fw)

    def check_restart_task_type(self, lp, fworker, tmpdir, fw_id, task_tag):

        # resume the task for tag
        wf = lp.get_wf_by_fw_id(fw_id)
        fw = get_fw_by_task_index(wf, task_tag, index=1)
        assert fw is not None
        assert fw.state == "PAUSED"
        lp.resume_fw(fw.fw_id)

        # run the FW
        rapidfire(lp, fworker, m_dir=str(tmpdir), nlaunches=1)

        # the job should have a detour for the restart
        wf = lp.get_wf_by_fw_id(fw_id)
        fw = get_fw_by_task_index(wf, task_tag, index=2)
        assert fw is not None
        assert fw.state == "READY"

        # run all the following and check that the last is correctly completed (if convergence is not achieved
        # the final state should be FIZZLED)
        rapidfire(lp, fworker, m_dir=str(tmpdir))

        wf = lp.get_wf_by_fw_id(fw_id)
        fw = get_fw_by_task_index(wf, task_tag, index=None)

        assert fw.state == "COMPLETED"

    def itest_not_converged(self, lp, fworker, tmpdir, input_scf_phonon_si_low):
        """
        Tests the missed convergence and restart for all the different kinds of tasks
        """

        # set a point not belonging to the grid so to trigger the calculation of WFQ and gamma for the DDE and DDK
        ph_inp = phonons_from_gsinput(input_scf_phonon_si_low, qpoints=[[0, 0, 0], [0.11111, 0.22222, 0.33333]],
                                      ph_tol = {"tolvrs": 1.0e-7}, ddk_tol = {"tolwfr": 1.0e-16},
                                      dde_tol = {"tolvrs": 1.0e-7}, wfq_tol = {"tolwfr": 1.0e-16})
        ph_inp.set_vars(nstep=3)
        wf_full = PhononFullFWWorkflow(input_scf_phonon_si_low, ph_inp, autoparal=False)

        scf_id = wf_full.scf_fw.fw_id
        old_new = wf_full.add_to_db(lpad=lp)
        scf_id = old_new[scf_id]

        # run the scf
        rapidfire(lp, fworker, m_dir=str(tmpdir), nlaunches=1)

        # pause all the remaining workflow and reignite the task types one by one to check the restart
        lp.pause_wf(scf_id)

        # DDK
        self.check_restart_task_type(lp, fworker, tmpdir, scf_id, "ddk_0")

        # reignite and run the other DDK to get to the DDE
        wf = lp.get_wf_by_fw_id(scf_id)
        lp.resume_fw(get_fw_by_task_index(wf, "ddk_1", index=1).fw_id)
        lp.resume_fw(get_fw_by_task_index(wf, "ddk_2", index=1).fw_id)
        rapidfire(lp, fworker, m_dir=str(tmpdir))

        # DDE
        self.check_restart_task_type(lp, fworker, tmpdir, scf_id, "dde_0")

        # NSCF
        self.check_restart_task_type(lp, fworker, tmpdir, scf_id, "nscf_wfq_0")

        # phonon
        self.check_restart_task_type(lp, fworker, tmpdir, scf_id, "phonon_0")

        # don't run the wf until the end to save time. Other tests covered that.
        wf = lp.get_wf_by_fw_id(scf_id)
        assert wf.state == "PAUSED"