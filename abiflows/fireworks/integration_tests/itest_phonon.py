from __future__ import print_function, division, unicode_literals, absolute_import

import pytest
import os
import glob
import unittest
import tempfile
import filecmp
import numpy.testing.utils as nptu
import abipy.data as abidata

from abipy.abio.factories import phonons_from_gsinput, PhononsFromGsFactory
from abipy.flowtk.tasks import TaskManager
from fireworks.core.rocket_launcher import rapidfire
from abiflows.fireworks.workflows.abinit_workflows import PhononFullFWWorkflow, PhononFWWorkflow
from abiflows.fireworks.utils.fw_utils import get_fw_by_task_index, load_abitask
from abiflows.core.testing import AbiflowsIntegrationTest, check_restart_task_type


ABINIT_VERSION = "8.6.1"

# pytestmark = [pytest.mark.skipif(not has_abinit(ABINIT_VERSION), reason="Abinit version {} is not in PATH".format(ABINIT_VERSION)),
#               pytest.mark.skipif(not has_fireworks(), reason="fireworks paackage is missing"),
#               pytest.mark.skipif(not has_mongodb(), reason="no connection to mongodb")]

# pytestmark = pytest.mark.usefixtures("cleandb")

class ItestPhonon(AbiflowsIntegrationTest):

    def itest_phonon_wf(self, lp, fworker, tmpdir, input_scf_phonon_si_low, use_autoparal, db_data):
        """
        Tests the complete running of PhononFullFWWorkflow and PhononFWWorkflow
        """

        # test at gamma. Pass a custom manager, to check proper serialization
        manager_path = os.path.join(abidata.dirpath, 'managers', "travis_manager.yml")
        ph_fac = PhononsFromGsFactory(qpoints=[[0,0,0]], ph_tol = {"tolvrs": 1.0e-7}, ddk_tol = {"tolwfr": 1.0e-16},
                                      dde_tol = {"tolvrs": 1.0e-7}, wfq_tol = {"tolwfr": 1.0e-16},
                                      manager=TaskManager.from_file(manager_path))

        # first run the phonon workflow with generation task
        wf_gen = PhononFWWorkflow(input_scf_phonon_si_low, ph_fac, autoparal=use_autoparal,
                                  initialization_info={"ngqpt": [1,1,1], "kppa": 100})

        wf_gen.add_anaddb_ph_bs_fw(input_scf_phonon_si_low.structure, ph_ngqpt=[1,1,1], ndivsm=2, nqsmall=2)
        wf_gen.add_mongoengine_db_insertion(db_data)
        wf_gen.add_final_cleanup(["WFK"])

        scf_id = wf_gen.scf_fw.fw_id
        ph_generation_fw_id = wf_gen.ph_generation_fw.fw_id
        old_new = wf_gen.add_to_db(lpad=lp)
        scf_id = old_new[scf_id]
        ph_generation_fw_id = old_new[ph_generation_fw_id]

        # run all the workflow
        rapidfire(lp, fworker, m_dir=str(tmpdir))

        wf_gen = lp.get_wf_by_fw_id(scf_id)

        assert wf_gen.state == "COMPLETED"

        ph_task = load_abitask(get_fw_by_task_index(wf_gen, "phonon_0", index=-1))

        # check the effect of the final cleanup
        assert len(glob.glob(os.path.join(ph_task.outdir.path, "*_WFK"))) == 0
        assert len(glob.glob(os.path.join(ph_task.outdir.path, "*_DEN1"))) > 0
        assert len(glob.glob(os.path.join(ph_task.tmpdir.path, "*"))) == 0
        assert len(glob.glob(os.path.join(ph_task.indir.path, "*"))) == 0

        # check the save in the DB
        from abiflows.database.mongoengine.abinit_results import PhononResult
        with db_data.switch_collection(PhononResult) as PhononResult:
            results = PhononResult.objects()
            assert len(results) == 1
            r = results[0]

            assert r.abinit_input.structure.to_mgobj() == input_scf_phonon_si_low.structure
            assert r.abinit_output.structure.to_mgobj() == input_scf_phonon_si_low.structure
            assert r.abinit_input.ecut == input_scf_phonon_si_low['ecut']
            assert r.abinit_input.kppa == 100
            nptu.assert_array_equal(r.abinit_input.gs_input.to_mgobj()['ngkpt'], input_scf_phonon_si_low['ngkpt'])
            nptu.assert_array_equal(r.abinit_input.ngqpt, [1,1,1])

            ana_task = load_abitask(get_fw_by_task_index(wf_gen, "anaddb", index=None))

            with tempfile.NamedTemporaryFile(mode="wb") as db_file:
                db_file.write(r.abinit_output.phonon_bs.read())
                db_file.seek(0)
                assert filecmp.cmp(ana_task.phbst_path, db_file.name)

            mrgddb_task = load_abitask(get_fw_by_task_index(wf_gen, "mrgddb", index=None))

            # read/write in binary for py3k compatibility with mongoengine
            with tempfile.NamedTemporaryFile(mode="wb") as db_file:
                db_file.write(r.abinit_output.ddb.read())
                db_file.seek(0)
                assert filecmp.cmp(mrgddb_task.merged_ddb_path, db_file.name)

        # then rerun a similar workflow, but completely generated at its creation
        wf_full = PhononFullFWWorkflow(input_scf_phonon_si_low, ph_fac, autoparal=use_autoparal)
        wf_full.add_anaddb_ph_bs_fw(input_scf_phonon_si_low.structure, ph_ngqpt=[1,1,1], ndivsm=2, nqsmall=2)
        wf_full.add_mongoengine_db_insertion(db_data)

        scf_id = wf_full.scf_fw.fw_id
        old_new = wf_full.add_to_db(lpad=lp)
        scf_id = old_new[scf_id]

        # run all the workflow
        rapidfire(lp, fworker, m_dir=str(tmpdir))

        wf_full = lp.get_wf_by_fw_id(scf_id)

        assert wf_full.state == "COMPLETED"

        # the full workflow doesn't contain the generation FW and the cleanup tasks, but should have the same
        # amount of perturbations.
        if use_autoparal:
            diff = 1
        else:
            diff = 2
        assert len(wf_full.id_fw) + diff == len(wf_gen.id_fw)

        if self.check_numerical_values:
            gen_scf_task = load_abitask(get_fw_by_task_index(wf_gen, "scf", index=-1))
            with gen_scf_task.open_gsr() as gen_gsr:
                gen_energy = gen_gsr.energy
                assert gen_energy == pytest.approx(-240.264972012, rel=0.01)

            gen_ana_task = load_abitask(get_fw_by_task_index(wf_gen, "anaddb", index=None))
            with gen_ana_task.open_phbst() as gen_phbst:
                gen_phfreq = gen_phbst.phbands.phfreqs[0, 3]
                assert gen_phfreq == pytest.approx(0.06029885, rel=0.1)

            full_scf_task = load_abitask(get_fw_by_task_index(wf_gen, "scf", index=-1))
            with full_scf_task.open_gsr() as full_gsr:
                full_energy = full_gsr.energy
                assert full_energy == pytest.approx(-240.264972012, rel=0.01)

            full_ana_task = load_abitask(get_fw_by_task_index(wf_gen, "anaddb", index=None))
            with full_ana_task.open_phbst() as full_phbst:
                full_phfreqs = full_phbst.phbands.phfreqs[0, 3]
                assert full_phfreqs == pytest.approx(0.06029885, rel=0.1)

            assert gen_energy == pytest.approx(full_energy, rel=1e-6)
            assert gen_phfreq == pytest.approx(full_phfreqs, rel=1e-6)

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
        check_restart_task_type(lp, fworker, tmpdir, scf_id, "ddk_0")

        # reignite and run the other DDK to get to the DDE
        wf = lp.get_wf_by_fw_id(scf_id)
        lp.resume_fw(get_fw_by_task_index(wf, "ddk_1", index=1).fw_id)
        lp.resume_fw(get_fw_by_task_index(wf, "ddk_2", index=1).fw_id)
        rapidfire(lp, fworker, m_dir=str(tmpdir))

        # DDE
        check_restart_task_type(lp, fworker, tmpdir, scf_id, "dde_0")

        # NSCF
        check_restart_task_type(lp, fworker, tmpdir, scf_id, "nscf_wfq_0")

        # phonon
        check_restart_task_type(lp, fworker, tmpdir, scf_id, "phonon_0")

        # don't run the wf until the end to save time. Other tests covered that.
        wf = lp.get_wf_by_fw_id(scf_id)
        assert wf.state == "PAUSED"

    def itest_phonon_wfq_wf(self, lp, fworker, tmpdir, input_scf_phonon_si_low, db_data):
        """
        Tests the PhononFullFWWorkflow and PhononFWWorkflow
        """
        qpt = [[0.1111,0.2222,0.3333]]

        # test at gamma. Pass a custom manager, to check proper serialization
        manager_path = os.path.join(abidata.dirpath, 'managers', "travis_manager.yml")
        ph_fac = PhononsFromGsFactory(qpoints=qpt, ph_tol = {"tolvrs": 1.0e-7},
                                      wfq_tol = {"tolwfr": 1.0e-16}, with_ddk=False, with_dde=False,
                                      manager=TaskManager.from_file(manager_path))

        # first run the phonon workflow with generation task
        wf_gen = PhononFWWorkflow(input_scf_phonon_si_low, ph_fac, autoparal=False,
                                  initialization_info={"qpoints": qpt, "kppa": 100})

        wf_gen.add_mongoengine_db_insertion(db_data)

        scf_id = wf_gen.scf_fw.fw_id
        old_new = wf_gen.add_to_db(lpad=lp)
        scf_id = old_new[scf_id]

        # run all the workflow
        rapidfire(lp, fworker, m_dir=str(tmpdir))

        wf_gen = lp.get_wf_by_fw_id(scf_id)

        assert wf_gen.state == "COMPLETED"

        # then rerun a similar workflow, but completely generated at its creation
        wf_full = PhononFullFWWorkflow(input_scf_phonon_si_low, ph_fac, autoparal=False)

        scf_id = wf_full.scf_fw.fw_id
        old_new = wf_full.add_to_db(lpad=lp)
        scf_id = old_new[scf_id]

        # run all the workflow
        rapidfire(lp, fworker, m_dir=str(tmpdir))

        wf_full = lp.get_wf_by_fw_id(scf_id)

        assert wf_full.state == "COMPLETED"
