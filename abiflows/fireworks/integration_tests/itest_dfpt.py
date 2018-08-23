from __future__ import print_function, division, unicode_literals, absolute_import

import pytest
import os
import glob
import tempfile
import filecmp
import numpy.testing.utils as nptu

from abipy.abio.input_tags import *
from abipy.abio.factories import dfpt_from_gsinput
from abipy.core.testing import has_abinit
from fireworks.core.rocket_launcher import rapidfire
from abiflows.fireworks.workflows.abinit_workflows import DfptFWWorkflow
from abiflows.fireworks.utils.fw_utils import get_fw_by_task_index, load_abitask
from abiflows.core.testing import AbiflowsIntegrationTest, check_restart_task_type

ABINIT_VERSION = "8.6.1"

# pytestmark = [pytest.mark.skipif(not has_abinit(ABINIT_VERSION), reason="Abinit version {} is not in PATH".format(ABINIT_VERSION)),
#               pytest.mark.skipif(not has_fireworks(), reason="fireworks paackage is missing"),
#               pytest.mark.skipif(not has_mongodb(), reason="no connection to mongodb")]

pytestmark = pytest.mark.usefixtures("cleandb")


class ItestDfpt(AbiflowsIntegrationTest):

    def itest_dfpt_full(self, lp, fworker, tmpdir, input_scf_phonon_gan_low, use_autoparal, db_data):
        """
        Simple test of DteFWWorkflow with autoparal True and False.
        Doesn't run anaddb since anaddb does not support a DDB with third order perturbations along with
        other perturbations.
        Skips dte permutations.
        """

        # dte calculations only work with selected values of ixc
        input_scf_phonon_gan_low['ixc'] = 7
        dfpt_inputs = dfpt_from_gsinput(input_scf_phonon_gan_low, ph_ngqpt=[2, 2, 2], do_ddk=True, do_dde=True,
                                        do_strain=True, do_dte=True, skip_dte_permutations=True,
                                        ddk_tol = {"tolwfr": 1.0e-16}, dde_tol = {"tolvrs": 1.0e-7},
                                        strain_tol={"tolvrs": 1.0e-7}, ph_tol={"tolvrs": 1.0e-7})

        wf = DfptFWWorkflow(input_scf_phonon_gan_low, ddk_inp = dfpt_inputs.filter_by_tags(DDK),
                           dde_inp = dfpt_inputs.filter_by_tags(DDE), strain_inp=dfpt_inputs.filter_by_tags(STRAIN),
                           ph_inp = dfpt_inputs.filter_by_tags(PH_Q_PERT), dte_inp = dfpt_inputs.filter_by_tags(DTE),
                            nscf_inp=dfpt_inputs.filter_by_tags(NSCF),initialization_info={"kppa": 100},
                            autoparal=use_autoparal)

        wf.add_mongoengine_db_insertion(db_data)
        wf.add_final_cleanup(["WFK"])

        scf_fw_id = wf.scf_fw.fw_id
        old_new = wf.add_to_db(lpad=lp)
        scf_fw_id = old_new[scf_fw_id]

        rapidfire(lp, fworker, m_dir=str(tmpdir))

        wf = lp.get_wf_by_fw_id(scf_fw_id)

        assert wf.state == "COMPLETED"

        # check the effect of the final cleanup
        scf_task = load_abitask(get_fw_by_task_index(wf, "scf", index=1))

        assert len(glob.glob(os.path.join(scf_task.outdir.path, "*_WFK"))) == 0
        assert len(glob.glob(os.path.join(scf_task.outdir.path, "*_DEN"))) == 1
        assert len(glob.glob(os.path.join(scf_task.tmpdir.path, "*"))) == 0
        assert len(glob.glob(os.path.join(scf_task.indir.path, "*"))) == 0

    def itest_dfpt_anaddb_ph(self, lp, fworker, tmpdir, input_scf_phonon_gan_low, db_data):
        """
        Simple test of DteFWWorkflow with autoparal True and False.
        Skips dte permutations.
        """

        dfpt_inputs = dfpt_from_gsinput(input_scf_phonon_gan_low, ph_ngqpt=[2, 2, 2], do_ddk=True, do_dde=True,
                                        do_strain=True, do_dte=False,
                                        ddk_tol = {"tolwfr": 1.0e-16}, dde_tol = {"tolvrs": 1.0e-7},
                                        strain_tol={"tolvrs": 1.0e-7}, ph_tol={"tolvrs": 1.0e-7})

        wf = DfptFWWorkflow(input_scf_phonon_gan_low, ddk_inp = dfpt_inputs.filter_by_tags(DDK),
                           dde_inp = dfpt_inputs.filter_by_tags(DDE), strain_inp=dfpt_inputs.filter_by_tags(STRAIN),
                           ph_inp = dfpt_inputs.filter_by_tags(PH_Q_PERT), dte_inp = dfpt_inputs.filter_by_tags(DTE),
                            nscf_inp=dfpt_inputs.filter_by_tags(NSCF),initialization_info={"kppa": 100},
                            autoparal=False)

        wf.add_anaddb_dfpt_fw(input_scf_phonon_gan_low.structure, ph_ngqpt=[2, 2, 2], nqsmall=2, ndivsm=3)
        wf.add_mongoengine_db_insertion(db_data)
        wf.add_final_cleanup(["WFK"])

        scf_fw_id = wf.scf_fw.fw_id
        old_new = wf.add_to_db(lpad=lp)
        scf_fw_id = old_new[scf_fw_id]

        rapidfire(lp, fworker, m_dir=str(tmpdir))

        wf = lp.get_wf_by_fw_id(scf_fw_id)

        assert wf.state == "COMPLETED"

        scf_task = load_abitask(get_fw_by_task_index(wf, "scf", index=1))

        # check the save in the DB
        from abiflows.database.mongoengine.abinit_results import DfptResult
        with db_data.switch_collection(DfptResult) as DteResult:
            results = DteResult.objects()
            assert len(results) == 1
            r = results[0]

            assert r.abinit_input.structure.to_mgobj() == input_scf_phonon_gan_low.structure
            assert r.abinit_output.structure.to_mgobj() == input_scf_phonon_gan_low.structure
            assert r.abinit_input.ecut == input_scf_phonon_gan_low['ecut']
            assert r.abinit_input.kppa == 100
            nptu.assert_array_equal(r.abinit_input.gs_input.to_mgobj()['ngkpt'], input_scf_phonon_gan_low['ngkpt'])

            ana_task = load_abitask(get_fw_by_task_index(wf, "anaddb", index=None))

            with tempfile.NamedTemporaryFile(mode="wb") as db_file:
                db_file.write(r.abinit_output.anaddb_nc.read())
                db_file.seek(0)
                assert filecmp.cmp(ana_task.anaddb_nc_path, db_file.name)

            mrgddb_task = load_abitask(get_fw_by_task_index(wf, "mrgddb", index=None))

            with tempfile.NamedTemporaryFile(mode="wb") as db_file:
                db_file.write(r.abinit_output.ddb.read())
                db_file.seek(0)
                assert filecmp.cmp(mrgddb_task.merged_ddb_path, db_file.name)

        if self.check_numerical_values:
            with scf_task.open_gsr() as gsr:
                assert gsr.energy == pytest.approx(-680.402255069, rel=0.005)

            ana_task = load_abitask(get_fw_by_task_index(wf, "anaddb", index=None))
            with ana_task.open_anaddbnc() as ananc:
                assert float(ananc.eps0[0,0]) == pytest.approx(64.8276774889143, rel=0.15)

                e = ananc.elastic_data
                if has_abinit("8.9.3"):
                    assert float(e.elastic_relaxed[0,0,0,0]) == pytest.approx(41.230540749230556, rel=0.15)

    def itest_dfpt_anaddb_dte(self, lp, fworker, tmpdir, input_scf_phonon_gan_low, db_data):
        """
        Simple test of DteFWWorkflow with dte.
        Does not run anaddb due to problems when dealing with DDB file containing 3rd order derivatives and
        other terms.
        """

        # dte calculations only work with selected values of ixc
        input_scf_phonon_gan_low['ixc'] = 7
        dfpt_inputs = dfpt_from_gsinput(input_scf_phonon_gan_low, ph_ngqpt=[1,1,1], do_ddk=True, do_dde=True,
                                        do_strain=False, do_dte=True, skip_dte_permutations=True,
                                        ddk_tol = {"tolwfr": 1.0e-16}, dde_tol = {"tolvrs": 1.0e-7},
                                        strain_tol={"tolvrs": 1.0e-7}, ph_tol={"tolvrs": 1.0e-7})

        wf = DfptFWWorkflow(input_scf_phonon_gan_low, ddk_inp = dfpt_inputs.filter_by_tags(DDK),
                            dde_inp = dfpt_inputs.filter_by_tags(DDE), strain_inp=dfpt_inputs.filter_by_tags(STRAIN),
                            ph_inp = dfpt_inputs.filter_by_tags(PH_Q_PERT), dte_inp = dfpt_inputs.filter_by_tags(DTE),
                            nscf_inp=dfpt_inputs.filter_by_tags(NSCF),initialization_info={"kppa": 100},
                            autoparal=False)

        wf.add_anaddb_dfpt_fw(input_scf_phonon_gan_low.structure)
        wf.add_mongoengine_db_insertion(db_data)
        wf.add_final_cleanup(additional_spec={"test_spec":1})

        scf_fw_id = wf.scf_fw.fw_id
        old_new = wf.add_to_db(lpad=lp)
        scf_fw_id = old_new[scf_fw_id]

        rapidfire(lp, fworker, m_dir=str(tmpdir))

        wf = lp.get_wf_by_fw_id(scf_fw_id)

        assert wf.state == "COMPLETED"

        if self.check_numerical_values:
            ana_task = load_abitask(get_fw_by_task_index(wf, "anaddb", index=None))
            with ana_task.open_anaddbnc() as ananc:
                assert float(ananc.dchide[0,0,2]) == pytest.approx(-1.69328765210, rel=0.15)

    def itest_not_converged(self, lp, fworker, tmpdir, input_scf_phonon_si_low):
        """
        Tests the missed convergence and restart for the Strain task. Other restarts are checked in
        the PhononWorkflow tests.
        """

        # only strain perturbations included
        dfpt_inputs = dfpt_from_gsinput(input_scf_phonon_si_low, do_ddk=False, do_dde=False,
                                        do_strain=True, do_dte=False, skip_dte_permutations=True,
                                        strain_tol={"tolvrs": 1.0e-7} )
        dfpt_inputs.set_vars(nstep=3)
        wf = DfptFWWorkflow(input_scf_phonon_si_low, ddk_inp=None, dde_inp=None, ph_inp=None,
                            strain_inp=dfpt_inputs.filter_by_tags(STRAIN), dte_inp=None,
                            nscf_inp=None, initialization_info=None, autoparal=False)

        scf_id = wf.scf_fw.fw_id
        old_new = wf.add_to_db(lpad=lp)
        scf_id = old_new[scf_id]

        # run the scf
        rapidfire(lp, fworker, m_dir=str(tmpdir), nlaunches=1)
        # pause all the remaining workflow and reignite the task types one by one to check the restart
        lp.pause_wf(scf_id)

        # strain
        check_restart_task_type(lp, fworker, tmpdir, scf_id, "strain_pert_0")
