from __future__ import print_function, division, unicode_literals, absolute_import

import pytest
import os
import glob
import tempfile
import filecmp
import numpy.testing.utils as nptu

from abipy.abio.input_tags import *
from abipy.abio.factories import dte_from_gsinput
from fireworks.core.rocket_launcher import rapidfire
from abiflows.fireworks.workflows.abinit_workflows import DteFWWorkflow
from abiflows.fireworks.utils.fw_utils import get_fw_by_task_index, load_abitask
from abiflows.core.testing import AbiflowsIntegrationTest

ABINIT_VERSION = "8.6.1"

# pytestmark = [pytest.mark.skipif(not has_abinit(ABINIT_VERSION), reason="Abinit version {} is not in PATH".format(ABINIT_VERSION)),
#               pytest.mark.skipif(not has_fireworks(), reason="fireworks paackage is missing"),
#               pytest.mark.skipif(not has_mongodb(), reason="no connection to mongodb")]

pytestmark = pytest.mark.usefixtures("cleandb")


class ItestDte(AbiflowsIntegrationTest):

    def itest_dte_with_phonons(self, lp, fworker, tmpdir, input_scf_phonon_gan_low, use_autoparal, db_data):
        """
        Simple test of DteFWWorkflow with autoparal True and False.
        Skips dte permutations.
        """

        # dte calculations only work with selected values of ixc
        input_scf_phonon_gan_low['ixc'] = 7
        dte_inputs = dte_from_gsinput(input_scf_phonon_gan_low, use_phonons=True, skip_dte_permutations=True,
                                      ph_tol={"tolvrs": 1.0e-7}, ddk_tol = {"tolwfr": 1.0e-16},
                                      dde_tol = {"tolvrs": 1.0e-7})

        wf = DteFWWorkflow(input_scf_phonon_gan_low, ddk_inp = dte_inputs.filter_by_tags(DDK),
                           dde_inp = dte_inputs.filter_by_tags(DDE), dte_inp = dte_inputs.filter_by_tags(DTE),
                           ph_inp = dte_inputs.filter_by_tags(PH_Q_PERT), autoparal=use_autoparal,
                           initialization_info={"kppa": 100})

        wf.add_anaddb_dte_fw(input_scf_phonon_gan_low.structure, dieflag=1, nlflag=1)
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

        # check the save in the DB
        from abiflows.database.mongoengine.abinit_results import DteResult
        with db_data.switch_collection(DteResult) as DteResult:
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
                assert float(ananc.dchide[0,0,2]) == pytest.approx(-1.69328765210, rel=0.15)

    def itest_dte_skip_permutations(self, lp, fworker, tmpdir, input_scf_phonon_gan_low):
        """
        Simple test of DteFWWorkflow without phonons.
        """

        # dte calculations only work with selected values of ixc
        input_scf_phonon_gan_low['ixc'] = 7

        dte_inputs = dte_from_gsinput(input_scf_phonon_gan_low, use_phonons=False, skip_dte_permutations=False,
                                      ph_tol={"tolvrs": 1.0e-7}, ddk_tol = {"tolwfr": 1.0e-16},
                                      dde_tol = {"tolvrs": 1.0e-7})

        wf = DteFWWorkflow(input_scf_phonon_gan_low, ddk_inp = dte_inputs.filter_by_tags(DDK),
                           dde_inp = dte_inputs.filter_by_tags(DDE), dte_inp = dte_inputs.filter_by_tags(DTE),
                           ph_inp = dte_inputs.filter_by_tags(PH_Q_PERT), autoparal=False)

        wf.add_anaddb_dte_fw(input_scf_phonon_gan_low.structure, dieflag=2, nlflag=3, ramansr=0, alphon=0, prtmbm=0)

        scf_fw_id = wf.scf_fw.fw_id
        old_new = wf.add_to_db(lpad=lp)
        scf_fw_id = old_new[scf_fw_id]

        rapidfire(lp, fworker, m_dir=str(tmpdir))

        wf = lp.get_wf_by_fw_id(scf_fw_id)

        assert wf.state == "COMPLETED"

        if self.check_numerical_values:
            scf_task = load_abitask(get_fw_by_task_index(wf, "scf", index=1))
            with scf_task.open_gsr() as gsr:
                assert gsr.energy == pytest.approx(-680.402255069, rel=0.005)

            ana_task = load_abitask(get_fw_by_task_index(wf, "anaddb", index=None))
            with ana_task.open_anaddbnc() as ananc:
                assert float(ananc.dchide[0,0,2]) == pytest.approx(-1.69328765210, rel=0.15)
