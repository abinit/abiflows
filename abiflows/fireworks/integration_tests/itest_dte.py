from __future__ import print_function, division, unicode_literals

import pytest

from abiflows.fireworks.workflows.abinit_workflows import DteFWWorkflow
from abipy.abio.input_tags import *
from abipy.abio.factories import dte_from_gsinput
from fireworks.core.rocket_launcher import rapidfire


ABINIT_VERSION = "8.6.1"

# pytestmark = [pytest.mark.skipif(not has_abinit(ABINIT_VERSION), reason="Abinit version {} is not in PATH".format(ABINIT_VERSION)),
#               pytest.mark.skipif(not has_fireworks(), reason="fireworks paackage is missing"),
#               pytest.mark.skipif(not has_mongodb(), reason="no connection to mongodb")]

# pytestmark = pytest.mark.usefixtures("cleandb")

class ItestNscf():

    def itest_dte_with_phonons(self, lp, fworker, tmpdir, input_scf_phonon_gan_low, use_autoparal):
        """
        Simple test of DteFWWorkflow with autoparal True and False.
        """

        # dte calculations only work with selected values of ixc
        input_scf_phonon_gan_low['ixc'] = 7
        dte_inputs = dte_from_gsinput(input_scf_phonon_gan_low, use_phonons=True, skip_dte_permutations=False,
                                      ph_tol={"tolvrs": 1.0e-7}, ddk_tol = {"tolwfr": 1.0e-16},
                                      dde_tol = {"tolvrs": 1.0e-7}, dte_tol = {"tolwfr": 1.0e-16})

        dte_inputs = dte_from_gsinput(input_scf_phonon_gan_low, use_phonons=False, skip_dte_permutations=True,
                                      ph_tol={"tolvrs": 1.0e-7}, ddk_tol = {"tolwfr": 1.0e-16},
                                      dde_tol = {"tolvrs": 1.0e-7}, dte_tol = {"tolwfr": 1.0e-16})

        wf = DteFWWorkflow(input_scf_phonon_gan_low, ddk_inp = dte_inputs.filter_by_tags(DDK),
                           dde_inp = dte_inputs.filter_by_tags(DDE), dte_inp = dte_inputs.filter_by_tags(DTE),
                           ph_inp = dte_inputs.filter_by_tags(PH_Q_PERT), autoparal=False)
        scf_fw_id = wf.scf_fw.fw_id
        old_new = wf.add_to_db(lpad=lp)
        scf_fw_id = old_new[scf_fw_id]

        rapidfire(lp, fworker, m_dir=str(tmpdir))

        wf = lp.get_wf_by_fw_id(scf_fw_id)

        assert wf.state == "COMPLETED"

    def itest_dte_skip_permutations(self, lp, fworker, tmpdir, input_scf_phonon_gan_low):
        """
        Simple test of DteFWWorkflow without phonons and skipping dte permutations
        """

        # dte calculations only work with selected values of ixc
        input_scf_phonon_gan_low['ixc'] = 7

        dte_inputs = dte_from_gsinput(input_scf_phonon_gan_low, use_phonons=False, skip_dte_permutations=True,
                                      ph_tol={"tolvrs": 1.0e-7}, ddk_tol = {"tolwfr": 1.0e-16},
                                      dde_tol = {"tolvrs": 1.0e-7}, dte_tol = {"tolwfr": 1.0e-16})

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