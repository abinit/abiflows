from __future__ import print_function, division, unicode_literals, absolute_import

import pytest
import os
import glob
import numpy as np

from fireworks.core.rocket_launcher import rapidfire
from abiflows.fireworks.workflows.abinit_workflows import NscfFWWorkflow
from abiflows.fireworks.tasks.abinit_tasks import NscfFWTask
from abiflows.fireworks.utils.fw_utils import load_abitask, get_fw_by_task_index
from abiflows.core.testing import AbiflowsIntegrationTest

ABINIT_VERSION = "8.6.1"

# pytestmark = [pytest.mark.skipif(not has_abinit(ABINIT_VERSION), reason="Abinit version {} is not in PATH".format(ABINIT_VERSION)),
#               pytest.mark.skipif(not has_fireworks(), reason="fireworks paackage is missing"),
#               pytest.mark.skipif(not has_mongodb(), reason="no connection to mongodb")]

pytestmark = pytest.mark.usefixtures("cleandb")

class ItestNscf(AbiflowsIntegrationTest):

    def itest_nscf_wf(self, lp, fworker, tmpdir, input_ebands_si_low, use_autoparal):
        """
        Simple test of NscfFWWorkflow with autoparal True and False.
        """
        wf = NscfFWWorkflow(*input_ebands_si_low, autoparal=False)

        wf.add_final_cleanup(["WFK"])

        scf_fw_id = wf.scf_fw.fw_id
        nscf_fw_id = wf.nscf_fw.fw_id
        old_new = wf.add_to_db(lpad=lp)
        scf_fw_id = old_new[scf_fw_id]
        nscf_fw_id = old_new[nscf_fw_id]

        rapidfire(lp, fworker, m_dir=str(tmpdir))

        wf = lp.get_wf_by_fw_id(scf_fw_id)

        assert wf.state == "COMPLETED"

        # check the effect of the final cleanup
        scf_task = load_abitask(get_fw_by_task_index(wf, "scf", index=1))

        assert len(glob.glob(os.path.join(scf_task.outdir.path, "*_WFK"))) == 0
        assert len(glob.glob(os.path.join(scf_task.outdir.path, "*_DEN"))) == 1
        assert len(glob.glob(os.path.join(scf_task.tmpdir.path, "*"))) == 0
        assert len(glob.glob(os.path.join(scf_task.indir.path, "*"))) == 0

        if self.check_numerical_values:
            scf_task = load_abitask(lp.get_fw_by_id(scf_fw_id))

            with scf_task.open_gsr() as scf_gsr:
                assert scf_gsr.energy == pytest.approx(-241.239839134, rel=0.01)

            nscf_task = load_abitask(lp.get_fw_by_id(nscf_fw_id))
            with nscf_task.open_gsr() as nscf_gsr:
                assert np.allclose((-6.2581504, 5.5974646, 5.5974646), nscf_gsr.ebands.eigens[0, 0, :3], rtol=0.1)


    def itest_not_converged(self, lp, fworker, tmpdir, input_ebands_si_low):
        """
        Tests the running of the NscfFWWorkflow with non convergence and restart
        """
        input_ebands_si_low[1].set_vars(nstep=5)

        wf = NscfFWWorkflow(*input_ebands_si_low, autoparal=False)

        scf_fw_id = wf.scf_fw.fw_id
        nscf_fw_id = wf.nscf_fw.fw_id
        old_new = wf.add_to_db(lpad=lp)
        scf_fw_id = old_new[scf_fw_id]
        nscf_fw_id = old_new[nscf_fw_id]

        rapidfire(lp, fworker, m_dir=str(tmpdir), nlaunches=2)

        nscf_fw = lp.get_fw_by_id(nscf_fw_id)

        assert nscf_fw.state == "COMPLETED"

        launch = nscf_fw.launches[-1]

        assert any(event.yaml_tag == NscfFWTask.CRITICAL_EVENTS[0].yaml_tag for event in launch.action.stored_data['report'])

        links = lp.get_wf_by_fw_id(scf_fw_id).links

        assert nscf_fw_id in links and len(links[nscf_fw_id]) == 1

        fw_child_id = links[nscf_fw_id][0]
        fw_child = lp.get_fw_by_id(fw_child_id)

        assert fw_child.state == "READY"

        rapidfire(lp, fworker, m_dir=str(tmpdir))

        fw_child = lp.get_fw_by_id(fw_child_id)

        assert fw_child.state == "COMPLETED"

        wf = lp.get_wf_by_fw_id(scf_fw_id)

        assert wf.state == "COMPLETED"

        if self.check_numerical_values:
            scf_task = load_abitask(lp.get_fw_by_id(scf_fw_id))
            with scf_task.open_gsr() as scf_gsr:
                assert scf_gsr.energy == pytest.approx(-241.239839134, rel=0.01)

            last_nscf_task = load_abitask(get_fw_by_task_index(wf, "nscf", index=-1))
            with last_nscf_task.open_gsr() as nscf_gsr:
                assert np.allclose((-6.2581504, 5.5974646, 5.5974646), nscf_gsr.ebands.eigens[0, 0, :3], rtol=0.1)



