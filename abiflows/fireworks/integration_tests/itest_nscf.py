from __future__ import print_function, division, unicode_literals

import pytest

from abiflows.fireworks.workflows.abinit_workflows import NscfFWWorkflow
from abiflows.fireworks.tasks.abinit_tasks import NscfFWTask
from fireworks.core.rocket_launcher import rapidfire


ABINIT_VERSION = "8.6.1"

# pytestmark = [pytest.mark.skipif(not has_abinit(ABINIT_VERSION), reason="Abinit version {} is not in PATH".format(ABINIT_VERSION)),
#               pytest.mark.skipif(not has_fireworks(), reason="fireworks paackage is missing"),
#               pytest.mark.skipif(not has_mongodb(), reason="no connection to mongodb")]

# pytestmark = pytest.mark.usefixtures("cleandb")

class ItestNscf():

    def itest_nscf(self, lp, fworker, tmpdir, input_ebands_si_low, use_autoparal):
        """
        Simple test of NscfFWWorkflow with autoparal True and False.
        """
        wf = NscfFWWorkflow(*input_ebands_si_low, autoparal=False)

        scf_fw_id = wf.scf_fw.fw_id
        nscf_fw_id = wf.nscf_fw.fw_id
        old_new = wf.add_to_db(lpad=lp)
        scf_fw_id = old_new[scf_fw_id]
        nscf_fw_id = old_new[nscf_fw_id]

        rapidfire(lp, fworker, m_dir=str(tmpdir))

        wf = lp.get_wf_by_fw_id(scf_fw_id)

        assert wf.state == "COMPLETED"

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




