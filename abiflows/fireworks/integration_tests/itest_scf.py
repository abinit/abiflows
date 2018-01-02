from __future__ import print_function, division, unicode_literals, absolute_import

import pytest
import os
import glob

from fireworks.core.rocket_launcher import rapidfire
from abipy.electrons.gsr import GsrFile
from abiflows.fireworks.workflows.abinit_workflows import InputFWWorkflow, ScfFWWorkflow
from abiflows.fireworks.tasks.abinit_tasks import ScfFWTask, OUTDIR_NAME, INDIR_NAME, TMPDIR_NAME
from abiflows.fireworks.utils.fw_utils import load_abitask, get_fw_by_task_index
from abiflows.core.testing import AbiflowsIntegrationTest


ABINIT_VERSION = "8.6.1"

# pytestmark = [pytest.mark.skipif(not has_abinit(ABINIT_VERSION), reason="Abinit version {} is not in PATH".format(ABINIT_VERSION)),
#               pytest.mark.skipif(not has_fireworks(), reason="fireworks paackage is missing"),
#               pytest.mark.skipif(not has_mongodb(), reason="no connection to mongodb")]

pytestmark = pytest.mark.usefixtures("cleandb")


class ItestScf(AbiflowsIntegrationTest):

    def itest_input_wf(self, lp, fworker, tmpdir, input_scf_si_low, use_autoparal):
        """
        Tests a simple scf run with the InputFWWorkflow
        """
        wf = InputFWWorkflow(input_scf_si_low, task_type=ScfFWTask, autoparal=use_autoparal)

        scf_fw_id = wf.fw.fw_id
        old_new = wf.add_to_db(lpad=lp)
        scf_fw_id = old_new[scf_fw_id]

        rapidfire(lp, fworker, m_dir=str(tmpdir))

        fw = lp.get_fw_by_id(scf_fw_id)

        assert fw.state == "COMPLETED"

        wf = lp.get_wf_by_fw_id(scf_fw_id)

        assert wf.state == "COMPLETED"
        assert len(wf.leaf_fw_ids) == 1

        if self.check_numerical_values:
            task = load_abitask(lp.get_fw_by_id(wf.leaf_fw_ids[0]))

            with task.open_gsr() as gsr:
                assert gsr.energy == pytest.approx(-241.239839134, rel=0.05)


    def itest_scf_wf(self, lp, fworker, tmpdir, input_scf_si_low, use_autoparal):
        """
        Tests a simple scf run with the ScfFWWorkflow
        """
        wf = ScfFWWorkflow(input_scf_si_low, autoparal=use_autoparal)

        wf.add_final_cleanup(["WFK"])

        scf_fw_id = wf.scf_fw.fw_id
        old_new = wf.add_to_db(lpad=lp)
        scf_fw_id = old_new[scf_fw_id]

        rapidfire(lp, fworker, m_dir=str(tmpdir))

        fw = lp.get_fw_by_id(scf_fw_id)

        assert fw.state == "COMPLETED"

        wf = lp.get_wf_by_fw_id(scf_fw_id)

        assert wf.state == "COMPLETED"

        # check the effect of the final cleanup
        scf_task = load_abitask(get_fw_by_task_index(wf, "scf", index=1))

        assert len(glob.glob(os.path.join(scf_task.outdir.path, "*_WFK"))) == 0
        assert len(glob.glob(os.path.join(scf_task.outdir.path, "*_DEN"))) == 1
        assert len(glob.glob(os.path.join(scf_task.tmpdir.path, "*"))) == 0
        assert len(glob.glob(os.path.join(scf_task.indir.path, "*"))) == 0

        if self.check_numerical_values:
            with scf_task.open_gsr() as gsr:
                assert gsr.energy == pytest.approx(-241.239839134, rel=0.05)

    def itest_not_converged(self, lp, fworker, tmpdir, input_scf_si_low):
        """
        Tests the ScfFWWorkflow with a calculation that does not converge on the first run.
        The calculation is continued until convergence is reached
        """

        input_scf_si_low.set_vars(nstep=3)

        wf = ScfFWWorkflow(input_scf_si_low, autoparal=False)

        scf_fw_id = wf.scf_fw.fw_id
        old_new = wf.add_to_db(lpad=lp)
        scf_fw_id = old_new[scf_fw_id]

        rapidfire(lp, fworker, m_dir=str(tmpdir), nlaunches=1)

        fw = lp.get_fw_by_id(scf_fw_id)

        assert fw.state == "COMPLETED"

        launch = fw.launches[-1]

        assert any(event.yaml_tag == ScfFWTask.CRITICAL_EVENTS[0].yaml_tag
                   for event in launch.action.stored_data['report'])

        links = lp.get_wf_by_fw_id(scf_fw_id).links

        assert scf_fw_id in links and len(links[scf_fw_id]) == 1

        fw_child_id = links[scf_fw_id][0]
        fw_child = lp.get_fw_by_id(fw_child_id)

        assert fw_child.state == "READY"

        # run all the workflow to check that the calculation converged
        rapidfire(lp, fworker, m_dir=str(tmpdir))

        fw_child = lp.get_fw_by_id(fw_child_id)

        assert fw_child.state == "COMPLETED"

        wf = lp.get_wf_by_fw_id(scf_fw_id)

        assert wf.state == "COMPLETED"

        if self.check_numerical_values:
            task = load_abitask(lp.get_fw_by_id(wf.leaf_fw_ids[0]))

            with task.open_gsr() as gsr:
                assert gsr.energy == pytest.approx(-241.239839133, rel=0.05)


    def itest_not_converged_fizzled(self, lp, fworker, tmpdir, input_scf_si_low):
        """
        Tests the ScfFWWorkflow with a calculation that does not converge within the maximum number
        of restarts allowed. The job will fizzle. The maximum number of restarts is increased and
        the job can continue. This works as a generic test for the maximum number of restarts.
        """

        input_scf_si_low.set_vars(nstep=1)

        # set the maximum number of restarts to 3 through the spec
        # following FWs should have the same option since the spec is passed down when there is a detour
        wf = ScfFWWorkflow(input_scf_si_low, autoparal=False, spec={'fw_policy':{'max_restarts':3}})

        scf_fw_id = wf.scf_fw.fw_id
        old_new = wf.add_to_db(lpad=lp)
        scf_fw_id = old_new[scf_fw_id]

        rapidfire(lp, fworker, m_dir=str(tmpdir))

        wf = lp.get_wf_by_fw_id(scf_fw_id)

        assert wf.state == "FIZZLED"
        assert len(wf.id_fw) == 4

        # find the fw_id of the last FW
        fw_ids = list(wf.id_fw.keys())
        for father_id, children_ids in wf.links.items():
            if children_ids:
                fw_ids.remove(father_id)
        # there should be only one FW without children
        assert len(fw_ids) == 1
        last_id = fw_ids[0]

        # increase the number of restarts allowed
        lp.update_spec([last_id], {'fw_policy.max_restarts': 10})
        lp.rerun_fw(last_id)

        rapidfire(lp, fworker, m_dir=str(tmpdir), nlaunches=5)

        wf = lp.get_wf_by_fw_id(scf_fw_id)

        assert len(wf.id_fw) == 9
        assert wf.state == 'RUNNING'






