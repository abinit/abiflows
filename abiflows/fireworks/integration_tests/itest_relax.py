from __future__ import print_function, division, unicode_literals, absolute_import

import pytest
import os
import glob
import unittest
import tempfile
import filecmp
import numpy.testing.utils as nptu
import numpy as np

from fireworks.core.rocket_launcher import rapidfire
from abipy.dynamics.hist import HistFile
from abipy.flowtk.events import DilatmxError
from abiflows.fireworks.workflows.abinit_workflows import RelaxFWWorkflow
from abiflows.fireworks.tasks.abinit_tasks import RelaxFWTask
from abiflows.fireworks.utils.fw_utils import get_fw_by_task_index,load_abitask,get_last_completed_launch
from abiflows.core.testing import AbiflowsIntegrationTest


ABINIT_VERSION = "8.6.1"

# pytestmark = [pytest.mark.skipif(not has_abinit(ABINIT_VERSION), reason="Abinit version {} is not in PATH".format(ABINIT_VERSION)),
#               pytest.mark.skipif(not has_fireworks(), reason="fireworks paackage is missing"),
#               pytest.mark.skipif(not has_mongodb(), reason="no connection to mongodb")]

pytestmark = pytest.mark.usefixtures("cleandb")

class ItestRelax(AbiflowsIntegrationTest):

    def itest_relax_wf(self, lp, fworker, tmpdir, inputs_relax_si_low, use_autoparal, db_data):
        """
        Tests the basic functionality of a RelaxFWWorkflow with autoparal True and False.
        """
        wf = RelaxFWWorkflow(*inputs_relax_si_low, autoparal=use_autoparal,
                             initialization_info={"kppa": 100})

        wf.add_mongoengine_db_insertion(db_data)
        wf.add_final_cleanup(["WFK"])

        initial_ion_structure = inputs_relax_si_low[0].structure

        ion_fw_id = wf.ion_fw.fw_id
        ioncell_fw_id = wf.ioncell_fw.fw_id
        old_new = wf.add_to_db(lpad=lp)
        ion_fw_id = old_new[ion_fw_id]
        ioncell_fw_id = old_new[ioncell_fw_id]

        rapidfire(lp, fworker, m_dir=str(tmpdir))

        wf = lp.get_wf_by_fw_id(ion_fw_id)

        assert wf.state == "COMPLETED"

        ioncell_fw = get_fw_by_task_index(wf, "ioncell", index=-1)
        ioncell_task = load_abitask(ioncell_fw)

        ioncell_hist_path = ioncell_task.outdir.has_abiext("HIST")

        with HistFile(ioncell_hist_path) as hist:
            initial_ioncell_structure = hist.structures[0]

        assert initial_ion_structure != initial_ioncell_structure

        # check the effect of the final cleanup
        assert len(glob.glob(os.path.join(ioncell_task.outdir.path, "*_WFK"))) == 0
        assert len(glob.glob(os.path.join(ioncell_task.outdir.path, "*_DEN"))) > 0
        assert len(glob.glob(os.path.join(ioncell_task.tmpdir.path, "*"))) == 0
        assert len(glob.glob(os.path.join(ioncell_task.indir.path, "*"))) == 0

        # check the result in the DB
        from abiflows.database.mongoengine.abinit_results import RelaxResult
        with db_data.switch_collection(RelaxResult) as RelaxResult:
            results = RelaxResult.objects()
            assert len(results) == 1
            r = results[0]

            # test input structure
            assert r.abinit_input.structure.to_mgobj() == initial_ion_structure
            # test output structure
            # remove site properties, otherwise the "cartesian_forces" won't match due to the presence of a
            # list instead of an array in the deserialization
            db_structure = r.abinit_output.structure.to_mgobj()
            for s in db_structure:
                s._properties = {}
            hist_structure = hist.structures[-1]
            for s in hist_structure:
                s._properties = {}
            assert db_structure == hist_structure
            assert r.abinit_input.ecut == inputs_relax_si_low[0]['ecut']
            assert r.abinit_input.kppa == 100
            nptu.assert_array_equal(r.abinit_input.last_input.to_mgobj()['ngkpt'], inputs_relax_si_low[0]['ngkpt'])

            with tempfile.NamedTemporaryFile(mode="wb") as db_file:
                db_file.write(r.abinit_output.gsr.read())
                db_file.seek(0)
                assert filecmp.cmp(ioncell_task.gsr_path, db_file.name)

        if self.check_numerical_values:
            with ioncell_task.open_gsr() as gsr:
                assert gsr.energy == pytest.approx(-240.28203726305696, rel=0.01)
                assert np.allclose((3.8101419256822333, 3.8101444012342616, 3.8101434297177068),
                                   gsr.structure.lattice.abc, rtol=0.05)

    def itest_uncoverged(self, lp, fworker, tmpdir, inputs_relax_si_low):
        """
        Testing restart when the ionic convercence is not reached
        """
        inputs_relax_si_low[0]['ntime']=3
        wf = RelaxFWWorkflow(*inputs_relax_si_low, autoparal=False)
        initial_ion_structure = inputs_relax_si_low[0].structure

        ion_fw_id = wf.ion_fw.fw_id
        ioncell_fw_id = wf.ioncell_fw.fw_id
        old_new = wf.add_to_db(lpad=lp)
        ion_fw_id = old_new[ion_fw_id]
        ioncell_fw_id = old_new[ioncell_fw_id]

        rapidfire(lp, fworker, m_dir=str(tmpdir), nlaunches=1)

        ion_fw = lp.get_fw_by_id(ion_fw_id)
        ioncell_fw = lp.get_fw_by_id(ioncell_fw_id)

        assert ion_fw.state == "COMPLETED"
        assert ioncell_fw.state == "WAITING"

        launch = ion_fw.launches[-1]

        assert any(event.yaml_tag == RelaxFWTask.CRITICAL_EVENTS[0].yaml_tag for event in launch.action.stored_data['report'])

        links_ion = lp.get_wf_by_fw_id(ion_fw_id).links[ion_fw_id]

        # there should be an additional child (the detour)
        assert len(links_ion) == 2

        links_ion.remove(ioncell_fw_id)

        fw_detour_id = links_ion[0]

        # run the detour
        rapidfire(lp, fworker, m_dir=str(tmpdir))

        fw_detour = lp.get_fw_by_id(fw_detour_id)

        assert fw_detour.state == "COMPLETED"

        restart_structure = fw_detour.spec['_tasks'][0].abiinput.structure

        wf = lp.get_wf_by_fw_id(ion_fw_id)

        assert wf.state == "COMPLETED"

        # check that the structure has been updated when restarting
        assert initial_ion_structure != restart_structure

        if self.check_numerical_values:
            last_ioncell_task = load_abitask(get_fw_by_task_index(wf, "ioncell", index=-1))
            with last_ioncell_task.open_gsr() as gsr:
                assert gsr.energy == pytest.approx(-240.28203726305696, rel=0.01)
                assert gsr.structure.lattice.abc == pytest.approx(
                    np.array((3.8101428225862084, 3.810143911539674, 3.8101432797789698)), rel=0.05)

    def itest_dilatmx(self, lp, fworker, tmpdir, inputs_relax_si_low):
        """
        Test the workflow with a target dilatmx
        """

        # set the dilatmx to 1.05 to keep the change independt on the generation of the input
        inputs_relax_si_low[1]['dilatmx'] = 1.05

        wf = RelaxFWWorkflow(*inputs_relax_si_low, autoparal=False, target_dilatmx=1.03)
        initial_ion_structure = inputs_relax_si_low[0].structure

        ion_fw_id = wf.ion_fw.fw_id
        ioncell_fw_id = wf.ioncell_fw.fw_id
        old_new = wf.add_to_db(lpad=lp)
        ion_fw_id = old_new[ion_fw_id]
        ioncell_fw_id = old_new[ioncell_fw_id]

        rapidfire(lp, fworker, m_dir=str(tmpdir), nlaunches=2)

        ion_fw = lp.get_fw_by_id(ion_fw_id)
        ioncell_fw = lp.get_fw_by_id(ioncell_fw_id)

        assert ion_fw.state == "COMPLETED"
        assert ioncell_fw.state == "COMPLETED"

        launch = ioncell_fw.launches[-1]

        links_ioncell = lp.get_wf_by_fw_id(ioncell_fw_id).links[ioncell_fw_id]

        # there should be an additional child (the detour)
        assert len(links_ioncell) == 1

        fw_detour_id = links_ioncell[0]

        # run the detour with lowered dilatmx
        rapidfire(lp, fworker, m_dir=str(tmpdir))

        fw_detour = lp.get_fw_by_id(fw_detour_id)

        assert fw_detour.state == "COMPLETED"

        detour_abiinput = fw_detour.spec['_tasks'][0].abiinput

        assert detour_abiinput['dilatmx'] == 1.03

        restart_structure = detour_abiinput.structure

        # check that the structure has been updated when restarting
        assert initial_ion_structure != restart_structure

        wf = lp.get_wf_by_fw_id(ion_fw_id)

        assert wf.state == "COMPLETED"

        # check that the structure has been updated when restarting
        assert initial_ion_structure != restart_structure

        if self.check_numerical_values:
            last_ioncell_task = load_abitask(get_fw_by_task_index(wf, "ioncell", index=-1))
            with last_ioncell_task.open_gsr() as gsr:
                assert gsr.structure.lattice.abc == pytest.approx(
                    np.array((3.8101419255677951, 3.8101444011173897, 3.8101434296150889)), rel=0.05)

    def itest_dilatmx_error(self, lp, fworker, tmpdir, inputs_relax_si_low, db_data):
        """
        Test the workflow when a dilatmx error shows up.
        Also tests the skip_ion option of RelaxFWWorkflow
        """

        # set the dilatmx to a small value, so that the dilatmx error will show up
        initial_dilatmx = 1.001
        inputs_relax_si_low[1]['dilatmx'] = initial_dilatmx

        # also test the skip_ion
        wf = RelaxFWWorkflow(*inputs_relax_si_low, autoparal=False, skip_ion=True)
        wf.add_mongoengine_db_insertion(db_data)

        initial_ion_structure = inputs_relax_si_low[0].structure

        ioncell_fw_id = wf.ioncell_fw.fw_id
        old_new = wf.add_to_db(lpad=lp)
        ioncell_fw_id = old_new[ioncell_fw_id]

        rapidfire(lp, fworker, m_dir=str(tmpdir), nlaunches=1)

        ioncell_fw = lp.get_fw_by_id(ioncell_fw_id)

        assert ioncell_fw.state == "COMPLETED"

        launch = ioncell_fw.launches[-1]

        assert any(event.yaml_tag == DilatmxError.yaml_tag for event in launch.action.stored_data['report'])

        links_ioncell = lp.get_wf_by_fw_id(ioncell_fw_id).links[ioncell_fw_id]

        # there should be an additional child (the detour)
        assert len(links_ioncell) == 2

        # run the detour restarting froom previous structure
        rapidfire(lp, fworker, m_dir=str(tmpdir), nlaunches=1)

        wf = lp.get_wf_by_fw_id(ioncell_fw_id)

        fw_detour = get_fw_by_task_index(wf, "ioncell", index=2)

        assert fw_detour.state == "COMPLETED"

        detour_abiinput = fw_detour.spec['_tasks'][0].abiinput

        assert detour_abiinput['dilatmx'] == initial_dilatmx

        restart_structure = detour_abiinput.structure

        # check that the structure has been updated when restarting
        assert initial_ion_structure != restart_structure

        # complete the wf. Just check that the saving without the ion tasks completes without error
        rapidfire(lp, fworker, m_dir=str(tmpdir))

        wf = lp.get_wf_by_fw_id(ioncell_fw_id)

        assert wf.state == "COMPLETED"