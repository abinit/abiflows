from __future__ import print_function, division, unicode_literals

import pytest
import os
import unittest

from abiflows.fireworks.workflows.abinit_workflows import RelaxFWWorkflow
from abiflows.fireworks.tasks.abinit_tasks import RelaxFWTask
from abiflows.fireworks.utils.fw_utils import get_fw_by_task_index, load_abitask
from fireworks.core.rocket_launcher import rapidfire
from abipy.dynamics.hist import HistFile
from pymatgen.io.abinit.utils import Directory
from pymatgen.io.abinit.events import DilatmxError
import abiflows.fireworks.tasks.abinit_tasks as abinit_tasks


ABINIT_VERSION = "8.6.1"

# pytestmark = [pytest.mark.skipif(not has_abinit(ABINIT_VERSION), reason="Abinit version {} is not in PATH".format(ABINIT_VERSION)),
#               pytest.mark.skipif(not has_fireworks(), reason="fireworks paackage is missing"),
#               pytest.mark.skipif(not has_mongodb(), reason="no connection to mongodb")]

# pytestmark = pytest.mark.usefixtures("cleandb")

class ItestRelax():

    def itest_relax(self, lp, fworker, tmpdir, inputs_relax_si_low, use_autoparal):
        """
        Tests the basic functionality of a RelaxFWWorkflow with autoparal True and False.
        """
        wf = RelaxFWWorkflow(*inputs_relax_si_low, autoparal=use_autoparal)
        initial_ion_structure = inputs_relax_si_low[0].structure

        ion_fw_id = wf.ion_fw.fw_id
        ioncell_fw_id = wf.ioncell_fw.fw_id
        old_new = wf.add_to_db(lpad=lp)
        ion_fw_id = old_new[ion_fw_id]
        ioncell_fw_id = old_new[ioncell_fw_id]

        rapidfire(lp, fworker, m_dir=str(tmpdir))

        wf = lp.get_wf_by_fw_id(ion_fw_id)

        assert wf.state == "COMPLETED"

        ioncell_fw = get_fw_by_task_index(wf, "ioncell", index=None)

        ioncell_hist_path = Directory(os.path.join(ioncell_fw.launches[-1].launch_dir,
                                                   abinit_tasks.OUTDIR_NAME)).has_abiext("HIST")

        with HistFile(ioncell_hist_path) as hist:
            initial_ioncell_structure = hist.structures[0]

        assert initial_ion_structure != initial_ioncell_structure

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
        rapidfire(lp, fworker, m_dir=str(tmpdir), nlaunches=1)

        fw_detour = lp.get_fw_by_id(fw_detour_id)

        assert fw_detour.state == "COMPLETED"

        restart_structure = fw_detour.spec['_tasks'][0].abiinput.structure

        # check that the structure has been updated when restarting
        assert initial_ion_structure != restart_structure

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
        rapidfire(lp, fworker, m_dir=str(tmpdir), nlaunches=1)

        fw_detour = lp.get_fw_by_id(fw_detour_id)

        assert fw_detour.state == "COMPLETED"

        detour_abiinput = fw_detour.spec['_tasks'][0].abiinput

        assert detour_abiinput['dilatmx'] == 1.03

        restart_structure = detour_abiinput.structure

        # check that the structure has been updated when restarting
        assert initial_ion_structure != restart_structure

    def itest_dilatmx_error(self, lp, fworker, tmpdir, inputs_relax_si_low):
        """
        Test the workflow when a dilatmx error shows up.
        Also tests the skip_ion option of RelaxFWWorkflow
        """

        # set the dilatmx to a small value, so that the dilatmx error will show up
        initial_dilatmx = 1.001
        inputs_relax_si_low[1]['dilatmx'] = initial_dilatmx

        # also test the skip_ion
        wf = RelaxFWWorkflow(*inputs_relax_si_low, autoparal=False, skip_ion=True)
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
        assert len(links_ioncell) == 1

        fw_detour_id = links_ioncell[0]

        # run the detour with lowered dilatmx
        rapidfire(lp, fworker, m_dir=str(tmpdir), nlaunches=1)

        fw_detour = lp.get_fw_by_id(fw_detour_id)

        assert fw_detour.state == "COMPLETED"

        detour_abiinput = fw_detour.spec['_tasks'][0].abiinput

        assert detour_abiinput['dilatmx'] == initial_dilatmx

        restart_structure = detour_abiinput.structure

        # check that the structure has been updated when restarting
        assert initial_ion_structure != restart_structure