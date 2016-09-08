# coding: utf-8
"""
Firework workflows
"""
from __future__ import print_function, division, unicode_literals

import logging
import sys

import abc
import os
import six
from fireworks.core.firework import Firework, Workflow
from fireworks.core.launchpad import LaunchPad

from abiflows.core.controllers import WalltimeController, MemoryController, VaspXMLValidatorController
from abiflows.core.mastermind_abc import ControlProcedure
from abiflows.fireworks.utils.fw_utils import append_fw_to_wf, get_short_single_core_spec, links_dict_update
from abiflows.fireworks.tasks.vasp_tasks_src import createVaspSRCFireworks
from abiflows.fireworks.tasks.vasp_tasks_src import MITRelaxTaskHelper

from pymatgen.io.vasp.sets import MITRelaxSet

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))



@six.add_metaclass(abc.ABCMeta)
class AbstractFWWorkflow(Workflow):
    """
    Abstract Workflow class.
    """

    def add_to_db(self, lpad=None):
        if not lpad:
            lpad = LaunchPad.auto_load()
        return lpad.add_wf(self.wf)

    def append_fw(self, fw, short_single_spec=False):
        if short_single_spec:
            fw.spec.update(self.set_short_single_core_to_spec())
        append_fw_to_wf(fw, self.wf)

    @staticmethod
    def set_short_single_core_to_spec(spec={}, master_mem_overhead=0):
        spec = dict(spec)

        qadapter_spec = get_short_single_core_spec(master_mem_overhead=master_mem_overhead)
        spec['mpi_ncpus'] = 1
        spec['_queueadapter'] = qadapter_spec
        return spec

    def add_metadata(self, structure=None, additional_metadata={}):
        metadata = dict(wf_type = self.__class__.__name__)
        if structure:
            composition = structure.composition
            metadata['nsites'] = len(structure)
            metadata['elements'] = [el.symbol for el in composition.elements]
            metadata['reduced_formula'] = composition.reduced_formula

        metadata.update(additional_metadata)

        self.wf.metadata.update(metadata)

    def add_spec_to_all_fws(self, spec):
        for fw in self.wf.fws:
            fw.spec.update(spec)

    def set_preserve_fworker(self):
        self.add_spec_to_all_fws(dict(_preserve_fworker=True))


class MITRelaxFWWorkflowSRC(AbstractFWWorkflow):
    workflow_class = 'MITRelaxFWWorkflowSRC'
    workflow_module = 'abiflows.fireworks.workflows.vasp_workflows'

    def __init__(self, vasp_input_set, spec):
        # Initializes fws list and links_dict
        fws = []
        links_dict = {}

        if 'additional_controllers' in spec:
            additional_controllers = spec['additional_controllers']
            spec.pop('additional_controllers')
        else:
            additional_controllers = [WalltimeController(), MemoryController(), VaspXMLValidatorController()]

        control_procedure = ControlProcedure(controllers=additional_controllers)
        task_helper = MITRelaxTaskHelper()
        task_type = task_helper.task_type
        src_fws = createVaspSRCFireworks(vasp_input_set=vasp_input_set, task_helper=task_helper, task_type=task_type,
                                         control_procedure=control_procedure,
                                         custodian_handlers=[], max_restarts=10, src_cleaning=None, task_index=None,
                                         spec=None,
                                         setup_spec_update=None, run_spec_update=None)

        fws.extend(src_fws['fws'])
        links_dict_update(links_dict=links_dict, links_update=src_fws['links_dict'])

        self.wf = Workflow(fireworks=fws, links_dict=links_dict,
                           metadata={'workflow_class': self.workflow_class,
                                     'workflow_module': self.workflow_module})

    @classmethod
    def from_structure(cls, structure, spec):
        vis = MITRelaxSet(structure=structure)
        return cls(vasp_input_set=vis, spec=spec)

    @classmethod
    def get_final_structure(cls, wf):
        assert wf.metadata['workflow_class'] == cls.workflow_class
        assert wf.metadata['workflow_module'] == cls.workflow_module
        step_index = -1
        final_fw_id = None
        for fw_id, fw in wf.id_fw.items():
            if 'SRC_task_index' in fw.spec:
                if fw.tasks[-1].src_type != 'run':
                    continue
                task_index = fw.spec['SRC_task_index']
                if task_index.task_type == 'MITRelaxVasp':
                    if task_index.index > step_index:
                        step_index = task_index.index
                        final_fw_id = fw_id
        if final_fw_id is None:
            raise RuntimeError('Final structure not found ...')
        myfw = wf.id_fw[final_fw_id]
        mytask = myfw.tasks[-1]
        last_launch = (myfw.archived_launches + myfw.launches)[-1]
        # myfw.tasks[-1].set_workdir(workdir=last_launch.launch_dir)
        # mytask.setup_rundir(last_launch.launch_dir, create_dirs=False)
        helper = MITRelaxTaskHelper()
        helper.set_task(mytask)
        helper.task.setup_rundir(last_launch.launch_dir)

        structure = helper.get_final_structure()

        return {'structure': structure.as_dict()}