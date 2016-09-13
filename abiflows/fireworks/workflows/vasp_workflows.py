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
from abiflows.fireworks.tasks.vasp_tasks_src import MPRelaxTaskHelper
from abiflows.fireworks.tasks.vasp_tasks_src import GenerateNEBRelaxationTask
from abiflows.fireworks.tasks.vasp_sets import MPNEBSet
from abiflows.fireworks.tasks.utility_tasks import DatabaseInsertTask

from pymatgen.io.vasp.sets import MPRelaxSet

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

    def add_db_insert(self, mongo_database, insertion_data=None,
                      criteria=None):
        if insertion_data is None:
            insertion_data = {'structure': 'get_final_structure'}
        spec = self.set_short_single_core_to_spec()
        spec['mongo_database'] = mongo_database.as_dict()
        spec['_add_launchpad_and_fw_id'] = True
        insert_fw = Firework([DatabaseInsertTask(insertion_data=insertion_data, criteria=criteria)],
                             spec=spec,
                             name=(self.wf.name + "_insert"))

        append_fw_to_wf(insert_fw, self.wf)

    def add_metadata(self, structure=None, additional_metadata={}):
        metadata = dict(wf_type = self.__class__.__name__)
        if structure:
            composition = structure.composition
            metadata['nsites'] = len(structure)
            metadata['elements'] = [el.symbol for el in composition.elements]
            metadata['reduced_formula'] = compossrc_fwsition.reduced_formula

        metadata.update(additional_metadata)

        self.wf.metadata.update(metadata)

    def add_spec_to_all_fws(self, spec):
        for fw in self.wf.fws:
            fw.spec.update(spec)

    def set_preserve_fworker(self):
        self.add_spec_to_all_fws(dict(_preserve_fworker=True))


class MPRelaxFWWorkflowSRC(AbstractFWWorkflow):
    workflow_class = 'MPRelaxFWWorkflowSRC'
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
        task_helper = MPRelaxTaskHelper()
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
                                     'workflow_module': self.workflow_module},
                           name='MPRelaxFWWorkflowSRC')

    @classmethod
    def from_structure(cls, structure, spec):
        vis = MPRelaxSet(structure=structure)
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
                if task_index.task_type == 'MPRelaxVasp':
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
        helper = MPRelaxTaskHelper()
        helper.set_task(mytask)
        helper.task.setup_rundir(last_launch.launch_dir)

        structure = helper.get_final_structure()

        return {'structure': structure.as_dict()}


class MPNEBRelaxFWWorkflowSRC(AbstractFWWorkflow):
    workflow_class = 'MPNEBRelaxFWWorkflowSRC'
    workflow_module = 'abiflows.fireworks.workflows.vasp_workflows'

    def __init__(self, neb_vasp_input_set, spec, neb_terminals, relax_terminals=True, n_insert=1, n_nebs=3,
                 relax_vasp_input_set=None, initial_neb_structures=None):
        user_incar_settings = {'NPAR': 4, 'ISIF': 0, 'SIGMA': 0.2, 'ISMEAR': 0}
        if n_nebs < 1:
            raise ValueError('Minimum one NEB ...')
        if relax_terminals and initial_neb_structures is not None:
            raise ValueError('Cannot relax terminals and start from initial NEB structures')
        # Initializes fws list and links_dict
        fws = []
        links_dict = {}

        if 'additional_controllers' in spec:
            additional_controllers = spec['additional_controllers']
            spec.pop('additional_controllers')
        else:
            additional_controllers = [WalltimeController(), MemoryController(), VaspXMLValidatorController()]

        # Control procedure
        control_procedure = ControlProcedure(controllers=additional_controllers)

        # First NEB
        gen_neb_spec = spec.copy()
        gen_neb_spec['terminal_start'] = neb_terminals[0]
        gen_neb_spec['terminal_end'] = neb_terminals[1]
        gen_neb_spec['structures'] = neb_terminals
        gen_neb_task = GenerateNEBRelaxationTask(n_insert=n_insert, user_incar_settings=user_incar_settings)
        gen_neb_fw = Firework([gen_neb_task], spec=gen_neb_spec, name='gen-neb1')
        fws.append(gen_neb_fw)

        if relax_terminals:
            # Start terminal
            relax_task_helper = MPRelaxTaskHelper()
            relax_task_type = 'MPRelaxVasp-start'
            vis_start = relax_vasp_input_set(neb_terminals[0], user_incar_settings=user_incar_settings)
            start_src_fws = createVaspSRCFireworks(vasp_input_set=vis_start, task_helper=relax_task_helper,
                                                   task_type=relax_task_type,
                                                   control_procedure=control_procedure,
                                                   custodian_handlers=[], max_restarts=10, src_cleaning=None,
                                                   task_index=None,
                                                   spec=None,
                                                   setup_spec_update=None, run_spec_update=None)
            fws.extend(start_src_fws['fws'])
            links_dict_update(links_dict=links_dict, links_update=start_src_fws['links_dict'])
            linkupdate = {start_src_fws['control_fw'].fw_id: gen_neb_fw.fw_id}
            links_dict_update(links_dict=links_dict,
                              links_update=linkupdate)

            # End terminal
            relax_task_type = 'MPRelaxVasp-end'
            vis_end = relax_vasp_input_set(neb_terminals[1], user_incar_settings=user_incar_settings)
            end_src_fws = createVaspSRCFireworks(vasp_input_set=vis_end, task_helper=relax_task_helper,
                                                 task_type=relax_task_type,
                                                 control_procedure=control_procedure,
                                                 custodian_handlers=[], max_restarts=10, src_cleaning=None,
                                                 task_index=None,
                                                 spec=None,
                                                 setup_spec_update=None, run_spec_update=None)
            fws.extend(end_src_fws['fws'])
            links_dict_update(links_dict=links_dict, links_update=end_src_fws['links_dict'])
            linkupdate = {end_src_fws['control_fw'].fw_id: gen_neb_fw.fw_id}
            links_dict_update(links_dict=links_dict,
                              links_update=linkupdate)

        if n_nebs > 1:
            for ineb in range(2, n_nebs+1):
                prev_gen_neb_fw = gen_neb_fw
                gen_neb_spec = spec.copy()
                gen_neb_spec['terminal_start'] = neb_terminals[0]
                gen_neb_spec['terminal_end'] = neb_terminals[1]
                gen_neb_spec['structures'] = neb_terminals
                gen_neb_task = GenerateNEBRelaxationTask(n_insert=n_insert, user_incar_settings=user_incar_settings)
                gen_neb_fw = Firework([gen_neb_task], spec=gen_neb_spec, name='gen-neb{:d}'.format(ineb))
                fws.append(gen_neb_fw)
                linkupdate = {prev_gen_neb_fw.fw_id: gen_neb_fw.fw_id}
                links_dict_update(links_dict=links_dict,
                                  links_update=linkupdate)

        self.wf = Workflow(fireworks=fws, links_dict=links_dict,
                           metadata={'workflow_class': self.workflow_class,
                                     'workflow_module': self.workflow_module},
                           name="MPNEBRelaxFWWorkflowSRC")


    @classmethod
    def from_terminals(cls, neb_terminals, spec, relax_terminals=True, n_insert=1, n_nebs=3,
                       neb_vasp_input_set=MPNEBSet,
                       relax_vasp_input_set=MPRelaxSet):
        return cls(neb_vasp_input_set=neb_vasp_input_set, spec=spec, neb_terminals=neb_terminals, relax_terminals=True,
                   n_insert=n_insert, n_nebs=n_nebs, relax_vasp_input_set=relax_vasp_input_set,
                   initial_neb_structures=None)

    @classmethod
    def get_final_structure(cls, wf):
        pass