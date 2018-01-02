# coding: utf-8
"""
Firework workflows
"""
from __future__ import print_function, division, unicode_literals, absolute_import

import logging
import sys
import abc
import os
import re
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
from abiflows.fireworks.utils.fw_utils import set_short_single_core_to_spec

from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.analysis.transition_state import NEBAnalysis

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
            metadata['reduced_formula'] = composition.reduced_formula

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
    def from_structure(cls, structure, spec, user_incar_settings=None):
        vis = MPRelaxSet(structure=structure, user_incar_settings=user_incar_settings)
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
                 relax_vasp_input_set=None, initial_neb_structures=None, climbing_image=True):
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
        if relax_terminals:
            gen_neb_spec['terminal_start'] = None
            gen_neb_spec['terminal_end'] = None
        else:
            gen_neb_spec['terminal_start'] = neb_terminals[0]
            gen_neb_spec['terminal_end'] = neb_terminals[1]

        # gen_neb_spec['structures'] = neb_terminals
        gen_neb_spec = set_short_single_core_to_spec(gen_neb_spec)

        terminal_start_relax_task_type = 'MPRelaxVasp-start'
        terminal_end_relax_task_type = 'MPRelaxVasp-end'
        terminal_start_task_type = None
        terminal_end_task_type = None
        if relax_terminals:
            terminal_start_task_type = terminal_start_relax_task_type
            terminal_end_task_type = terminal_end_relax_task_type
        gen_neb_task = GenerateNEBRelaxationTask(n_insert=n_insert, user_incar_settings=user_incar_settings,
                                                 climbing_image=climbing_image, task_index='neb1',
                                                 terminal_start_task_type=terminal_start_task_type,
                                                 terminal_end_task_type=terminal_end_task_type)
        gen_neb_fw = Firework([gen_neb_task], spec=gen_neb_spec, name='gen-neb1')
        fws.append(gen_neb_fw)

        if relax_terminals:
            # Start terminal
            relax_task_helper = MPRelaxTaskHelper()
            vis_start = relax_vasp_input_set(neb_terminals[0], user_incar_settings=user_incar_settings)
            start_src_fws = createVaspSRCFireworks(vasp_input_set=vis_start, task_helper=relax_task_helper,
                                                   task_type=terminal_start_relax_task_type,
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

            vis_end = relax_vasp_input_set(neb_terminals[1], user_incar_settings=user_incar_settings)
            end_src_fws = createVaspSRCFireworks(vasp_input_set=vis_end, task_helper=relax_task_helper,
                                                 task_type=terminal_end_relax_task_type,
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
                gen_neb_spec['structures'] = None
                gen_neb_spec = set_short_single_core_to_spec(gen_neb_spec)
                gen_neb_task = GenerateNEBRelaxationTask(n_insert=n_insert, user_incar_settings=user_incar_settings,
                                                         climbing_image=climbing_image,
                                                         task_index='neb{:d}'.format(ineb),
                                                         prev_neb_task_type='neb{:d}'.format(ineb-1),
                                                         terminal_start_task_type=terminal_start_task_type,
                                                         terminal_end_task_type=terminal_end_task_type
                                                         )
                gen_neb_fw = Firework([gen_neb_task], spec=gen_neb_spec, name='gen-neb{:d}'.format(ineb))
                fws.append(gen_neb_fw)
                linkupdate = {prev_gen_neb_fw.fw_id: gen_neb_fw.fw_id}
                links_dict_update(links_dict=links_dict,
                                  links_update=linkupdate)
                if relax_terminals:
                    linkupdate = {start_src_fws['control_fw'].fw_id: gen_neb_fw.fw_id}
                    links_dict_update(links_dict=links_dict,
                                      links_update=linkupdate)
                    linkupdate = {end_src_fws['control_fw'].fw_id: gen_neb_fw.fw_id}
                    links_dict_update(links_dict=links_dict,
                                      links_update=linkupdate)

        if climbing_image:
            wfname = "MPcNEBRelaxFWWorkflowSRC"
        else:
            wfname = "MPcNEBRelaxFWWorkflowSRC"
        self.wf = Workflow(fireworks=fws, links_dict=links_dict,
                           metadata={'workflow_class': self.workflow_class,
                                     'workflow_module': self.workflow_module},
                           name=wfname)


    @classmethod
    def from_terminals(cls, neb_terminals, spec, relax_terminals=True, n_insert=1, n_nebs=3,
                       neb_vasp_input_set=MPNEBSet,
                       relax_vasp_input_set=MPRelaxSet):
        return cls(neb_vasp_input_set=neb_vasp_input_set, spec=spec, neb_terminals=neb_terminals, relax_terminals=True,
                   n_insert=n_insert, n_nebs=n_nebs, relax_vasp_input_set=relax_vasp_input_set,
                   initial_neb_structures=None)

    @classmethod
    def get_nebs_analysis(cls, wf):
        assert wf.metadata['workflow_class'] == cls.workflow_class
        assert wf.metadata['workflow_module'] == cls.workflow_module

        terminal_start_step_index = -1
        terminal_start_final_fw_id = None
        terminal_end_step_index = -1
        terminal_end_final_fw_id = None

        neb_tasks_step_index = {}
        neb_tasks_step_final_fw_id = {}

        for fw_id, fw in wf.id_fw.items():
            if 'SRC_task_index' in fw.spec:
                if fw.tasks[-1].src_type != 'run':
                    continue
                task_index = fw.spec['SRC_task_index']
                if re.match("\<neb\d\>", task_index.task_type):
                    if task_index.task_type not in neb_tasks_step_index:
                        neb_tasks_step_index[task_index.task_type] = task_index.index
                        neb_tasks_step_final_fw_id[task_index.task_type] = fw_id
                    else:
                        if task_index.index > neb_tasks_step_index[task_index.task_type]:
                            neb_tasks_step_index[task_index.task_type] = task_index.index
                            neb_tasks_step_final_fw_id[task_index.task_type] = fw_id
                elif task_index.task_type == 'MPRelaxVasp-start':
                    if task_index.index > terminal_start_step_index:
                        terminal_start_step_index = task_index.index
                        terminal_start_final_fw_id = fw_id
                elif task_index.task_type == 'MPRelaxVasp-end':
                    if task_index.index > terminal_end_step_index:
                        terminal_end_step_index = task_index.index
                        terminal_end_final_fw_id = fw_id
        if terminal_start_final_fw_id is None:
            raise RuntimeError('No terminal-start relaxation found ...')
        if terminal_end_final_fw_id is None:
            raise RuntimeError('No terminal-end relaxation found ...')
        if len(neb_tasks_step_index) == 0:
            raise RuntimeError('No NEB analysis found ...')
        # Terminal start dir
        terminal_start_fw = wf.id_fw[terminal_start_final_fw_id]
        terminal_start_last_launch = (terminal_start_fw.archived_launches + terminal_start_fw.launches)[-1]
        terminal_start_run_dir = terminal_start_last_launch.launch_dir
        # Terminal end dir
        terminal_end_fw = wf.id_fw[terminal_end_final_fw_id]
        terminal_end_last_launch = (terminal_end_fw.archived_launches + terminal_end_fw.launches)[-1]
        terminal_end_run_dir = terminal_end_last_launch.launch_dir
        nebs_analysis = []
        for ineb in range(1, len(neb_tasks_step_index)+1):
            neb_task = 'neb{:d}'.format(ineb)
            nebfw = wf.id_fw[neb_tasks_step_final_fw_id[neb_task]]
            last_launch = (nebfw.archived_launches + nebfw.launches)[-1]
            neb_analysis = NEBAnalysis.from_dir(last_launch.launch_dir,
                                                relaxation_dirs=(terminal_start_run_dir, terminal_end_run_dir))
            nebs_analysis.append(neb_analysis)
        return {'nebs_analysis': [neb_analysis.as_dict() for neb_analysis in nebs_analysis]}