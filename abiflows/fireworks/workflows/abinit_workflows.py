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
from abipy.abio.factories import HybridOneShotFromGsFactory, ScfFactory, IoncellRelaxFromGsFactory
from abipy.abio.factories import PhononsFromGsFactory, ScfForPhononsFactory
from abipy.abio.factories import ion_ioncell_relax_input, scf_input
from abipy.abio.inputs import AbinitInput, AnaddbInput
from fireworks.core.firework import Firework, Workflow
from fireworks.core.launchpad import LaunchPad
from monty.serialization import loadfn

from abiflows.core.mastermind_abc import ControlProcedure
from abiflows.core.controllers import AbinitController, WalltimeController, MemoryController
from abiflows.fireworks.tasks.abinit_tasks import AbiFireTask, ScfFWTask, RelaxFWTask, NscfFWTask
from abiflows.fireworks.tasks.abinit_tasks_src import AbinitSetupTask, AbinitRunTask, AbinitControlTask
from abiflows.fireworks.tasks.abinit_tasks_src import ScfTaskHelper, NscfTaskHelper, DdkTaskHelper
from abiflows.fireworks.tasks.abinit_tasks_src import GeneratePiezoElasticFlowFWSRCAbinitTask
from abiflows.fireworks.tasks.abinit_tasks import HybridFWTask, RelaxDilatmxFWTask, GeneratePhononFlowFWAbinitTask
from abiflows.fireworks.tasks.abinit_tasks import GeneratePiezoElasticFlowFWAbinitTask
from abiflows.fireworks.tasks.abinit_tasks import AnaDdbAbinitTask, StrainPertTask, DdkTask, MergeDdbAbinitTask
from abiflows.fireworks.tasks.handlers import MemoryHandler, WalltimeHandler
from abiflows.fireworks.tasks.src_tasks_abc import createSRCFireworks
from abiflows.fireworks.tasks.utility_tasks import FinalCleanUpTask, DatabaseInsertTask
from abiflows.fireworks.tasks.utility_tasks import createSRCFireworksOld
from abiflows.fireworks.utils.fw_utils import append_fw_to_wf, get_short_single_core_spec, links_dict_update
from abiflows.fireworks.utils.fw_utils import set_short_single_core_to_spec

# logging.basicConfig()
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

    def add_final_cleanup(self, out_exts=["WFK"]):
        spec = self.set_short_single_core_to_spec()
        # high priority
        #TODO improve the handling of the priorities
        spec['_priority'] = 100
        cleanup_fw = Firework(FinalCleanUpTask(out_exts=out_exts), spec=spec,
                              name=(self.wf.name+"_cleanup")[:15])

        append_fw_to_wf(cleanup_fw, self.wf)

    def add_db_insert_and_cleanup(self, mongo_database, out_exts=["WFK"], insertion_data=None, criteria=None):
        if insertion_data is None:
            insertion_data = {'structure': 'get_final_structure_and_history'}
        spec = self.set_short_single_core_to_spec()
        spec['mongo_database'] = mongo_database.as_dict()
        insert_and_cleanup_fw = Firework([DatabaseInsertTask(insertion_data=insertion_data, criteria=criteria),
                                          FinalCleanUpTask(out_exts=out_exts)],
                                         spec=spec,
                                         name=(self.wf.name+"_insclnup")[:15])

        append_fw_to_wf(insert_and_cleanup_fw, self.wf)

    def add_anaddb_task(self, structure):
        spec = self.set_short_single_core_to_spec()
        anaddb_task = AnaDdbAbinitTask(AnaddbInput.piezo_elastic(structure))
        anaddb_fw = Firework([anaddb_task],
                             spec=spec,
                             name='anaddb')
        append_fw_to_wf(anaddb_fw, self.wf)

    def add_metadata(self, structure=None, additional_metadata={}):
        metadata = dict(wf_type = self.__class__.__name__)
        if structure:
            composition = structure.composition
            metadata['nsites'] = len(structure)
            metadata['elements'] = [el.symbol for el in composition.elements]
            metadata['reduced_formula'] = composition.reduced_formula

        metadata.update(additional_metadata)

        self.wf.metadata.update(metadata)

    def get_reduced_formula(self, input):
        structure = None
        try:
            if isinstance(input, AbinitInput):
                structure = input.structure
            elif 'structure' in input.kwargs:
                structure = input.kwargs['structure']
            else:
                structure = input.args[0]
        except Exception as e:
            logger.warning("Couldn't get the structure from the input: {} {}".format(e.__class__.__name__, e.message))

        return structure.composition.reduced_formula if structure else ""

    def add_spec_to_all_fws(self, spec):
        for fw in self.wf.fws:
            fw.spec.update(spec)

    def set_preserve_fworker(self):
        self.add_spec_to_all_fws(dict(_preserve_fworker=True))


class InputFWWorkflow(AbstractFWWorkflow):
    def __init__(self, abiinput, task_type=AbiFireTask, autoparal=False, spec={}, initialization_info={}):
        abitask = task_type(abiinput, is_autoparal=autoparal)

        spec = dict(spec)
        spec['initialization_info'] = initialization_info
        if autoparal:
            spec = self.set_short_single_core_to_spec(spec)

        self.fw = Firework(abitask, spec=spec)

        self.wf = Workflow([self.fw])
        # Workflow.__init__([self.fw])


class ScfFWWorkflow(AbstractFWWorkflow):
    def __init__(self, abiinput, autoparal=False, spec={}, initialization_info={}):
        abitask = ScfFWTask(abiinput, is_autoparal=autoparal)

        spec = dict(spec)
        spec['initialization_info'] = initialization_info
        if autoparal:
            spec = self.set_short_single_core_to_spec(spec)

        self.scf_fw = Firework(abitask, spec=spec)

        self.wf = Workflow([self.scf_fw])

    @classmethod
    def from_factory(cls, structure, pseudos, kppa=None, ecut=None, pawecutdg=None, nband=None, accuracy="normal",
                     spin_mode="polarized", smearing="fermi_dirac:0.1 eV", charge=0.0, scf_algorithm=None,
                     shift_mode="Monkhorst-Pack", extra_abivars={}, decorators=[], autoparal=False, spec={}):
        abiinput = scf_input(structure, pseudos, kppa=kppa, ecut=ecut, pawecutdg=pawecutdg, nband=nband,
                             accuracy=accuracy, spin_mode=spin_mode, smearing=smearing, charge=charge,
                             scf_algorithm=scf_algorithm, shift_mode=shift_mode)
        abiinput.set_vars(extra_abivars)
        for d in decorators:
            d(abiinput)

        return cls(abiinput, autoparal=autoparal, spec=spec)


class ScfFWWorkflowSRC(AbstractFWWorkflow):

    workflow_class = 'ScfFWWorkflowSRC'
    workflow_module = 'abiflows.fireworks.workflows.abinit_workflows'

    def __init__(self, abiinput, spec={}, initialization_info={}):

        scf_helper = ScfTaskHelper()
        control_procedure = ControlProcedure(controllers=[AbinitController.from_helper(scf_helper),
                                                          WalltimeController(), MemoryController()])
        setup_task = AbinitSetupTask(abiinput=abiinput, task_helper=scf_helper)
        run_task = AbinitRunTask(control_procedure=control_procedure, task_helper=scf_helper)
        control_task = AbinitControlTask(control_procedure=control_procedure, task_helper=scf_helper)

        scf_fws = createSRCFireworks(setup_task=setup_task, run_task=run_task, control_task=control_task)

        self.wf = Workflow(fireworks=scf_fws['fws'], links_dict=scf_fws['links_dict'],
                           metadata={'workflow_class': self.workflow_class,
                                     'workflow_module': self.workflow_module})

    @classmethod
    def from_factory(cls, structure, pseudos, kppa=None, ecut=None, pawecutdg=None, nband=None, accuracy="normal",
                     spin_mode="polarized", smearing="fermi_dirac:0.1 eV", charge=0.0, scf_algorithm=None,
                     shift_mode="Monkhorst-Pack", extra_abivars={}, decorators=[], autoparal=False, spec={}):
        abiinput = scf_input(structure, pseudos, kppa=kppa, ecut=ecut, pawecutdg=pawecutdg, nband=nband,
                             accuracy=accuracy, spin_mode=spin_mode, smearing=smearing, charge=charge,
                             scf_algorithm=scf_algorithm, shift_mode=shift_mode)
        abiinput.set_vars(extra_abivars)
        for d in decorators:
            d(abiinput)

        return cls(abiinput, spec=spec)


class RelaxFWWorkflow(AbstractFWWorkflow):
    workflow_class = 'RelaxFWWorkflow'
    workflow_module = 'abiflows.fireworks.workflows.abinit_workflows'

    def __init__(self, ion_input, ioncell_input, autoparal=False, spec={}, initialization_info={}, target_dilatmx=None):
        start_task_index = 1
        spec = dict(spec)
        spec['initialization_info'] = initialization_info
        if autoparal:
            spec = self.set_short_single_core_to_spec(spec)
            start_task_index = 'autoparal'

        spec['wf_task_index'] = 'ion_' + str(start_task_index)
        ion_task = RelaxFWTask(ion_input, is_autoparal=autoparal)
        self.ion_fw = Firework(ion_task, spec=spec)

        spec['wf_task_index'] = 'ioncell_' + str(start_task_index)
        if target_dilatmx:
            ioncell_task = RelaxDilatmxFWTask(ioncell_input, is_autoparal=autoparal, target_dilatmx=target_dilatmx,
                                              deps={ion_task.task_type: '@structure'})
        else:
            ioncell_task = RelaxFWTask(ioncell_input, is_autoparal=autoparal, deps={ion_task.task_type: '@structure'})

        self.ioncell_fw = Firework(ioncell_task, spec=spec)

        self.wf = Workflow([self.ion_fw, self.ioncell_fw], {self.ion_fw: [self.ioncell_fw]},
                           metadata={'workflow_class': self.workflow_class,
                                     'workflow_module': self.workflow_module})

    @classmethod
    def get_final_structure_and_history(cls, wf):
        assert wf.metadata['workflow_class'] == cls.workflow_class
        assert wf.metadata['workflow_module'] == cls.workflow_module
        ioncell = -1
        final_fw_id = None
        for fw_id, fw in wf.id_fw.items():
            if 'wf_task_index' in fw.spec and fw.spec['wf_task_index'][:8] == 'ioncell_':
                try:
                    this_ioncell =  int(fw.spec['wf_task_index'].split('_')[-1])
                except ValueError:
                    # skip if the index is not an int
                    continue
                if this_ioncell > ioncell:
                    ioncell = this_ioncell
                    final_fw_id = fw_id
        if final_fw_id is None:
            raise RuntimeError('Final structure not found ...')
        myfw = wf.id_fw[final_fw_id]
        #TODO add a check on the state of the launches
        last_launch = (myfw.archived_launches + myfw.launches)[-1]
        #TODO add a cycle to find the instance of AbiFireTask?
        myfw.tasks[-1].set_workdir(workdir=last_launch.launch_dir)
        structure = myfw.tasks[-1].get_final_structure()
        history = loadfn(os.path.join(last_launch.launch_dir, 'history.json'))

        return {'structure': structure.as_dict(), 'history': history}

    @classmethod
    def get_runtime_secs(cls, wf):
        assert wf.metadata['workflow_class'] == cls.workflow_class
        assert wf.metadata['workflow_module'] == cls.workflow_module
        time_secs = 0.0
        for fw_id, fw in wf.id_fw.items():
            if 'wf_task_index' in fw.spec:
                if fw.spec['wf_task_index'][-9:] == 'autoparal':
                    time_secs += fw.launches[-1].runtime_secs
                elif fw.spec['wf_task_index'][:4] == 'ion_':
                    time_secs += fw.launches[-1].runtime_secs * fw.spec['mpi_ncpus']
                elif fw.spec['wf_task_index'][:8] == 'ioncell_':
                    time_secs += fw.launches[-1].runtime_secs * fw.spec['mpi_ncpus']
        return time_secs

    @classmethod
    def from_factory(cls, structure, pseudos, kppa=None, nband=None, ecut=None, pawecutdg=None, accuracy="normal",
                     spin_mode="polarized", smearing="fermi_dirac:0.1 eV", charge=0.0, scf_algorithm=None,
                     extra_abivars={}, decorators=[], autoparal=False, spec={}, initialization_info={},
                     target_dilatmx=None):

        ion_input = ion_ioncell_relax_input(structure=structure, pseudos=pseudos, kppa=kppa, nband=nband, ecut=ecut,
                                            pawecutdg=pawecutdg, accuracy=accuracy, spin_mode=spin_mode,
                                            smearing=smearing, charge=charge, scf_algorithm=scf_algorithm)[0]

        ion_input.set_vars(**extra_abivars)
        for d in decorators:
            ion_input = d(ion_input)

        ioncell_fact = IoncellRelaxFromGsFactory(accuracy=accuracy, extra_abivars=extra_abivars, decorators=decorators)

        return cls(ion_input, ioncell_fact, autoparal=autoparal, spec=spec, initialization_info=initialization_info,
                   target_dilatmx=target_dilatmx)


class RelaxFWWorkflowSRC(AbstractFWWorkflow):
    workflow_class = 'RelaxFWWorkflowSRC'
    workflow_module = 'abiflows.fireworks.workflows.abinit_workflows'

    def __init__(self, ion_input, ioncell_input, spec={}, initialization_info={}):

        fws = []
        links_dict = {}

        if 'queue_adapter_update' in initialization_info:
            queue_adapter_update = initialization_info['queue_adapter_update']
        else:
            queue_adapter_update = None

        SRC_ion_fws = createSRCFireworksOld(task_class=RelaxFWTask, task_input=ion_input, SRC_spec=spec,
                                            initialization_info=initialization_info,
                                            wf_task_index_prefix='ion', queue_adapter_update=queue_adapter_update)
        fws.extend(SRC_ion_fws['fws'])
        links_dict.update(SRC_ion_fws['links_dict'])

        SRC_ioncell_fws = createSRCFireworksOld(task_class=RelaxFWTask, task_input=ioncell_input, SRC_spec=spec,
                                                initialization_info=initialization_info,
                                                wf_task_index_prefix='ioncell',
                                                deps={SRC_ion_fws['run_fw'].tasks[0].task_type: '@structure'},
                                                queue_adapter_update=queue_adapter_update)
        fws.extend(SRC_ioncell_fws['fws'])
        links_dict.update(SRC_ioncell_fws['links_dict'])

        links_dict.update({SRC_ion_fws['check_fw']: SRC_ioncell_fws['setup_fw']})

        self.wf = Workflow(fireworks=fws,
                           links_dict=links_dict,
                           metadata={'workflow_class': self.workflow_class,
                                     'workflow_module': self.workflow_module})

    @classmethod
    def get_final_structure_and_history(cls, wf):
        assert wf.metadata['workflow_class'] == cls.workflow_class
        assert wf.metadata['workflow_module'] == cls.workflow_module
        ioncell = -1
        final_fw_id = None
        for fw_id, fw in wf.id_fw.items():
            if 'wf_task_index' in fw.spec:
                if fw.spec['wf_task_index'][:12] == 'run_ioncell_':
                    try:
                        this_ioncell =  int(fw.spec['wf_task_index'].split('_')[-1])
                    except ValueError:
                        # skip if the index is not an int
                        continue
                    if this_ioncell > ioncell:
                        ioncell = this_ioncell
                        final_fw_id = fw_id
        if final_fw_id is None:
            raise RuntimeError('Final structure not found ...')
        myfw = wf.id_fw[final_fw_id]
        #TODO add a check on the state of the launches
        last_launch = (myfw.archived_launches + myfw.launches)[-1]
        #TODO add a cycle to find the instance of AbiFireTask?
        myfw.tasks[-1].set_workdir(workdir=last_launch.launch_dir)
        structure = myfw.tasks[-1].get_final_structure()
        history = loadfn(os.path.join(last_launch.launch_dir, 'history.json'))

        return {'structure': structure.as_dict(), 'history': history}


class NscfFWWorkflow(AbstractFWWorkflow):
    def __init__(self, scf_input, nscf_input, autoparal=False, spec={}, initialization_info={}):

        start_task_index = 1
        spec = dict(spec)
        spec['initialization_info'] = initialization_info
        if autoparal:
            spec = self.set_short_single_core_to_spec(spec)
            start_task_index = "autoparal"

        spec['wf_task_index'] = 'scf_' + str(start_task_index)
        scf_task = ScfFWTask(scf_input, is_autoparal=autoparal)
        self.scf_fw = Firework(scf_task, spec=spec)

        spec['wf_task_index'] = 'nscf_' + str(start_task_index)
        nscf_task = NscfFWTask(nscf_input, deps={scf_task.task_type: 'DEN'}, is_autoparal=autoparal)
        self.nscf_fw = Firework(nscf_task, spec=spec)

        self.wf = Workflow([self.scf_fw, self.nscf_fw], {self.scf_fw: [self.nscf_fw]},
                           metadata={'workflow_class': self.workflow_class,
                                     'workflow_module': self.workflow_module})


class NscfFWWorkflowSRC(AbstractFWWorkflow):
    workflow_class = 'NscfFWWorkflowSRC'
    workflow_module = 'abiflows.fireworks.workflows.abinit_workflows'

    def __init__(self, scf_input, nscf_input, spec={}, initialization_info={}):

        # Initializes fws list and links_dict
        fws = []
        links_dict = {}

        if 'additional_controllers' in spec:
            additional_controllers = spec['additional_controllers']
            spec.pop('additional_controllers')
        else:
            additional_controllers = [WalltimeController(), MemoryController()]
        # Self-consistent calculation
        scf_helper = ScfTaskHelper()
        scf_controllers = [AbinitController.from_helper(scf_helper)]
        scf_controllers.extend(additional_controllers)
        scf_control_procedure = ControlProcedure(controllers=scf_controllers)
        setup_scf_task = AbinitSetupTask(abiinput=scf_input, task_helper=scf_helper)
        run_scf_task = AbinitRunTask(control_procedure=scf_control_procedure, task_helper=scf_helper)
        control_scf_task = AbinitControlTask(control_procedure=scf_control_procedure, task_helper=scf_helper)

        scf_fws = createSRCFireworks(setup_task=setup_scf_task, run_task=run_scf_task, control_task=control_scf_task,
                                     task_index=scf_helper.task_type,
                                     spec=spec, initialization_info=initialization_info)

        fws.extend(scf_fws['fws'])
        links_dict_update(links_dict=links_dict, links_update=scf_fws['links_dict'])


        # Non self-consistent calculation
        nscf_helper = NscfTaskHelper()
        nscf_controllers = [AbinitController.from_helper(nscf_helper)]
        nscf_controllers.extend(additional_controllers)
        nscf_control_procedure = ControlProcedure(controllers=nscf_controllers)
        setup_nscf_task = AbinitSetupTask(abiinput=nscf_input, task_helper=nscf_helper,
                                          deps={run_scf_task.task_type: 'DEN'})
        run_nscf_task = AbinitRunTask(control_procedure=nscf_control_procedure, task_helper=nscf_helper)
        control_nscf_task = AbinitControlTask(control_procedure=nscf_control_procedure, task_helper=nscf_helper)

        nscf_fws = createSRCFireworks(setup_task=setup_nscf_task, run_task=run_nscf_task,
                                      control_task=control_nscf_task, task_index=nscf_helper.task_type, spec=spec,
                                      initialization_info=initialization_info)

        fws.extend(nscf_fws['fws'])
        links_dict_update(links_dict=links_dict, links_update=nscf_fws['links_dict'])
        #Link with previous SCF
        links_dict_update(links_dict=links_dict,
                          links_update={scf_fws['control_fw'].fw_id: nscf_fws['setup_fw'].fw_id})

        self.wf = Workflow(fireworks=fws, links_dict=links_dict,
                           metadata={'workflow_class': self.workflow_class,
                                     'workflow_module': self.workflow_module})

    @classmethod
    def from_factory(cls, structure, pseudos, kppa=None, ecut=None, pawecutdg=None, nband=None, accuracy="normal",
                     spin_mode="polarized", smearing="fermi_dirac:0.1 eV", charge=0.0, scf_algorithm=None,
                     shift_mode="Monkhorst-Pack", extra_abivars={}, decorators=[], autoparal=False, spec={}):
        raise NotImplementedError('from_factory class method not yet implemented for NscfWorkflowSRC')


class HybridOneShotFWWorkflow(AbstractFWWorkflow):
    def __init__(self, scf_inp, hybrid_input, autoparal=False, spec={}, initialization_info={}):
        rf = self.get_reduced_formula(scf_inp)

        scf_task = ScfFWTask(scf_inp, is_autoparal=autoparal)

        spec = dict(spec)
        spec['initialization_info'] = initialization_info
        if autoparal:
            spec = self.set_short_single_core_to_spec(spec)

        self.scf_fw = Firework(scf_task, spec=spec, name=rf+"_"+scf_task.task_type)

        hybrid_task = HybridFWTask(hybrid_input, is_autoparal=autoparal, deps=["WFK"])

        self.hybrid_fw = Firework(hybrid_task, spec=spec, name=rf+"_"+hybrid_task.task_type)

        self.wf = Workflow([self.scf_fw, self.hybrid_fw], {self.scf_fw: self.hybrid_fw})

    @classmethod
    def from_factory(cls, structure, pseudos, kppa=None, ecut=None, pawecutdg=None, nband=None, accuracy="normal",
                     spin_mode="polarized", smearing="fermi_dirac:0.1 eV", charge=0.0, scf_algorithm=None,
                     shift_mode="Monkhorst-Pack", hybrid_functional="hse06", ecutsigx=None, gw_qprange=1,
                     extra_abivars={}, decorators=[], autoparal=False, spec={}, initialization_info={}):

        scf_fact = ScfFactory(structure=structure, pseudos=pseudos, kppa=kppa, ecut=ecut, pawecutdg=pawecutdg,
                              nband=nband, accuracy=accuracy, spin_mode=spin_mode, smearing=smearing, charge=charge,
                              scf_algorithm=scf_algorithm, shift_mode=shift_mode, extra_abivars=extra_abivars,
                              decorators=decorators)

        hybrid_fact = HybridOneShotFromGsFactory(functional=hybrid_functional, ecutsigx=ecutsigx, gw_qprange=gw_qprange,
                                                 decorators=decorators, extra_abivars=extra_abivars)

        return cls(scf_fact, hybrid_fact, autoparal=autoparal, spec=spec, initialization_info=initialization_info)


# class NscfFWWorkflow(AbstractFWWorkflow):
#     def __init__(self, scf_input, nscf_input, autoparal=False, spec={}):
#
#         spec = dict(spec)
#         if autoparal:
#             spec = self.set_short_single_core_to_spec(spec)
#
#         ion_task = ScfFWTask(scf_input, is_autoparal=autoparal)
#         self.ion_fw = Firework(ion_task, spec=spec)
#
#         ioncell_task = NscfFWTask(nscf_input, deps={ion_task.task_type: 'DEN'}, is_autoparal=autoparal)
#         self.ioncell_fw = Firework(ioncell_task, spec=spec)
#
#         self.wf = Workflow([self.ion_fw, self.ioncell_fw], {self.ion_fw: [self.ioncell_fw]})


class PhononFWWorkflow(AbstractFWWorkflow):
    workflow_class = 'PhononFWWorkflow'
    workflow_module = 'abiflows.fireworks.workflows.abinit_workflows'

    def __init__(self, scf_inp, phonon_factory, autoparal=False, spec={}, initialization_info={}):
        start_task_index = 1

        rf = self.get_reduced_formula(scf_inp)

        scf_task = ScfFWTask(scf_inp, is_autoparal=autoparal)

        spec = dict(spec)
        spec['initialization_info'] = initialization_info
        if autoparal:
            spec = self.set_short_single_core_to_spec(spec)
            start_task_index = 'autoparal'

        spec['wf_task_index'] = 'scf_' + str(start_task_index)


        self.scf_fw = Firework(scf_task, spec=spec, name=rf+"_"+scf_task.task_type)

        ph_generation_task = GeneratePhononFlowFWAbinitTask(phonon_factory, previous_task_type=scf_task.task_type,
                                                            with_autoparal=autoparal)

        spec['wf_task_index'] = 'gen_ph'

        self.ph_generation_fw = Firework(ph_generation_task, spec=spec, name=rf+"_gen_ph")

        self.wf = Workflow([self.scf_fw, self.ph_generation_fw], {self.scf_fw: self.ph_generation_fw},
                           metadata={'workflow_class': self.workflow_class,
                                     'workflow_module': self.workflow_module})

    @classmethod
    def from_factory(cls, structure, pseudos, kppa=None, ecut=None, pawecutdg=None, nband=None, accuracy="normal",
                     spin_mode="polarized", smearing="fermi_dirac:0.1 eV", charge=0.0, scf_algorithm=None,
                     shift_mode="Symmetric", ph_ngqpt=None, with_ddk=True, with_dde=True, with_bec=False,
                     scf_tol=None, ph_tol=None, ddk_tol=None, dde_tol=None, extra_abivars={}, decorators=[],
                     autoparal=False, spec={}, initialization_info={}):

        extra_abivars_scf = dict(extra_abivars)
        extra_abivars_scf['tolwfr'] = scf_tol if scf_tol else 1.e-22
        scf_fact = ScfForPhononsFactory(structure=structure, pseudos=pseudos, kppa=kppa, ecut=ecut, pawecutdg=pawecutdg,
                                        nband=nband, accuracy=accuracy, spin_mode=spin_mode, smearing=smearing,
                                        charge=charge, scf_algorithm=scf_algorithm, shift_mode=shift_mode,
                                        extra_abivars=extra_abivars_scf, decorators=decorators)

        phonon_fact = PhononsFromGsFactory(ph_ngqpt=ph_ngqpt, with_ddk=with_ddk, with_dde=with_dde, with_bec=with_bec,
                                           ph_tol=ph_tol, ddk_tol=ddk_tol, dde_tol=dde_tol, extra_abivars=extra_abivars,
                                           decorators=decorators)

        return cls(scf_fact, phonon_fact, autoparal=autoparal, spec=spec, initialization_info=initialization_info)


class PiezoElasticFWWorkflow(AbstractFWWorkflow):
    workflow_class = 'PiezoElasticFWWorkflow'
    workflow_module = 'abiflows.fireworks.workflows.abinit_workflows'

    def __init__(self, scf_inp, ddk_inp, rf_inp, autoparal=False, spec={}, initialization_info={}):
        rf = self.get_reduced_formula(scf_inp)

        scf_task = ScfFWTask(scf_inp, is_autoparal=autoparal)

        spec = dict(spec)
        spec['initialization_info'] = initialization_info
        if autoparal:
            spec = self.set_short_single_core_to_spec(spec)

        self.scf_fw = Firework(scf_task, spec=spec, name=rf+"_"+scf_task.task_type)

        ddk_task = DdkTask(ddk_inp, is_autoparal=autoparal, deps={scf_task.task_type: 'WFK'})

        ddk_fw_name = rf+ddk_task.task_type
        ddk_fw_name = ddk_fw_name[:8]
        self.ddk_fw = Firework(ddk_task, spec=spec, name=ddk_fw_name)

        rf_task = StrainPertTask(rf_inp, is_autoparal=autoparal, deps={scf_task.task_type: 'WFK', ddk_task.task_type: 'DDK'})

        rf_fw_name = rf+rf_task.task_type
        rf_fw_name = rf_fw_name[:8]
        self.rf_fw = Firework(rf_task, spec=spec, name=rf_fw_name)

        self.wf = Workflow(fireworks=[self.scf_fw, self.ddk_fw, self.rf_fw],
                           links_dict={self.scf_fw: self.ddk_fw, self.ddk_fw: self.rf_fw},
                           metadata={'workflow_class': self.workflow_class,
                                     'workflow_module': self.workflow_module})

        self.add_anaddb_task(scf_inp.structure)

    def add_anaddb_task(self, structure):
        spec = self.set_short_single_core_to_spec()
        anaddb_task = AnaDdbAbinitTask(AnaddbInput.piezo_elastic(structure))
        anaddb_fw = Firework([anaddb_task],
                             spec=spec,
                             name='anaddb')
        append_fw_to_wf(anaddb_fw, self.wf)

    def add_mrgddb_task(self, structure):
        spec = self.set_short_single_core_to_spec()
        spec['ddb_files_task_types'] = ['scf', 'strain_pert']
        mrgddb_task = MergeDdbAbinitTask()
        mrgddb_fw = Firework([mrgddb_task], spec=spec, name='mrgddb')
        append_fw_to_wf(mrgddb_fw, self.wf)

    @classmethod
    def get_elastic_tensor_and_history(cls, wf):
        assert wf.metadata['workflow_class'] == cls.workflow_class
        assert wf.metadata['workflow_module'] == cls.workflow_module

        final_fw_id = None
        for fw_id, fw in wf.id_fw.items():
            if fw.name == 'anaddb':
                final_fw_id = fw_id
        if final_fw_id is None:
            raise RuntimeError('Final anaddb task not found ...')
        myfw = wf.id_fw[final_fw_id]
        #TODO add a check on the state of the launches
        last_launch = (myfw.archived_launches + myfw.launches)[-1]
        #TODO add a cycle to find the instance of AbiFireTask?
        myfw.tasks[-1].set_workdir(workdir=last_launch.launch_dir)
        elastic_tensor = myfw.tasks[-1].get_elastic_tensor()
        history = loadfn(os.path.join(last_launch.launch_dir, 'history.json'))

        return {'elastic_properties': elastic_tensor.extended_dict(), 'history': history}

    @classmethod
    def get_all_elastic_tensors(cls, wf):
        assert wf.metadata['workflow_class'] == cls.workflow_class
        assert wf.metadata['workflow_module'] == cls.workflow_module

        final_fw_id = None
        for fw_id, fw in wf.id_fw.items():
            if fw.name == 'anaddb':
                final_fw_id = fw_id
        if final_fw_id is None:
            raise RuntimeError('Final anaddb task not found ...')
        myfw = wf.id_fw[final_fw_id]
        #TODO add a check on the state of the launches
        last_launch = (myfw.archived_launches + myfw.launches)[-1]
        #TODO add a cycle to find the instance of AbiFireTask?
        myfw.tasks[-1].set_workdir(workdir=last_launch.launch_dir)
        elastic_tensor = myfw.tasks[-1].get_elastic_tensor()
        history = loadfn(os.path.join(last_launch.launch_dir, 'history.json'))

        return {'elastic_properties': elastic_tensor.extended_dict(), 'history': history}

    @classmethod
    def from_factory(cls):
        raise NotImplemented('from factory method not yet implemented for piezoelasticworkflow')


class PiezoElasticFWWorkflowSRCOld(AbstractFWWorkflow):
    workflow_class = 'PiezoElasticFWWorkflowSRC'
    workflow_module = 'abiflows.fireworks.workflows.abinit_workflows'

    STANDARD_HANDLERS = {'_all': [MemoryHandler(), WalltimeHandler()]}
    STANDARD_VALIDATORS = {'_all': []}

    def __init__(self, scf_inp_ibz, ddk_inp, rf_inp, spec={}, initialization_info={},
                 handlers=STANDARD_HANDLERS, validators=STANDARD_VALIDATORS, ddk_split=False, rf_split=False):

        fws = []
        links_dict = {}

        if 'queue_adapter_update' in initialization_info:
            queue_adapter_update = initialization_info['queue_adapter_update']
        else:
            queue_adapter_update = None

        # If handlers are passed as a list, they should be applied on all task_types
        if isinstance(handlers, (list, tuple)):
            handlers = {'_all': handlers}
        # If validators are passed as a list, they should be applied on all task_types
        if isinstance(validators, (list, tuple)):
            validators = {'_all': validators}

        #1. First SCF run in the irreducible Brillouin Zone
        SRC_scf_ibz_fws = createSRCFireworksOld(task_class=ScfFWTask, task_input=scf_inp_ibz, SRC_spec=spec,
                                                initialization_info=initialization_info,
                                                wf_task_index_prefix='scfibz', task_type='scfibz',
                                                handlers=handlers['_all'], validators=validators['_all'],
                                                queue_adapter_update=queue_adapter_update)
        fws.extend(SRC_scf_ibz_fws['fws'])
        links_dict_update(links_dict=links_dict, links_update=SRC_scf_ibz_fws['links_dict'])

        #2. Second SCF run in the full Brillouin Zone with kptopt 3 in order to allow merging 1st derivative DDB's with
        #2nd derivative DDB's from the DFPT RF run
        scf_inp_fbz = scf_inp_ibz.deepcopy()
        scf_inp_fbz['kptopt'] = 2
        SRC_scf_fbz_fws = createSRCFireworksOld(task_class=ScfFWTask, task_input=scf_inp_fbz, SRC_spec=spec,
                                                initialization_info=initialization_info,
                                                wf_task_index_prefix='scffbz', task_type='scffbz',
                                                handlers=handlers['_all'], validators=validators['_all'],
                                                deps={SRC_scf_ibz_fws['run_fw'].tasks[0].task_type: ['DEN', 'WFK']},
                                                queue_adapter_update=queue_adapter_update)
        fws.extend(SRC_scf_fbz_fws['fws'])
        links_dict_update(links_dict=links_dict, links_update=SRC_scf_fbz_fws['links_dict'])
        #Link with previous SCF
        links_dict_update(links_dict=links_dict,
                          links_update={SRC_scf_ibz_fws['check_fw'].fw_id: SRC_scf_fbz_fws['setup_fw'].fw_id})

        #3. DDK calculation
        if ddk_split:
            raise NotImplementedError('Split Ddk to be implemented in PiezoElasticWorkflow ...')
        else:
            SRC_ddk_fws = createSRCFireworksOld(task_class=DdkTask, task_input=ddk_inp, SRC_spec=spec,
                                                initialization_info=initialization_info,
                                                wf_task_index_prefix='ddk',
                                                handlers=handlers['_all'], validators=validators['_all'],
                                                deps={SRC_scf_ibz_fws['run_fw'].tasks[0].task_type: 'WFK'},
                                                queue_adapter_update=queue_adapter_update)
            fws.extend(SRC_ddk_fws['fws'])
            links_dict_update(links_dict=links_dict, links_update=SRC_ddk_fws['links_dict'])
            #Link with the IBZ SCF run
            links_dict_update(links_dict=links_dict,
                              links_update={SRC_scf_ibz_fws['check_fw'].fw_id: SRC_ddk_fws['setup_fw'].fw_id})

        #4. Response-Function calculation(s) of the elastic constants
        if rf_split:
            rf_ddb_source_task_type = 'mrgddb-strains'
            scf_task_type = SRC_scf_ibz_fws['run_fw'].tasks[0].task_type
            ddk_task_type = SRC_ddk_fws['run_fw'].tasks[0].task_type
            gen_task = GeneratePiezoElasticFlowFWAbinitTask(previous_scf_task_type=scf_task_type,
                                                            previous_ddk_task_type=ddk_task_type,
                                                            handlers=handlers, validators=validators,
                                                            mrgddb_task_type=rf_ddb_source_task_type)
            genrfstrains_spec = set_short_single_core_to_spec(spec)
            gen_fw = Firework([gen_task], spec=genrfstrains_spec, name='gen-piezo-elast')
            fws.append(gen_fw)
            links_dict_update(links_dict=links_dict,
                              links_update={SRC_scf_ibz_fws['check_fw'].fw_id: gen_fw.fw_id,
                                            SRC_ddk_fws['check_fw'].fw_id: gen_fw.fw_id})
            rf_ddb_src_fw = gen_fw
        else:
            SRC_rf_fws = createSRCFireworksOld(task_class=StrainPertTask, task_input=rf_inp, SRC_spec=spec,
                                               initialization_info=initialization_info,
                                               wf_task_index_prefix='rf',
                                               handlers=handlers['_all'], validators=validators['_all'],
                                               deps={SRC_scf_ibz_fws['run_fw'].tasks[0].task_type: 'WFK',
                                                     SRC_ddk_fws['run_fw'].tasks[0].task_type: 'DDK'},
                                               queue_adapter_update=queue_adapter_update)
            fws.extend(SRC_rf_fws['fws'])
            links_dict_update(links_dict=links_dict, links_update=SRC_rf_fws['links_dict'])
            #Link with the IBZ SCF run and the DDK run
            links_dict_update(links_dict=links_dict,
                              links_update={SRC_scf_ibz_fws['check_fw'].fw_id: SRC_rf_fws['setup_fw'].fw_id,
                                            SRC_ddk_fws['check_fw'].fw_id: SRC_rf_fws['setup_fw'].fw_id})
            rf_ddb_source_task_type = SRC_rf_fws['run_fw'].tasks[0].task_type
            rf_ddb_src_fw = SRC_rf_fws['check_fw']

        #5. Merge DDB files from response function (second derivatives for the elastic constants) and from the
        # SCF run on the full Brillouin zone (first derivatives for the stress tensor, to be used for the
        # stress-corrected elastic constants)
        mrgddb_task = MergeDdbAbinitTask(ddb_source_task_types=[rf_ddb_source_task_type,
                                                                SRC_scf_fbz_fws['run_fw'].tasks[0].task_type],
                                         delete_source_ddbs=False, num_ddbs=2)
        mrgddb_spec = set_short_single_core_to_spec(spec)
        mrgddb_fw = Firework(tasks=[mrgddb_task], spec=mrgddb_spec, name='mrgddb')
        fws.append(mrgddb_fw)
        links_dict_update(links_dict=links_dict,
                          links_update={rf_ddb_src_fw.fw_id: mrgddb_fw.fw_id,
                                        SRC_scf_fbz_fws['check_fw'].fw_id: mrgddb_fw.fw_id})

        #6. Anaddb task to get elastic constants based on the RF run (no stress correction)
        anaddb_tag = 'anaddb-piezo-elast'
        spec = set_short_single_core_to_spec(spec)
        anaddb_task = AnaDdbAbinitTask(AnaddbInput.piezo_elastic(structure=scf_inp_ibz.structure,
                                                                 stress_correction=False),
                                       deps={rf_ddb_source_task_type: ['DDB']},
                                       task_type=anaddb_tag)
        anaddb_fw = Firework([anaddb_task],
                             spec=spec,
                             name=anaddb_tag)
        fws.append(anaddb_fw)
        links_dict_update(links_dict=links_dict,
                          links_update={rf_ddb_src_fw.fw_id: anaddb_fw.fw_id})

        #7. Anaddb task to get elastic constants based on the RF run and the SCF run (with stress correction)
        anaddb_tag = 'anaddb-piezo-elast-stress-corrected'
        spec = set_short_single_core_to_spec(spec)
        anaddb_stress_task = AnaDdbAbinitTask(AnaddbInput.piezo_elastic(structure=scf_inp_ibz.structure,
                                                                        stress_correction=True),
                                              deps={mrgddb_task.task_type: ['DDB']},
                                              task_type=anaddb_tag)
        anaddb_stress_fw = Firework([anaddb_stress_task],
                                    spec=spec,
                                    name=anaddb_tag)
        fws.append(anaddb_stress_fw)
        links_dict_update(links_dict=links_dict,
                          links_update={mrgddb_fw.fw_id: anaddb_stress_fw.fw_id})

        self.wf = Workflow(fireworks=fws,
                           links_dict=links_dict,
                           metadata={'workflow_class': self.workflow_class,
                                     'workflow_module': self.workflow_module})

    @classmethod
    def get_all_elastic_tensors(cls, wf):
        assert wf.metadata['workflow_class'] == cls.workflow_class
        assert wf.metadata['workflow_module'] == cls.workflow_module

        anaddb_no_stress_id = None
        anaddb_stress_id = None
        for fw_id, fw in wf.id_fw.items():
            if fw.name == 'anaddb-piezo-elast':
                anaddb_no_stress_id = fw_id
            if fw.name == 'anaddb-piezo-elast-stress-corrected':
                anaddb_stress_id = fw_id
        if anaddb_no_stress_id is None or anaddb_stress_id is None:
            raise RuntimeError('Final anaddb tasks not found ...')
        myfw_nostress = wf.id_fw[anaddb_no_stress_id]
        last_launch_nostress = (myfw_nostress.archived_launches + myfw_nostress.launches)[-1]
        myfw_nostress.tasks[-1].set_workdir(workdir=last_launch_nostress.launch_dir)

        myfw_stress = wf.id_fw[anaddb_stress_id]
        last_launch_stress = (myfw_stress.archived_launches + myfw_stress.launches)[-1]
        myfw_stress.tasks[-1].set_workdir(workdir=last_launch_stress.launch_dir)

        ec_nostress_clamped = myfw_nostress.tasks[-1].get_elastic_tensor(tensor_type='clamped_ion')
        ec_nostress_relaxed = myfw_nostress.tasks[-1].get_elastic_tensor(tensor_type='relaxed_ion')
        ec_stress_relaxed = myfw_stress.tasks[-1].get_elastic_tensor(tensor_type='relaxed_ion_stress_corrected')

        ec_dicts = {'clamped_ion': ec_nostress_clamped.extended_dict(),
                    'relaxed_ion': ec_nostress_relaxed.extended_dict(),
                    'relaxed_ion_stress_corrected': ec_stress_relaxed.extended_dict()}

        return {'elastic_properties': ec_dicts}

    @classmethod
    def from_factory(cls):
        raise NotImplemented('from factory method not yet implemented for piezoelasticworkflow')


class PiezoElasticFWWorkflowSRC(AbstractFWWorkflow):
    workflow_class = 'PiezoElasticFWWorkflowSRC'
    workflow_module = 'abiflows.fireworks.workflows.abinit_workflows'


    def __init__(self, scf_inp_ibz, ddk_inp, rf_inp, spec={}, initialization_info={},
                 ddk_split=False, rf_split=False, additional_controllers=None, new=False):


        if new:
            fws = []
            links_dict = {}

            if additional_controllers is None:
                additional_controllers = [WalltimeController(), MemoryController()]
            else:
                additional_controllers = additional_controllers

            #1. SCF run in the irreducible Brillouin Zone
            scf_helper = ScfTaskHelper()
            scf_controllers = [AbinitController.from_helper(scf_helper)]
            scf_controllers.extend(additional_controllers)
            scf_control_procedure = ControlProcedure(controllers=scf_controllers)
            setup_scf_task = AbinitSetupTask(abiinput=scf_inp_ibz, task_helper=scf_helper, pass_input=True)
            run_scf_task = AbinitRunTask(control_procedure=scf_control_procedure, task_helper=scf_helper,
                                         task_type='scfibz')
            control_scf_task = AbinitControlTask(control_procedure=scf_control_procedure, task_helper=scf_helper)

            scf_fws = createSRCFireworks(setup_task=setup_scf_task, run_task=run_scf_task, control_task=control_scf_task,
                                         spec=spec, initialization_info=initialization_info)

            fws.extend(scf_fws['fws'])
            links_dict_update(links_dict=links_dict, links_update=scf_fws['links_dict'])

            #2. nSCF run in the full Brillouin Zone with kptopt 2
            nscf_helper = NscfTaskHelper()
            nscf_controllers = [AbinitController.from_helper(nscf_helper)]
            nscf_controllers.extend(additional_controllers)
            nscf_control_procedure = ControlProcedure(controllers=nscf_controllers)
            nscf_inp_fbz = scf_inp_ibz.deepcopy()
            nscf_inp_fbz.set_vars({'tolwfr': 1.0e-20,
                                   'kptopt': 2,
                                   'iscf': -2})
            # Adding buffer to help convergence ...
            if 'nbdbuf' not in nscf_inp_fbz:
                nbdbuf = max(int(0.1*nscf_inp_fbz['nband']), 4)
                nscf_inp_fbz.set_vars(nband=nscf_inp_fbz['nband']+nbdbuf, nbdbuf=nbdbuf)
            setup_nscffbz_task = AbinitSetupTask(abiinput=nscf_inp_fbz, task_helper=nscf_helper,
                                                 deps={run_scf_task.task_type: ['DEN']})
            run_nscffbz_task = AbinitRunTask(control_procedure=nscf_control_procedure, task_helper=nscf_helper,
                                             task_type='nscffbz')
            control_nscffbz_task = AbinitControlTask(control_procedure=nscf_control_procedure, task_helper=nscf_helper)

            nscffbz_fws = createSRCFireworks(setup_task=setup_nscffbz_task, run_task=run_nscffbz_task,
                                             control_task=control_nscffbz_task,
                                             spec=spec, initialization_info=initialization_info)

            fws.extend(nscffbz_fws['fws'])
            links_dict_update(links_dict=links_dict, links_update=nscffbz_fws['links_dict'])
            #Link with the IBZ SCF run
            links_dict_update(links_dict=links_dict,
                              links_update={scf_fws['control_fw'].fw_id: nscffbz_fws['setup_fw'].fw_id})

            #3. DDK calculation
            if ddk_split:
                raise NotImplementedError('Split Ddk to be implemented in PiezoElasticWorkflow ...')
            else:
                ddk_helper = DdkTaskHelper()
                ddk_controllers = [AbinitController.from_helper(ddk_helper)]
                ddk_controllers.extend(additional_controllers)
                ddk_control_procedure = ControlProcedure(controllers=ddk_controllers)
                setup_ddk_task = AbinitSetupTask(abiinput=ddk_inp, task_helper=ddk_helper,
                                                 deps={run_nscffbz_task.task_type: 'WFK'})
                run_ddk_task = AbinitRunTask(control_procedure=ddk_control_procedure, task_helper=ddk_helper,
                                             task_type='ddk')
                control_ddk_task = AbinitControlTask(control_procedure=ddk_control_procedure, task_helper=ddk_helper)

                ddk_fws = createSRCFireworks(setup_task=setup_ddk_task, run_task=run_ddk_task,
                                             control_task=control_ddk_task,
                                             spec=spec, initialization_info=initialization_info)

                fws.extend(ddk_fws['fws'])
                links_dict_update(links_dict=links_dict, links_update=ddk_fws['links_dict'])
                #Link with the FBZ nSCF run
                links_dict_update(links_dict=links_dict,
                                  links_update={nscffbz_fws['control_fw'].fw_id: ddk_fws['setup_fw'].fw_id})

            #4. Response-Function calculation(s) of the elastic constants
            rf_ddb_source_task_type = 'mrgddb-strains'
            gen_task = GeneratePiezoElasticFlowFWSRCAbinitTask(previous_scf_task_type=run_nscffbz_task.task_type,
                                                               previous_ddk_task_type=run_ddk_task.task_type,
                                                               mrgddb_task_type=rf_ddb_source_task_type,
                                                               additional_controllers=additional_controllers)
            genrfstrains_spec = set_short_single_core_to_spec(spec)
            gen_fw = Firework([gen_task], spec=genrfstrains_spec, name='gen-piezo-elast')
            fws.append(gen_fw)
            links_dict_update(links_dict=links_dict,
                              links_update={nscffbz_fws['control_fw'].fw_id: gen_fw.fw_id,
                                            ddk_fws['control_fw'].fw_id: gen_fw.fw_id})

            rf_ddb_src_fw = gen_fw

            #5. Merge DDB files from response function (second derivatives for the elastic constants) and from the
            # SCF run on the full Brillouin zone (first derivatives for the stress tensor, to be used for the
            # stress-corrected elastic constants)
            mrgddb_task = MergeDdbAbinitTask(ddb_source_task_types=[rf_ddb_source_task_type,
                                                                    run_nscffbz_task.task_type],
                                             delete_source_ddbs=False, num_ddbs=2)
            mrgddb_spec = set_short_single_core_to_spec(spec)
            mrgddb_fw = Firework(tasks=[mrgddb_task], spec=mrgddb_spec, name='mrgddb')
            fws.append(mrgddb_fw)
            links_dict_update(links_dict=links_dict,
                              links_update={rf_ddb_src_fw.fw_id: mrgddb_fw.fw_id,
                                            nscffbz_fws['control_fw'].fw_id: mrgddb_fw.fw_id})

            #6. Anaddb task to get elastic constants based on the RF run (no stress correction)
            anaddb_tag = 'anaddb-piezo-elast'
            spec = set_short_single_core_to_spec(spec)
            anaddb_task = AnaDdbAbinitTask(AnaddbInput.piezo_elastic(structure=scf_inp_ibz.structure,
                                                                     stress_correction=False),
                                           deps={rf_ddb_source_task_type: ['DDB']},
                                           task_type=anaddb_tag)
            anaddb_fw = Firework([anaddb_task],
                                 spec=spec,
                                 name=anaddb_tag)
            fws.append(anaddb_fw)
            links_dict_update(links_dict=links_dict,
                              links_update={rf_ddb_src_fw.fw_id: anaddb_fw.fw_id})

            #7. Anaddb task to get elastic constants based on the RF run and the SCF run (with stress correction)
            anaddb_tag = 'anaddb-piezo-elast-stress-corrected'
            spec = set_short_single_core_to_spec(spec)
            anaddb_stress_task = AnaDdbAbinitTask(AnaddbInput.piezo_elastic(structure=scf_inp_ibz.structure,
                                                                            stress_correction=True),
                                                  deps={mrgddb_task.task_type: ['DDB']},
                                                  task_type=anaddb_tag)
            anaddb_stress_fw = Firework([anaddb_stress_task],
                                        spec=spec,
                                        name=anaddb_tag)
            fws.append(anaddb_stress_fw)
            links_dict_update(links_dict=links_dict,
                              links_update={mrgddb_fw.fw_id: anaddb_stress_fw.fw_id})

            self.wf = Workflow(fireworks=fws,
                               links_dict=links_dict,
                               metadata={'workflow_class': self.workflow_class,
                                         'workflow_module': self.workflow_module})
        else:
            fws = []
            links_dict = {}

            if additional_controllers is None:
                additional_controllers = [WalltimeController(), MemoryController()]
            else:
                additional_controllers = additional_controllers

            #1. First SCF run in the irreducible Brillouin Zone
            scf_helper = ScfTaskHelper()
            scf_controllers = [AbinitController.from_helper(scf_helper)]
            scf_controllers.extend(additional_controllers)
            scf_control_procedure = ControlProcedure(controllers=scf_controllers)
            setup_scf_task = AbinitSetupTask(abiinput=scf_inp_ibz, task_helper=scf_helper, pass_input=True)
            run_scf_task = AbinitRunTask(control_procedure=scf_control_procedure, task_helper=scf_helper,
                                         task_type='scfibz')
            control_scf_task = AbinitControlTask(control_procedure=scf_control_procedure, task_helper=scf_helper)

            scf_fws = createSRCFireworks(setup_task=setup_scf_task, run_task=run_scf_task, control_task=control_scf_task,
                                         spec=spec, initialization_info=initialization_info)

            fws.extend(scf_fws['fws'])
            links_dict_update(links_dict=links_dict, links_update=scf_fws['links_dict'])

            #2. Second SCF run in the full Brillouin Zone with kptopt 3 in order to allow merging 1st derivative DDB's with
            #2nd derivative DDB's from the DFPT RF run
            scf_inp_fbz = scf_inp_ibz.deepcopy()
            scf_inp_fbz['kptopt'] = 2
            setup_scffbz_task = AbinitSetupTask(abiinput=scf_inp_fbz, task_helper=scf_helper,
                                                deps={run_scf_task.task_type: ['WFK', 'DEN']})
            run_scffbz_task = AbinitRunTask(control_procedure=scf_control_procedure, task_helper=scf_helper,
                                            task_type='scffbz')
            control_scffbz_task = AbinitControlTask(control_procedure=scf_control_procedure, task_helper=scf_helper)

            scffbz_fws = createSRCFireworks(setup_task=setup_scffbz_task, run_task=run_scffbz_task,
                                            control_task=control_scffbz_task,
                                            spec=spec, initialization_info=initialization_info)

            fws.extend(scffbz_fws['fws'])
            links_dict_update(links_dict=links_dict, links_update=scffbz_fws['links_dict'])
            #Link with the IBZ SCF run
            links_dict_update(links_dict=links_dict,
                              links_update={scf_fws['control_fw'].fw_id: scffbz_fws['setup_fw'].fw_id})

            #3. DDK calculation
            if ddk_split:
                raise NotImplementedError('Split Ddk to be implemented in PiezoElasticWorkflow ...')
            else:
                ddk_helper = DdkTaskHelper()
                ddk_controllers = [AbinitController.from_helper(ddk_helper)]
                ddk_controllers.extend(additional_controllers)
                ddk_control_procedure = ControlProcedure(controllers=ddk_controllers)
                setup_ddk_task = AbinitSetupTask(abiinput=ddk_inp, task_helper=ddk_helper,
                                                 deps={run_scf_task.task_type: 'WFK'})
                run_ddk_task = AbinitRunTask(control_procedure=ddk_control_procedure, task_helper=ddk_helper,
                                             task_type='ddk')
                control_ddk_task = AbinitControlTask(control_procedure=ddk_control_procedure, task_helper=ddk_helper)

                ddk_fws = createSRCFireworks(setup_task=setup_ddk_task, run_task=run_ddk_task,
                                             control_task=control_ddk_task,
                                             spec=spec, initialization_info=initialization_info)

                fws.extend(ddk_fws['fws'])
                links_dict_update(links_dict=links_dict, links_update=ddk_fws['links_dict'])
                #Link with the IBZ SCF run
                links_dict_update(links_dict=links_dict,
                                  links_update={scf_fws['control_fw'].fw_id: ddk_fws['setup_fw'].fw_id})

            #4. Response-Function calculation(s) of the elastic constants
            rf_ddb_source_task_type = 'mrgddb-strains'
            gen_task = GeneratePiezoElasticFlowFWSRCAbinitTask(previous_scf_task_type=run_scf_task.task_type,
                                                               previous_ddk_task_type=run_ddk_task.task_type,
                                                               mrgddb_task_type=rf_ddb_source_task_type,
                                                               additional_controllers=additional_controllers)
            genrfstrains_spec = set_short_single_core_to_spec(spec)
            gen_fw = Firework([gen_task], spec=genrfstrains_spec, name='gen-piezo-elast')
            fws.append(gen_fw)
            links_dict_update(links_dict=links_dict,
                              links_update={scf_fws['control_fw'].fw_id: gen_fw.fw_id,
                                            ddk_fws['control_fw'].fw_id: gen_fw.fw_id})

            rf_ddb_src_fw = gen_fw

            #5. Merge DDB files from response function (second derivatives for the elastic constants) and from the
            # SCF run on the full Brillouin zone (first derivatives for the stress tensor, to be used for the
            # stress-corrected elastic constants)
            mrgddb_task = MergeDdbAbinitTask(ddb_source_task_types=[rf_ddb_source_task_type,
                                                                    run_scffbz_task.task_type],
                                             delete_source_ddbs=False, num_ddbs=2)
            mrgddb_spec = set_short_single_core_to_spec(spec)
            mrgddb_fw = Firework(tasks=[mrgddb_task], spec=mrgddb_spec, name='mrgddb')
            fws.append(mrgddb_fw)
            links_dict_update(links_dict=links_dict,
                              links_update={rf_ddb_src_fw.fw_id: mrgddb_fw.fw_id,
                                            scffbz_fws['control_fw'].fw_id: mrgddb_fw.fw_id})

            #6. Anaddb task to get elastic constants based on the RF run (no stress correction)
            anaddb_tag = 'anaddb-piezo-elast'
            spec = set_short_single_core_to_spec(spec)
            anaddb_task = AnaDdbAbinitTask(AnaddbInput.piezo_elastic(structure=scf_inp_ibz.structure,
                                                                     stress_correction=False),
                                           deps={rf_ddb_source_task_type: ['DDB']},
                                           task_type=anaddb_tag)
            anaddb_fw = Firework([anaddb_task],
                                 spec=spec,
                                 name=anaddb_tag)
            fws.append(anaddb_fw)
            links_dict_update(links_dict=links_dict,
                              links_update={rf_ddb_src_fw.fw_id: anaddb_fw.fw_id})

            #7. Anaddb task to get elastic constants based on the RF run and the SCF run (with stress correction)
            anaddb_tag = 'anaddb-piezo-elast-stress-corrected'
            spec = set_short_single_core_to_spec(spec)
            anaddb_stress_task = AnaDdbAbinitTask(AnaddbInput.piezo_elastic(structure=scf_inp_ibz.structure,
                                                                            stress_correction=True),
                                                  deps={mrgddb_task.task_type: ['DDB']},
                                                  task_type=anaddb_tag)
            anaddb_stress_fw = Firework([anaddb_stress_task],
                                        spec=spec,
                                        name=anaddb_tag)
            fws.append(anaddb_stress_fw)
            links_dict_update(links_dict=links_dict,
                              links_update={mrgddb_fw.fw_id: anaddb_stress_fw.fw_id})

            self.wf = Workflow(fireworks=fws,
                               links_dict=links_dict,
                               metadata={'workflow_class': self.workflow_class,
                                         'workflow_module': self.workflow_module})

    @classmethod
    def get_all_elastic_tensors(cls, wf):
        assert wf.metadata['workflow_class'] == cls.workflow_class
        assert wf.metadata['workflow_module'] == cls.workflow_module

        anaddb_no_stress_id = None
        anaddb_stress_id = None
        for fw_id, fw in wf.id_fw.items():
            if fw.name == 'anaddb-piezo-elast':
                anaddb_no_stress_id = fw_id
            if fw.name == 'anaddb-piezo-elast-stress-corrected':
                anaddb_stress_id = fw_id
        if anaddb_no_stress_id is None or anaddb_stress_id is None:
            raise RuntimeError('Final anaddb tasks not found ...')
        myfw_nostress = wf.id_fw[anaddb_no_stress_id]
        last_launch_nostress = (myfw_nostress.archived_launches + myfw_nostress.launches)[-1]
        myfw_nostress.tasks[-1].set_workdir(workdir=last_launch_nostress.launch_dir)

        myfw_stress = wf.id_fw[anaddb_stress_id]
        last_launch_stress = (myfw_stress.archived_launches + myfw_stress.launches)[-1]
        myfw_stress.tasks[-1].set_workdir(workdir=last_launch_stress.launch_dir)

        ec_nostress_clamped = myfw_nostress.tasks[-1].get_elastic_tensor(tensor_type='clamped_ion')
        ec_nostress_relaxed = myfw_nostress.tasks[-1].get_elastic_tensor(tensor_type='relaxed_ion')
        ec_stress_relaxed = myfw_stress.tasks[-1].get_elastic_tensor(tensor_type='relaxed_ion_stress_corrected')

        ec_dicts = {'clamped_ion': ec_nostress_clamped.extended_dict(),
                    'relaxed_ion': ec_nostress_relaxed.extended_dict(),
                    'relaxed_ion_stress_corrected': ec_stress_relaxed.extended_dict()}

        return {'elastic_properties': ec_dicts}

    @classmethod
    def from_factory(cls):
        raise NotImplemented('from factory method not yet implemented for piezoelasticworkflow')
