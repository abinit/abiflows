from __future__ import print_function, division, unicode_literals, absolute_import

import copy
import logging
import os
import threading
import subprocess

from monty.json import MSONable
from abiflows.fireworks.tasks.src_tasks_abc import SetupTask, RunTask, ControlTask
from abiflows.core.mastermind_abc import ControllerNote
from abiflows.fireworks.utils.fw_utils import FWTaskManager
from abiflows.fireworks.tasks.src_tasks_abc import SRCTaskIndex
from abiflows.fireworks.utils.fw_utils import set_short_single_core_to_spec
from abiflows.core.controllers import WalltimeController, MemoryController, VaspXMLValidatorController
from abiflows.core.controllers import VaspNEBValidatorController
from abiflows.core.mastermind_abc import ControlProcedure
from pymatgen.io.abinit.tasks import ParalHints
from fireworks import explicit_serialize
from fireworks.core.firework import Firework, Workflow
from fireworks.core.firework import FWAction
from pymatgen.util.serialization import pmg_serialize
from pymatgen.io.abinit.utils import Directory
from pymatgen.io.vasp import Vasprun
from pymatgen.analysis.transition_state import NEBAnalysis
from abiflows.fireworks.tasks.vasp_sets import MPNEBSet
from abiflows.fireworks.tasks.vasp_sets import MPcNEBSet
from custodian.custodian import Custodian
from custodian.vasp.jobs import VaspJob
from fireworks.utilities.fw_serializers import serialize_fw
from fireworks.core.firework import FireTaskBase

RESET_RESTART = ControllerNote.RESET_RESTART
SIMPLE_RESTART = ControllerNote.SIMPLE_RESTART
RESTART_FROM_SCRATCH = ControllerNote.RESTART_FROM_SCRATCH

logger = logging.getLogger(__name__)


class VaspSRCMixin(object):

    def get_fw_task_manager(self, fw_spec):
        if 'ftm_file' in fw_spec:
            ftm = FWTaskManager.from_file(fw_spec['ftm_file'])
        else:
            ftm = FWTaskManager.from_user_config()
        ftm.update_fw_policy(fw_spec.get('fw_policy', {}))
        return ftm

    def setup_rundir(self, rundir):
        self.run_dir = rundir


@explicit_serialize
class VaspSetupTask(VaspSRCMixin, SetupTask):

    RUN_PARAMETERS = ['_queueadapter', 'qtk_queueadapter']

    def __init__(self, vasp_input_set, deps=None, task_helper=None, task_type=None,
                 restart_info=None, pass_input=False):
        if task_type is None:
            task_type = task_helper.task_type
        SetupTask.__init__(self, deps=deps, restart_info=restart_info, task_type=task_type)
        self.vasp_input_set = vasp_input_set
        self.pass_input = pass_input

        self.task_helper = task_helper
        self.task_helper.set_task(self)

    def setup_directories(self, fw_spec, create_dirs=False):
        SetupTask.setup_directories(self, fw_spec=fw_spec, create_dirs=create_dirs)

    def run_task(self, fw_spec):
        #TODO create a initialize_setup abstract function in SetupTask and put it there? or move somewhere else?
        #setup the FWTaskManager
        self.ftm = self.get_fw_task_manager(fw_spec)
        if 'previous_src' in fw_spec:
            self.prev_outdir = Directory(fw_spec['previous_src']['src_directories']['run_dir'])
        return super(VaspSetupTask, self).run_task(fw_spec)

    def setup_run_parameters(self, fw_spec, parameters=RUN_PARAMETERS):
        ftm = self.get_fw_task_manager(fw_spec=fw_spec)
        tm = ftm.task_manager
        qtk_params = self.task_helper.qtk_parallelization(self.vasp_input_set)
        mpi_procs = qtk_params.pop('mpi_procs', 24)
        qnodes = qtk_params.pop('qnodes', None)
        if len(qtk_params) != 0:
            raise ValueError('Too many parameters for qtk ...')
        pconf = ParalHints({}, [{'tot_ncpus': mpi_procs, 'mpi_ncpus': mpi_procs, 'efficiency': 1}])
        tm.select_qadapter(pconf)
        tm.qadapter.set_master_mem_overhead(mem_mb=1000)
        if 'timelimit' in fw_spec:
            tm.qadapter.set_timelimit(fw_spec['timelimit'])
        else:
            tm.qadapter.set_timelimit(86000)
        tm.qadapter.set_mpi_procs(mpi_procs)
        qtk_qadapter = tm.qadapter
        if qnodes is not None:
            qtk_qadapter.qnodes = qnodes

        return {'_queueadapter': qtk_qadapter.get_subs_dict(qnodes=qnodes), 'qtk_queueadapter': qtk_qadapter}

    def file_transfers(self, fw_spec):
        pass

    def fetch_previous_info(self, fw_spec):
        # Copy the appropriate dependencies in the in dir
        pass

    def prepare_run(self, fw_spec):
        # if it's the restart of a previous task, perform specific task updates.
        # perform these updates before writing the input, but after creating the dirs.
        if self.restart_info:
            #TODO check if this is the correct way of doing the restart
            # self.history.log_restart(self.restart_info)
            self.task_helper.restart(self.restart_info)

        # Write input files
        self.vasp_input_set.write_input(self.run_dir)

@explicit_serialize
class VaspRunTask(VaspSRCMixin, RunTask):


    def __init__(self, control_procedure, task_helper, task_type=None, custodian_handlers=None):
        if task_type is None:
            task_type = task_helper.task_type
        RunTask.__init__(self, control_procedure=control_procedure, task_type=task_type)
        self.task_helper = task_helper
        self.task_helper.set_task(self)
        self.custodian_handlers = custodian_handlers

    def config(self, fw_spec):
        self.ftm = self.get_fw_task_manager(fw_spec)
        self.setup_rundir(self.run_dir)

    def run(self, fw_spec):
        # class VaspJob(Job):
        #     """
        #     A basic vasp job. Just runs whatever is in the directory. But conceivably
        #     can be a complex processing of inputs etc. with initialization.
        #     """
        #
        #     def __init__(self, vasp_cmd, output_file="vasp.out", stderr_file="std_err.txt",
        #                  suffix="", final=True, backup=True, auto_npar=True,
        #                  auto_gamma=True, settings_override=None,
        #                  gamma_vasp_cmd=None, copy_magmom=False, auto_continue=False):
        try:
            vasp_cmd = os.environ['VASP_CMD'].split()
        except:
            raise ValueError('Unable to find vasp command')
        if 'custodian_jobs' in fw_spec:
            jobs = fw_spec['custodian_jobs']
        else:
            jobs = [VaspJob(vasp_cmd=vasp_cmd, auto_npar=False,
                            output_file=os.path.join(self.run_dir, 'vasp.out'),
                            stderr_file=os.path.join(self.run_dir, 'std_err.txt'),
                            backup=False,
                            auto_gamma=False)]
        custodian = Custodian(handlers=self.custodian_handlers, jobs=jobs,
                              validators=None, max_errors=10,
                              polling_time_step=10, monitor_freq=30)
        custodian.run()

    def postrun(self, fw_spec):
        # TODO should this be a general feature of the SRC?
        self.task_helper.conclude_task()
        return {'qtk_queueadapter' :fw_spec['qtk_queueadapter']}


@explicit_serialize
class VaspControlTask(VaspSRCMixin, ControlTask):

    def __init__(self, control_procedure, manager=None, max_restarts=10, src_cleaning=None, task_helper=None):
        ControlTask.__init__(self, control_procedure=control_procedure, manager=manager, max_restarts=max_restarts,
                             src_cleaning=src_cleaning)
        self.task_helper = task_helper

    def get_initial_objects_info(self, setup_fw, run_fw, src_directories):
        run_dir = src_directories['run_dir']
        run_task = run_fw.tasks[-1]
        run_task.setup_rundir(rundir=run_dir)
        task_helper = run_task.task_helper
        task_helper.set_task(run_task)
        init_obj_info = {'vasprun_xml_file': {'object': os.path.join(run_dir, 'vasprun.xml')},
                         'run_dir': {'object': run_dir},
                         'additional_vasp_wf_info': {'object': run_fw.spec['additional_vasp_wf_info']
                         if 'additional_vasp_wf_info' in run_fw.spec else {}}}

        return init_obj_info


@explicit_serialize
class GenerateVacanciesRelaxationTask(FireTaskBase):
    def __init__(self):
        from abiflows.fireworks.tasks.utility_tasks import print_myself
        print_myself()


    def run_task(self, fw_spec):
        pass
        # from magdesign.diffusion.neb_structures import generate_terminal_vacancies
        # terminal_vacancies = generate_terminal_vacancies(None, None)

    @serialize_fw
    def to_dict(self):
        return {}

    @classmethod
    def from_dict(cls, d):
        return cls()


@explicit_serialize
class GenerateNEBRelaxationTask(FireTaskBase):

    def __init__(self, n_insert=1, user_incar_settings=None, climbing_image=True, task_index=None,
                 terminal_start_task_type=None, terminal_end_task_type=None, prev_neb_task_type=None):
        if user_incar_settings is None:
            user_incar_settings = {}
        self.user_incar_settings = user_incar_settings
        self.n_insert = n_insert
        self.climbing_image = climbing_image
        self.task_index = task_index
        self.terminal_start_task_type = terminal_start_task_type
        self.terminal_end_task_type = terminal_end_task_type
        self.prev_neb_task_type = prev_neb_task_type

    def run_task(self, fw_spec):
        from magdesign.diffusion.neb_structures import neb_structures_insert_in_existing
        terminal_start_rundir = fw_spec['previous_fws'][self.terminal_start_task_type][0]['dir']
        terminal_end_rundir = fw_spec['previous_fws'][self.terminal_end_task_type][0]['dir']
        if 'structures' in fw_spec:
            if fw_spec['structures'] is not None:
                structs = neb_structures_insert_in_existing(fw_spec['structures'], n_insert=self.n_insert)
            else:
                # Get the structures from the previous nebs ...
                if len(fw_spec['previous_fws'][self.prev_neb_task_type]) != 1:
                    raise RuntimeError('Multiple or no fws with task_type "{}"'.format(self.prev_neb_task_type))
                prev_neb_rundir = fw_spec['previous_fws'][self.prev_neb_task_type][0]['dir']
                prev_neb_analysis = NEBAnalysis.from_dir(prev_neb_rundir,
                                                         relaxation_dirs=(terminal_start_rundir, terminal_end_rundir))
                structs = neb_structures_insert_in_existing(prev_neb_analysis.structures, n_insert=self.n_insert)
        else:
            if fw_spec['terminal_start'] is not None:
                structs = neb_structures_insert_in_existing([fw_spec['terminal_start'],
                                                             fw_spec['terminal_end']],
                                                            n_insert=self.n_insert)
            else:
                # Get the terminals from the relaxation of the terminals.
                if len(fw_spec['previous_fws'][self.terminal_start_task_type]) != 1:
                    raise RuntimeError('Multiple or no fws with task_type "{}"'.format(self.terminal_start_task_type))
                if len(fw_spec['previous_fws'][self.terminal_end_task_type]) != 1:
                    raise RuntimeError('Multiple or no fws with task_type "{}"'.format(self.terminal_end_task_type))
                terminal_start_rundir = fw_spec['previous_fws'][self.terminal_start_task_type][0]['dir']
                terminal_end_rundir = fw_spec['previous_fws'][self.terminal_end_task_type][0]['dir']
                terminal_start_vasprun = Vasprun(os.path.join(terminal_start_rundir, 'vasprun.xml'))
                terminal_end_vasprun = Vasprun(os.path.join(terminal_end_rundir, 'vasprun.xml'))
                terminal_start_structure = terminal_start_vasprun.final_structure
                terminal_end_structure = terminal_end_vasprun.final_structure
                structs = neb_structures_insert_in_existing([terminal_start_structure, terminal_end_structure],
                                                            n_insert=self.n_insert)
        if self.climbing_image:
            neb_vis = MPcNEBSet(structures=structs, user_incar_settings=self.user_incar_settings)
            task_helper = MPcNEBTaskHelper()
        else:
            neb_vis = MPNEBSet(structures=structs, user_incar_settings=self.user_incar_settings)
            task_helper = MPNEBTaskHelper()
        if 'additional_controllers' in fw_spec:
            additional_controllers = fw_spec['additional_controllers']
            fw_spec.pop('additional_controllers')
        else:
            additional_controllers = [WalltimeController(), MemoryController(), VaspNEBValidatorController()]

        control_procedure = ControlProcedure(controllers=additional_controllers)

        if self.task_index is None:
            # Define the task_index as "MPNEBVaspN" where N is the number of structures in the NEB (e.g. 3 when two end
            #  points plus one structure are computed)
            if self.climbing_image:
                task_index = 'MPcNEBVasp{:d}'.format(len(structs))
            else:
                task_index = 'MPNEBVasp{:d}'.format(len(structs))
        else:
            task_index = self.task_index

        task_index = SRCTaskIndex.from_any(task_index)
        task_type = task_index.task_type

        src_fws = createVaspSRCFireworks(vasp_input_set=neb_vis, task_helper=task_helper, task_type=task_type,
                                         control_procedure=control_procedure,
                                         custodian_handlers=[], max_restarts=10, src_cleaning=None,
                                         task_index=task_index,
                                         spec={'additional_vasp_wf_info': {'terminal_start_run_dir': terminal_start_rundir,
                                                                           'terminal_end_run_dir': terminal_end_rundir}},
                                         setup_spec_update=None,
                                         run_spec_update=None)
        wf = Workflow(fireworks=src_fws['fws'], links_dict=src_fws['links_dict'])
        return FWAction(detours=[wf])

    @serialize_fw
    def to_dict(self):
        return {"n_insert": self.n_insert,
                "user_incar_settings": self.user_incar_settings,
                "climbing_image": self.climbing_image,
                "task_index": self.task_index,
                "terminal_start_task_type": self.terminal_start_task_type,
                "terminal_end_task_type": self.terminal_end_task_type,
                "prev_neb_task_type": self.prev_neb_task_type
                }

    @classmethod
    def from_dict(cls, d):
        return cls(n_insert=d["n_insert"],
                   user_incar_settings=d["user_incar_settings"],
                   climbing_image=d["climbing_image"],
                   task_index=d["task_index"],
                   terminal_start_task_type=d["terminal_start_task_type"],
                   terminal_end_task_type=d["terminal_end_task_type"],
                   prev_neb_task_type=d["prev_neb_task_type"]
                   )


def createVaspSRCFireworks(vasp_input_set, task_helper, task_type, control_procedure,
                           custodian_handlers, max_restarts, src_cleaning, task_index, spec,
                           setup_spec_update=None, run_spec_update=None):
    # Make a full copy of the spec
    if spec is None:
        spec = {}
    spec = copy.deepcopy(spec)
    spec['_add_launchpad_and_fw_id'] = True
    spec['_add_fworker'] = True
    # Initialize the SRC task_index
    if task_index is not None:
        src_task_index = SRCTaskIndex.from_any(task_index)
    else:
        src_task_index = SRCTaskIndex.from_string(task_type)
    spec['SRC_task_index'] = src_task_index

    # SetupTask
    setup_spec = copy.deepcopy(spec)
    # Remove any initial queue_adapter_update from the spec
    setup_spec.pop('queue_adapter_update', None)

    setup_spec = set_short_single_core_to_spec(setup_spec)
    setup_spec['_preserve_fworker'] = True
    setup_spec['_pass_job_info'] = True
    setup_spec.update({} if setup_spec_update is None else setup_spec_update)
    setup_task = VaspSetupTask(vasp_input_set=vasp_input_set, deps=None, task_helper=task_helper, task_type=task_type)
    setup_fw = Firework(setup_task, spec=setup_spec, name=src_task_index.setup_str)

    # RunTask
    run_spec = copy.deepcopy(spec)
    run_spec['SRC_task_index'] = src_task_index
    run_spec['_preserve_fworker'] = True
    run_spec['_pass_job_info'] = True
    run_spec.update({} if run_spec_update is None else run_spec_update)
    run_task = VaspRunTask(control_procedure=control_procedure, task_helper=task_helper, task_type=task_type,
                           custodian_handlers=custodian_handlers)
    run_fw = Firework(run_task, spec=run_spec, name=src_task_index.run_str)

    # ControlTask
    control_spec = copy.deepcopy(spec)
    control_spec = set_short_single_core_to_spec(control_spec)
    control_spec['SRC_task_index'] = src_task_index
    control_spec['_allow_fizzled_parents'] = True
    control_task = VaspControlTask(control_procedure=control_procedure, manager=None, max_restarts=max_restarts,
                                   src_cleaning=src_cleaning, task_helper=task_helper)
    control_fw = Firework(control_task, spec=control_spec, name=src_task_index.control_str)

    links_dict = {setup_fw.fw_id: [run_fw.fw_id],
                  run_fw.fw_id: [control_fw.fw_id]}
    return {'setup_fw': setup_fw, 'run_fw': run_fw, 'control_fw': control_fw, 'links_dict': links_dict,
            'fws': [setup_fw, run_fw, control_fw]}

####################
# Helpers
####################

class VaspTaskHelper(MSONable):
    task_type = 'vasp'

    CRITICAL_EVENTS = []

    def __init__(self):
        self.task = None

    def qtk_parallelization(self, vasp_input_set):
        ftm = FWTaskManager.from_user_config()
        qadapters = ftm.task_manager.qads
        cpus_per_node = []
        cpus_min = []
        cpus_max = []
        for qad in qadapters:
            cpus_per_node.append(qad.hw.cores_per_node)
            cpus_min.append(qad.min_cores)
            cpus_max.append(qad.max_cores)
        return {'mpi_procs': 24, 'qnodes': 'exclusive'}

    def set_task(self, task):
        self.task = task

    def restart(self, restart_info):
        """
        Restart method. Each subclass should implement its own restart.
        """
        pass

    def prepare_restart(self):
        pass

    def conclude_task(self):
        pass

    def additional_update_spec(self):
        pass

    @pmg_serialize
    def as_dict(self):
        return {}

    @classmethod
    def from_dict(cls, d):
        return cls()


class MPRelaxTaskHelper(VaspTaskHelper):
    task_type = "MPRelaxVasp"

    def restart(self, restart_info):
        pass

    def get_final_structure(self):
        try:
            vasprun = Vasprun(os.path.join(self.task.run_dir, 'vasprun.xml'))
        except:
            raise ValueError('Failed to get final structure ...')
        return vasprun.final_structure


class MPNEBTaskHelper(VaspTaskHelper):
    task_type = "MPNEBVasp"

    def qtk_parallelization(self, vasp_input_set):
        ftm = FWTaskManager.from_user_config()
        qadapters = ftm.task_manager.qads
        cores_per_node = []
        cores_min = []
        cores_max = []
        for qad in qadapters:
            cores_per_node.append(qad.hw.cores_per_node)
            cores_min.append(qad.min_cores)
            cores_max.append(qad.max_cores)
        return {'mpi_procs': 24*(len(vasp_input_set.structures)-2), 'qnodes': 'exclusive'}

    def restart(self, restart_info):
        pass


class MPcNEBTaskHelper(MPNEBTaskHelper):
    task_type = "MPcNEBVasp"


####################
# Exceptions
####################

class HelperError(Exception):
    pass


class InitializationError(Exception):
    pass


class RestartError(Exception):
    pass


class WalltimeError(Exception):
    pass


class PostProcessError(Exception):
    pass

##############################
# Other objects
##############################

class RestartInfo(MSONable):
    """
    Object that contains the information about the restart of a task.
    """
    def __init__(self, previous_dir, reset=False):
        self.previous_dir = previous_dir
        self.reset = reset

    @pmg_serialize
    def as_dict(self):
        return dict(previous_dir=self.previous_dir, reset=self.reset)

    @classmethod
    def from_dict(cls, d):
        return cls(previous_dir=d['previous_dir'], reset=d['reset'])

    @property
    def prev_outdir(self):
        return Directory(self.previous_dir)

    @property
    def prev_indir(self):
        return Directory(self.previous_dir)