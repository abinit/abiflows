from __future__ import print_function, division, unicode_literals

import logging
import threading
import subprocess
from monty.json import MSONable
from abiflows.fireworks.tasks.src_tasks_abc import SetupTask, RunTask, ControlTask
from abiflows.core.mastermind_abc import ControllerNote
from abiflows.fireworks.utils.fw_utils import FWTaskManager
from fireworks import explicit_serialize
from pymatgen.serializers.json_coders import pmg_serialize
from pymatgen.io.abinit.utils import Directory

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
        qtk_qadapter = None

        return {'_queueadapter': qtk_qadapter.get_subs_dict(), 'qtk_queueadapter': qtk_qadapter}

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


    def __init__(self, control_procedure, task_helper, task_type=None):
        if task_type is None:
            task_type = task_helper.task_type
        RunTask.__init__(self, control_procedure=control_procedure, task_type=task_type)
        self.task_helper = task_helper
        self.task_helper.set_task(self)

    def config(self, fw_spec):
        self.ftm = self.get_fw_task_manager(fw_spec)
        self.setup_rundir(self.run_dir, create_dirs=False)

    def run(self, fw_spec):
        #TODO switch back to a simple process instead of a separate thread?
        def vasp_process():
            command = []
            #consider the case of serial execution
            if self.ftm.fw_policy.mpirun_cmd:
                command.extend(self.ftm.fw_policy.mpirun_cmd.split())
                if 'mpi_ncpus' in fw_spec:
                    command.extend(['-np', str(fw_spec['mpi_ncpus'])])
            command.append(self.ftm.fw_policy.vasp_cmd)
            with open('vasp.out', 'w') as stdout, open('vasp.err', 'w') as stderr:
                self.process = subprocess.Popen(command, stdout=stdout, stderr=stderr)

            (stdoutdata, stderrdata) = self.process.communicate()
            self.returncode = self.process.returncode

        # initialize returncode to avoid missing references in case of exception in the other thread
        self.returncode = None

        thread = threading.Thread(target=vasp_process)
        thread.start()
        thread.join()

    def postrun(self, fw_spec):
        #TODO should this be a general feature of the SRC?
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
        run_task.setup_rundir(rundir=run_dir, create_dirs=False, directories_only=False)
        task_helper = run_task.task_helper
        task_helper.set_task(run_task)
        init_obj_info = {}

        return init_obj_info

####################
# Helpers
####################

class VaspTaskHelper(MSONable):
    task_type = 'vasp'

    CRITICAL_EVENTS = []

    def __init__(self):
        self.task = None

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

class NEBTaskHelper(VaspTaskHelper):
    task_type = "NEB_vasp"

    def restart(self, restart_info):
        pass


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