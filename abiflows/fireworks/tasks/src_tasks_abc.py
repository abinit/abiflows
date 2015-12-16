from __future__ import print_function, division, unicode_literals

import abc
import copy
import inspect
import os

from fireworks.core.firework import FireTaskBase
from fireworks.core.firework import FWAction
from fireworks.core.firework import Workflow
from fireworks.core.firework import Firework
from fireworks.core.launchpad import LaunchPad
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.utilities.fw_serializers import serialize_fw

from monty.json import MontyDecoder, MontyEncoder
from monty.json import MSONable
from monty.serialization import loadfn
from monty.subprocess import Command

# from abiflows.fireworks.utils.custodian_utils import MonitoringSRCErrorHandler
# from abiflows.fireworks.utils.custodian_utils import SRCErrorHandler
# from abiflows.fireworks.utils.custodian_utils import SRCValidator
from abiflows.core.mastermind_abc import ControlProcedure, ControlledItemType
from abiflows.fireworks.utils.fw_utils import set_short_single_core_to_spec, get_short_single_core_spec


class SRCTaskMixin(object):

    src_type = ''

    @serialize_fw
    def to_dict(self):
        d = {}
        for arg in inspect.getargspec(self.__init__).args:
            if arg != "self":
                val = self.__getattribute__(arg)
                if hasattr(val, "as_dict"):
                    val = val.as_dict()
                elif isinstance(val, (tuple, list)):
                    val = [v.as_dict() if hasattr(v, "as_dict") else v for v in val]
                d[arg] = val

        return d

    @classmethod
    def from_dict(cls, d):
        dec = MontyDecoder()
        kwargs = {k: dec.process_decoded(v) for k, v in d.items()
                  if k in inspect.getargspec(cls.__init__).args}
        return cls(**kwargs)

    def setup_directories(self, fw_spec, create_dirs=False):
        if 'src_directories' in fw_spec:
            self.src_root_dir = fw_spec['src_directories']['src_root_dir']
        elif self.src_type == 'setup':
            self.src_root_dir = fw_spec.get('_launch_dir', os.getcwd())
        # elif self.src_type in ['run', 'control']:
        #     self.src_root_dir = os.path.split(os.path.abspath(fw_spec['_launch_dir']))[0]
        else:
            raise ValueError('Cannot setup directories for "src_type" = "{}"'.format(self.src_type))
        self.setup_dir = os.path.join(self.src_root_dir, 'setup')
        self.run_dir = os.path.join(self.src_root_dir, 'run')
        self.control_dir = os.path.join(self.src_root_dir, 'control')
        # if 'src_directories' in fw_spec:
        #     if (self.src_root_dir != fw_spec['src_directories']['src_root_dir'] or
        #         self.setup_dir != fw_spec['src_directories']['setup_dir'] or
        #         self.run_dir != fw_spec['src_directories']['run_dir'] or
        #         self.control_dir != fw_spec['src_directories']['control_dir']):
        #         raise ValueError('src_directories in fw_spec do not match actual SRC directories ...')
        if create_dirs:
            os.makedirs(self.setup_dir)
            os.makedirs(self.run_dir)
            os.makedirs(self.control_dir)

    @property
    def src_directories(self):
        return {'src_root_dir': self.src_root_dir,
                'setup_dir': self.setup_dir,
                'run_dir': self.run_dir,
                'control_dir': self.control_dir
                }


@explicit_serialize
class SetupTask(SRCTaskMixin, FireTaskBase):

    src_type = 'setup'

    RUN_PARAMETERS = ['_queueadapter', 'mpi_ncpus']

    def __init__(self):
        pass

    def run_task(self, fw_spec):
        # The Setup and Run have to run on the same worker
        #TODO: Check if this works ... I think it should ... is it clean ? ... Should'nt we put that in
        #      the SRC factory function instead ?
        #TODO: Be carefull here about preserver fworker when we recreate a new SRC trio ...
        fw_spec['_preserve_fworker'] = True
        fw_spec['_pass_job_info'] = True
        # Set up and create the directory tree of the Setup/Run/Control trio, forward directory information to run and
        #  control fireworks
        self.setup_directories(fw_spec=fw_spec, create_dirs=True)
        self.setup_run_and_control_dirs(fw_spec=fw_spec)
        # Move to the setup directory
        os.chdir(self.setup_dir)
        # Make the file transfers from another worker if needed
        self.file_transfers(fw_spec=fw_spec)
        # Setup the parameters for the run (number of cpus, time, memory, openmp, ...)
        params = list(self.RUN_PARAMETERS)
        if 'src_modified_objects' in fw_spec:
            for target, modified_object in fw_spec['src_modified_objects'].items():
                params.remove(target)
        run_parameters = self._setup_run_parameters(fw_spec=fw_spec, parameters=params)
        # Prepare run (make links to output files from previous tasks, write input files, create the directory
        # tree of the program, ...)
        self.prepare_run(fw_spec=fw_spec)

        # Update the spec of the Run Firework with the directory tree, the run_parameters obtained from
        #  setup_run_parameters and the modified_objects transferred directly from the Control Firework
        update_spec = {'src_directories': self.src_directories}
        update_spec.update(run_parameters)
        if 'src_modified_objects' in fw_spec:
            update_spec.update(fw_spec['src_modified_objects'])
        return FWAction(update_spec=update_spec)

    def _setup_run_parameters(self, fw_spec, parameters):
        params = self.setup_run_parameters(fw_spec=fw_spec)
        return {param: params[param] for param in parameters}

    def setup_run_parameters(self, fw_spec):
        qadapter_spec = get_short_single_core_spec()
        return {'_queueadapter': qadapter_spec, 'mpi_ncpus': 1}

    def file_transfers(self, fw_spec):
        pass

    def prepare_run(self, fw_spec):
        pass

    def setup_run_and_control_dirs(self, fw_spec):
        # Get the launchpad
        if '_add_launchpad_and_fw_id' in fw_spec:
            lp = self.launchpad
            setup_fw_id = self.fw_id
        else:
            try:
                fw_dict = loadfn('FW.json')
            except IOError:
                try:
                    fw_dict = loadfn('FW.yaml')
                except IOError:
                    raise RuntimeError("Launchpad/fw_id not present in spec and No FW.json nor FW.yaml file present: "
                                       "impossible to determine fw_id")
            lp = LaunchPad.auto_load()
            setup_fw_id = fw_dict['fw_id']
        # Check that this ControlTask has only one parent firework
        this_lzy_wf = lp.get_wf_by_fw_id_lzyfw(setup_fw_id)
        child_fw_ids = this_lzy_wf.links[setup_fw_id]
        if len(child_fw_ids) != 1:
            raise ValueError('SetupTask\'s Firework should have exactly one child firework')
        run_fw_id = child_fw_ids[0]
        child_run_fw_ids = this_lzy_wf.links[run_fw_id]
        if len(child_run_fw_ids) != 1:
            raise ValueError('RunTask\'s Firework should have exactly one child firework')
        control_fw_id = child_run_fw_ids[0]
        lp.update_spec(fw_ids=[run_fw_id],
                       spec_document={'_launch_dir': self.run_dir, 'src_directories': self.src_directories})
        lp.update_spec(fw_ids=[control_fw_id],
                       spec_document={'_launch_dir': self.control_dir, 'src_directories': self.src_directories})


class RunTask(SRCTaskMixin, FireTaskBase):

    src_type = 'run'

    def __init__(self, control_procedure):
        self.set_control_procedure(control_procedure=control_procedure)

    def set_control_procedure(self, control_procedure):
        self.control_procedure = control_procedure
        #TODO: check something here with the monitors ?

    def run_task(self, fw_spec):
        self.setup_directories(fw_spec=fw_spec, create_dirs=False)
        launch_dir = os.getcwd()
        # Move to the run directory
        os.chdir(self.run_dir)
        f = open(os.path.join(self.run_dir, 'fw_info.txt'), 'a')
        f.write('FW launch_directory :\n{}'.format(launch_dir))
        f.close()
        # The Run and Control tasks have to run on the same worker
        fw_spec['_preserve_fworker'] = True
        fw_spec['_pass_job_info'] = True
        #TODO: do something here with the monitoring controllers ...
        #      should stop the RunTask but the correction should be applied in control !
        self.config(fw_spec=fw_spec)
        self.run(fw_spec=fw_spec)
        update_spec = self.postrun(fw_spec=fw_spec)

        if update_spec is None:
            update_spec = {}

        #TODO: the directory is passed thanks to _pass_job_info. Should we pass anything else ?
        return FWAction(stored_data=None, exit=False, update_spec=None, mod_spec=None,
                        additions=None, detours=None,
                        defuse_children=False)

    def config(self, fw_spec):
        pass

    @abc.abstractmethod
    def run(self, fw_spec):
        pass

    def postrun(self, fw_spec):
        pass


@explicit_serialize
class ScriptRunTask(RunTask):

    task_type = 'script'

    def __init__(self, script_str, control_procedure):
        RunTask.__init__(self, control_procedure=control_procedure)
        self.script_str = script_str

    def run(self, fw_spec):
        cmd = Command(self.script_str)
        cmd.run()

    @serialize_fw
    def to_dict(self):
        return {'script_str': self.script_str,
                'control_procedure': self.control_procedure.as_dict()}

    @classmethod
    def from_dict(cls, d):
        control_procedure = ControlProcedure.from_dict(d['control_procedure'])
        return cls(script_str=d['script_str'], control_procedure=control_procedure)


@explicit_serialize
class ControlTask(SRCTaskMixin, FireTaskBase):
    src_type = 'control'

    def __init__(self, control_procedure, manager=None, max_restarts=10):
        self.control_procedure = control_procedure
        self.manager = manager
        self.max_restarts = max_restarts

    def run_task(self, fw_spec):
        self.setup_directories(fw_spec=fw_spec, create_dirs=False)
        launch_dir = os.getcwd()
        # Move to the control directory
        os.chdir(self.control_dir)
        f = open(os.path.join(self.control_dir, 'fw_info.txt'), 'a')
        f.write('FW launch_directory :\n{}'.format(launch_dir))
        f.close()
        # Get the task index
        task_index = SRCTaskIndex.from_any(fw_spec['SRC_task_index'])
        # Get the setup and run fireworks
        setup_and_run_fws = self.get_setup_and_run_fw(fw_spec=fw_spec)
        setup_fw = setup_and_run_fws['setup_fw']
        run_fw = setup_and_run_fws['run_fw']

        # Specify the type of the task that is controlled:
        #  - aborted : the task has been aborted due to a monitoring controller during the Run Task, the FW state
        #              is COMPLETED
        #  - completed : the task has completed, the FW state is COMPLETE
        #  - failed : the task has failed, the FW state is FIZZLED
        if run_fw.state == 'COMPLETED':
            if 'src_run_task_aborted' in fw_spec:
                self.control_procedure.set_controlled_item_type(ControlledItemType.task_aborted())
            else:
                self.control_procedure.set_controlled_item_type(ControlledItemType.task_completed())
        elif run_fw.state == 'FIZZLED':
            self.control_procedure.set_controlled_item_type(ControlledItemType.task_failed())
        else:
            raise RuntimeError('The state of the Run Firework is "{}" '
                               'while it should be COMPLETED or FIZZLED'.format(run_fw.state))

        # Get the keyword_arguments to be passed to the process method of the control_procedure
        #TODO: how to do that kind of automatically ??
        # each key should have : how to get it from the run_fw/(setup_fw)
        #                        how to force/apply it to the next SRC (e.g. how do we say to setup that)
        qerr_filepath = os.path.join(run_fw.launches[-1].launch_dir, 'queue.qerr')
        qout_filepath = os.path.join(run_fw.launches[-1].launch_dir, 'queue.qout')
        initial_objects = {'queue_adapter': run_fw.spec['qtk_queueadapter'],
                           'qerr_filepath': qerr_filepath,
                           'qout_filepath': qout_filepath}
        control_report = self.control_procedure.process(**initial_objects)

        # If everything is ok, update the spec of the children
        if control_report.finalized:
            stored_data = {'control_report': control_report, 'finalized': True}
            update_spec = {}
            mod_spec = []
            for task_type, task_info in fw_spec['previous_fws'].items():
                mod_spec.append({'_push_all': {'previous_fws->'+task_type: task_info}})
            return FWAction(stored_data=stored_data, exit=False, update_spec=update_spec, mod_spec=mod_spec,
                            additions=None, detours=None, defuse_children=False)

        # Check the maximum number of restarts
        if task_index.index == self.max_restarts:
            raise ValueError('Maximum number of restarts ({:d}) reached'.format(self.max_restarts))

        # Increase the task_index
        task_index.increase_index()

        # Apply the actions on the objects to get the modified objects (to be passed to SetupTask)
        modified_objects = {}
        for target, action in control_report.actions.items():
            # Special case right now for the queue adapter ...
            if target == 'queue_adapter':
                qtk_qadapter = action.apply(initial_objects[target])
                modified_objects['qtk_queueadapter'] = qtk_qadapter
                modified_objects['_queueadapter'] = qtk_qadapter.get_subs_dict()
            else:
                modified_objects[target] = action.apply(initial_objects[target])

        # If everything is ok, update the spec of the children
        stored_data = {}
        update_spec = {'src_modified_objects': modified_objects}
        mod_spec = []
        #TODO: what to do here ? Right now this should work, just transfer information from the run_fw to the
        # next SRC group
        for task_type, task_info in fw_spec['previous_fws'].items():
            mod_spec.append({'_push_all': {'previous_fws->'+task_type: task_info}})
        return FWAction(stored_data=stored_data, update_spec=update_spec, mod_spec=mod_spec)

    def get_setup_and_run_fw(self, fw_spec):
        # Get previous job information
        previous_job_info = fw_spec['_job_info']
        run_fw_id = previous_job_info['fw_id']
        # Get the launchpad
        if '_add_launchpad_and_fw_id' in fw_spec:
            lp = self.launchpad
            control_fw_id = self.fw_id
        else:
            try:
                fw_dict = loadfn('FW.json')
            except IOError:
                try:
                    fw_dict = loadfn('FW.yaml')
                except IOError:
                    raise RuntimeError("Launchpad/fw_id not present in spec and No FW.json nor FW.yaml file present: "
                                       "impossible to determine fw_id")
            lp = LaunchPad.auto_load()
            control_fw_id = fw_dict['fw_id']
        # Check that this ControlTask has only one parent firework
        this_lzy_wf = lp.get_wf_by_fw_id_lzyfw(control_fw_id)
        parents_fw_ids = this_lzy_wf.links.parent_links[control_fw_id]
        if len(parents_fw_ids) != 1:
            raise ValueError('ControlTask\'s Firework should have exactly one parent firework')
        # Get the Run Firework and its state
        run_fw = lp.get_fw_by_id(fw_id=run_fw_id)
        run_is_fizzled = '_fizzled_parents' in fw_spec
        if run_is_fizzled and not run_fw.state == 'FIZZLED':
            raise ValueError('ControlTask has "_fizzled_parents" key but parent Run firework is not fizzled ...')
        run_is_completed = run_fw.state == 'COMPLETED'
        if run_is_completed and run_is_fizzled:
            raise ValueError('Run firework is FIZZLED and COMPLETED ...')
        if (not run_is_completed) and (not run_is_fizzled):
            raise ValueError('Run firework is neither FIZZLED nor COMPLETED ...')
        # Get the Setup Firework
        setup_job_info = run_fw.spec['_job_info']
        setup_fw_id = setup_job_info['fw_id']
        setup_fw = lp.get_fw_by_id(fw_id=setup_fw_id)
        return {'setup_fw': setup_fw, 'run_fw': run_fw}

    @classmethod
    def from_controllers(cls, controllers, max_restarts=10):
        cp = ControlProcedure(controllers=controllers)
        return cls(control_procedure=cp, max_restarts=max_restarts)

    @serialize_fw
    def to_dict(self):
        enc = MontyEncoder()
        return {'control_procedure': self.control_procedure.as_dict(),
                'manager': self.manager.as_dict() if self.manager is not None else None,
                'max_restarts': self.max_restarts}

    @classmethod
    def from_dict(cls, d):
        control_procedure = ControlProcedure.from_dict(d['control_procedure'])
        dec = MontyDecoder()
        if d['manager'] is None:
            manager = None
        else:
            manager = dec.process_decoded(d['manager']),
        return cls(control_procedure=control_procedure, manager=manager, max_restarts=d['max_restarts'])



def createSRCFireworks(setup_task, run_task, control_task, spec=None, initialization_info=None,
                       task_index=None, deps=None):
    # Make a full copy of the spec
    if spec is None:
        spec = {}
    spec = copy.deepcopy(spec)
    # Initialize the SRC task_index
    if task_index is not None:
        src_task_index = SRCTaskIndex.from_any(task_index)
    else:
        src_task_index = SRCTaskIndex.from_task(run_task)
    spec['SRC_task_index'] = src_task_index

    # SetupTask
    setup_spec = copy.deepcopy(spec)
    # Remove any initial queue_adapter_update from the spec
    setup_spec.pop('queue_adapter_update', None)

    setup_spec = set_short_single_core_to_spec(setup_spec)
    setup_fw = Firework(setup_task, spec=setup_spec, name=src_task_index.setup_str)

    # RunTask
    run_spec = copy.deepcopy(spec)
    run_spec['SRC_task_index'] = src_task_index
    run_fw = Firework(run_task, spec=run_spec, name=src_task_index.run_str)

    # ControlTask
    control_spec = copy.deepcopy(spec)
    control_spec = set_short_single_core_to_spec(control_spec)
    control_spec['SRC_task_index'] = src_task_index
    control_fw = Firework(control_task, spec=run_spec, name=src_task_index.control_str)

    links_dict = {setup_fw.fw_id: [run_fw.fw_id],
                  run_fw.fw_id: [control_fw.fw_id]}
    return {'setup_fw': setup_fw, 'run_fw': run_fw, 'control_fw': control_fw, 'links_dict': links_dict,
            'fws': [setup_fw, run_fw, control_fw]}



# def createSRCFireworks(task_class, task_input, SRC_spec, initialization_info, wf_task_index_prefix, current_task_index=1,
#                        handlers=None, validators=None,
#                        deps=None, task_type=None, queue_adapter_update=None):
#     SRC_spec = copy.deepcopy(SRC_spec)
#     SRC_spec['initialization_info'] = initialization_info
#     SRC_spec['_add_launchpad_and_fw_id'] = True
#     SRC_spec['SRCScheme'] = True
#     prefix_allowed_chars = ['-']
#     prefix_test_string = str(wf_task_index_prefix)
#     for allowed_char in prefix_allowed_chars:
#         prefix_test_string = prefix_test_string.replace(allowed_char, "")
#     if not prefix_test_string.isalnum():
#         raise ValueError('wf_task_index_prefix should only contain letters '
#                          'and the following characters : {}'.format(prefix_test_string))
#     SRC_spec['wf_task_index_prefix'] = wf_task_index_prefix
#
#     # Remove any initial queue_adapter_update from the spec
#     SRC_spec.pop('queue_adapter_update', None)
#     if queue_adapter_update is not None:
#         SRC_spec['queue_adapter_update'] = queue_adapter_update
#
#     # Setup (Autoparal) run
#     SRC_spec_setup = copy.deepcopy(SRC_spec)
#     SRC_spec_setup = set_short_single_core_to_spec(SRC_spec_setup)
#     SRC_spec_setup['wf_task_index'] = '_'.join(['setup', wf_task_index_prefix, str(current_task_index)])
#     setup_task = task_class(task_input, is_autoparal=True, use_SRC_scheme=True, deps=deps, task_type=task_type)
#     setup_fw = Firework(setup_task, spec=SRC_spec_setup, name=SRC_spec_setup['wf_task_index'])
#     # Actual run of simulation
#     SRC_spec_run = copy.deepcopy(SRC_spec)
#     SRC_spec_run['wf_task_index'] = '_'.join(['run', wf_task_index_prefix, str(current_task_index)])
#     run_task = task_class(task_input, is_autoparal=False, use_SRC_scheme=True, deps=deps, task_type=task_type)
#     run_fw = Firework(run_task, spec=SRC_spec_run, name=SRC_spec_run['wf_task_index'])
#     # Check memory firework
#     SRC_spec_check = copy.deepcopy(SRC_spec)
#     SRC_spec_check = set_short_single_core_to_spec(SRC_spec_check)
#     SRC_spec_check['wf_task_index'] = '_'.join(['check', wf_task_index_prefix, str(current_task_index)])
#     check_task = CheckTask(handlers=handlers, validators=validators)
#     SRC_spec_check['_allow_fizzled_parents'] = True
#     check_fw = Firework(check_task, spec=SRC_spec_check, name=SRC_spec_check['wf_task_index'])
#     links_dict = {setup_fw.fw_id: [run_fw.fw_id],
#                   run_fw.fw_id: [check_fw.fw_id]}
#     return {'setup_fw': setup_fw, 'run_fw': run_fw, 'check_fw': check_fw, 'links_dict': links_dict,
#             'fws': [setup_fw, run_fw, check_fw]}


class SRCTaskIndex(MSONable):

    ALLOWED_CHARS = ['-']

    def __init__(self, task_type, index=1):
        self.set_task_type(task_type=task_type)
        self.index = index

    def set_task_type(self, task_type):
        prefix_test_string = str(task_type)
        for allowed_char in self.ALLOWED_CHARS:
            prefix_test_string = prefix_test_string.replace(allowed_char, "")
        if not prefix_test_string.isalpha():
            ac_str = ', '.join(['"{}"'.format(ac) for ac in self.ALLOWED_CHARS])
            raise ValueError('task_type should only contain letters '
                             'and the following characters : {}'.format(ac_str))
        self.task_type = task_type

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        if isinstance(index, int):
            self._index = index
        elif isinstance(index, str):
            try:
                myindex = int(index)
                self._index = myindex
            except:
                raise ValueError('Index in SRCTaskIndex should be an integer or a string '
                                 'that can be cast into an integer')
        else:
            raise ValueError('Index in SRCTaskIndex should be an integer or a string '
                             'that can be cast into an integer')

    def increase_index(self):
        self.index += 1

    def __add__(self, other):
        if not isinstance(other, int):
            raise ValueError('The __add__ method in SRCTaskIndex should be an integer')
        self.index += other

    def __str__(self):
        return '_'.join([self.task_type, str(self.index)])

    @property
    def setup_str(self):
        return '_'.join(['setup', self.__str__()])

    @property
    def run_str(self):
        return '_'.join(['run', self.__str__()])

    @property
    def control_str(self):
        return '_'.join(['control', self.__str__()])

    @classmethod
    def from_string(cls, SRC_task_index_string):
        sp = SRC_task_index_string.split('_')
        if len(sp) not in [2, 3]:
            raise ValueError('SRC_task_index_string should contain 1 or 2 underscores ("_") '
                             'while it contains {:d}'.format(len(sp)-1))
        if any([len(part) == 0 for part in sp]):
            raise ValueError('SRC_task_index_string has an empty part when separated by underscores ...')
        if len(sp) == 2:
            return cls(task_type=sp[0], index=sp[1])
        elif len(sp) == 3:
            if sp[0] not in ['setup', 'run', 'control']:
                raise ValueError('SRC_task_index_string should start with "setup", "run" or "control" when 3 parts are '
                                 'identified')
            return cls(task_type=sp[1], index=sp[2])

    @classmethod
    def from_any(cls, SRC_task_index):
        if isinstance(SRC_task_index, str):
            return cls.from_string(SRC_task_index)
        elif isinstance(SRC_task_index, SRCTaskIndex):
            return cls(task_type=SRC_task_index.task_type, index=SRC_task_index.index)
        else:
            raise ValueError('SRC_task_index should be an instance of "str" or "SRCTaskIndex" '
                             'in "from_any" class method')

    @classmethod
    def from_task(cls, task):
        return cls(task_type=task.task_type)

    @classmethod
    def from_dict(cls, d):
        return cls(task_type=d['task_type'], index=d['index'])

    def as_dict(self):
        return {'@class': self.__class__.__name__,
                '@module': self.__class__.__module__,
                'task_type': self.task_type,
                'index': self.index}



def get_queue_adapter_update(qtk_queueadapter, corrections, qa_params=None):
    if qa_params is None:
        qa_params = ['timelimit', 'mem_per_proc', 'master_mem_overhead']
    queue_adapter_update = {}
    for qa_param in qa_params:
        if qa_param == 'timelimit':
            queue_adapter_update[qa_param] = qtk_queueadapter.timelimit
        elif qa_param == 'mem_per_proc':
            queue_adapter_update[qa_param] = qtk_queueadapter.mem_per_proc
        elif qa_param == 'master_mem_overhead':
            queue_adapter_update[qa_param] = qtk_queueadapter.master_mem_overhead
        else:
            raise ValueError('Wrong queue adapter parameter for update')
    for correction in corrections:
        for action in correction['actions']:
            if action['object']['key'] == 'qtk_queueadapter':
                qa_update = action['action']['_set']
                queue_adapter_update.update(qa_update)
    return queue_adapter_update


################
# Exceptions
################

class SRCError(Exception):
    pass


class SetupError(SRCError):
    pass


class RunError(SRCError):
    pass


class ControlError(SRCError):
    pass


# TODO: Remove the following when the new SRC is working ...
#
# class CheckTask(SRCTaskMixin, FireTaskBase):
#
#     src_type = 'check'
#
#     def __init__(self, handlers=None, validators=None, max_restarts=10):
#         self.set_handlers(handlers=handlers)
#         self.set_validators(validators=validators)
#
#         self.max_restarts = max_restarts
#
#     def set_handlers(self, handlers):
#         if handlers is None:
#             self.handlers = []
#         elif issubclass(handlers, SRCErrorHandler):
#             self.handlers = [handlers]
#         elif isinstance(handlers, (list, tuple)):
#             self.handlers = []
#             for handler in handlers:
#                 if not issubclass(handler, SRCErrorHandler):
#                     raise TypeError('One of items in "handlers" does not derive from '
#                                     'SRCErrorHandler')
#                 self.handlers.append(handler)
#         else:
#             raise TypeError('The handlers argument is neither None, nor a SRCErrorHandler, '
#                             'nor a list/tuple')
#         # Check that there is only one FIRST and one LAST handler (PRIORITY_FIRST and PRIORITY_LAST)
#         # and sort handlers by their priority
#         if self.handlers is not None:
#             h_priorities = [h.handler_priority for h in self.handlers]
#             nhfirst = h_priorities.count(SRCErrorHandler.PRIORITY_FIRST)
#             nhlast = h_priorities.count(SRCErrorHandler.PRIORITY_LAST)
#             if nhfirst > 1 or nhlast > 1:
#                 raise ValueError('More than one first or last handler :\n'
#                                  ' - nfirst : {:d}\n - nlast : {:d}'.format(nhfirst,
#                                                                             nhlast))
#             self.handlers = sorted([h for h in self.handlers if h.allow_completed],
#                                          key=lambda x: x.handler_priority)
#
#     def set_validators(self, validators):
#         if validators is None:
#             self.validators = []
#         elif issubclass(validators, SRCValidator):
#             self.validators = [validators]
#         elif isinstance(validators, (list, tuple)):
#             self.validators = []
#             for validator in validators:
#                 if not issubclass(validators, SRCValidator):
#                     raise TypeError('One of items in "validators" does not derive from '
#                                     'SRCValidator')
#                 self.validators.append(validator)
#         else:
#             raise TypeError('The validators argument is neither None, nor a SRCValidator, '
#                             'nor a list/tuple')
#         # Check that there is only one FIRST and one LAST validator (PRIORITY_FIRST and PRIORITY_LAST)
#         # and sort validators by their priority
#         if self.validators is not None:
#             v_priorities = [v.validator_priority for v in self.validators]
#             nvfirst = v_priorities.count(SRCValidator.PRIORITY_FIRST)
#             nvlast = v_priorities.count(SRCValidator.PRIORITY_LAST)
#             if nvfirst > 1 or nvlast > 1:
#                 raise ValueError('More than one first or last validator :\n'
#                                  ' - nfirst : {:d}\n - nlast : {:d}'.format(nvfirst,
#                                                                             nvlast))
#             self.validators = sorted([v for v in self.validators],
#                                          key=lambda x: x.validator_priority)
#
#     def run_task(self, fw_spec):
#         # Get the setup and run fireworks
#         setup_and_run_fws = self.get_setup_and_run_fw(fw_spec=fw_spec)
#         setup_fw = setup_and_run_fws['setup_fw']
#         run_fw = setup_and_run_fws['run_fw']
#         # Get the handlers
#         handlers = self.get_handlers(run_fw=run_fw)
#
#         # Check/detect errors and get the corrections
#         corrections = self.get_corrections(fw_spec=fw_spec, run_fw=run_fw, handlers=handlers)
#
#         # In case of a fizzled parent, at least one correction is needed !
#         if run_fw.state == 'FIZZLED' and len(corrections) == 0:
#             # TODO: should we do something else here ? like return a FWAction with defuse_childrens = True ??
#             raise RuntimeError('No corrections found for fizzled firework ...')
#
#         # If some errors were found, apply the corrections and return the FWAction
#         if len(corrections) > 0:
#             fw_action = self.apply_corrections(setup_fw=setup_fw, run_fw=run_fw, corrections=corrections)
#             return fw_action
#
#         # Validate the results if no error was found
#         self.validate()
#
#         # If everything is ok, update the spec of the children
#         stored_data = {}
#         update_spec = {}
#         mod_spec = []
#         #TODO: what to do here ? Right now this should work, just transfer information from the run_fw to the
#         # next SRC group
#         for task_type, task_info in fw_spec['previous_fws'].items():
#             mod_spec.append({'_push_all': {'previous_fws->'+task_type: task_info}})
#         return FWAction(stored_data=stored_data, update_spec=update_spec, mod_spec=mod_spec)
#
#     def get_setup_and_run_fw(self, fw_spec):
#         # Get previous job information
#         previous_job_info = fw_spec['_job_info']
#         run_fw_id = previous_job_info['fw_id']
#         # Get the launchpad
#         if '_add_launchpad_and_fw_id' in fw_spec:
#             lp = self.launchpad
#             check_fw_id = self.fw_id
#         else:
#             try:
#                 fw_dict = loadfn('FW.json')
#             except IOError:
#                 try:
#                     fw_dict = loadfn('FW.yaml')
#                 except IOError:
#                     raise RuntimeError("Launchpad/fw_id not present in spec and No FW.json nor FW.yaml file present: "
#                                        "impossible to determine fw_id")
#             lp = LaunchPad.auto_load()
#             check_fw_id = fw_dict['fw_id']
#         # Check that this CheckTask has only one parent firework
#         this_lzy_wf = lp.get_wf_by_fw_id_lzyfw(check_fw_id)
#         parents_fw_ids = this_lzy_wf.links.parent_links[check_fw_id]
#         if len(parents_fw_ids) != 1:
#             raise ValueError('CheckTask\'s Firework should have exactly one parent firework')
#         # Get the Run Firework and its state
#         run_fw = lp.get_fw_by_id(fw_id=run_fw_id)
#         run_is_fizzled = '_fizzled_parents' in fw_spec
#         if run_is_fizzled and not run_fw.state == 'FIZZLED':
#             raise ValueError('CheckTask has "_fizzled_parents" key but parent Run firework is not fizzled ...')
#         run_is_completed = run_fw.state == 'COMPLETED'
#         if run_is_completed and run_is_fizzled:
#             raise ValueError('Run firework is FIZZLED and COMPLETED ...')
#         if (not run_is_completed) and (not run_is_fizzled):
#             raise ValueError('Run firework is neither FIZZLED nor COMPLETED ...')
#         # Get the Setup Firework
#         setup_job_info = run_fw.spec['_job_info']
#         setup_fw_id = setup_job_info['fw_id']
#         setup_fw = lp.get_fw_by_id(fw_id=setup_fw_id)
#         return {'setup_fw': setup_fw, 'run_fw': run_fw}
#
#     def get_handlers(self, run_fw):
#         if run_fw.state == 'FIZZLED':
#             handlers = [h for h in self.handlers if h.allow_fizzled]
#         elif run_fw.state == 'COMPLETED':
#             handlers = [h for h in self.handlers if h.allow_completed]
#         else:
#             raise ValueError('Run firework is neither FIZZLED nor COMPLETED ...')
#         return handlers
#
#     def get_corrections(self, fw_spec, run_fw, handlers):
#         #TODO: we should add here something about the corrections that have already been applied and that cannot
#         #      be applied anymore ...
#         corrections = []
#         for handler in handlers:
#             # Set needed data for the handlers (the spec of this check task/fw and the fw that has to be checked)
#             handler.src_setup(fw_spec=fw_spec, fw_to_check=run_fw)
#             if handler.check():
#                 corrections.append(handler.correct())
#                 if handler.skip_remaining_handlers:
#                     break
#         return corrections
#
#     # def get_corrections(self, fw_spec, run_fw, handlers):
#     #     #TODO: we should add here something about the corrections that have already been applied and that cannot
#     #     #      be applied anymore ...
#     #     corrections = []
#     #     for handler in handlers:
#     #         # Set needed data for the handlers (the spec of this check task/fw and the fw that has to be checked)
#     #         handler.src_setup(fw_spec=fw_spec, fw_to_check=run_fw)
#     #         if handler.check():
#     #             # TODO: add something whether we have a possible correction here or not ? has_correction() in handler ?
#     #             if handler.has_correction():
#     #                 corrections.append(handler.correct())
#     #                 if handler.skip_remaining_handlers:
#     #                     break
#     #             else:
#     #                 if handler.correction_is_required():
#     #                     raise ValueError('No correction found for error while correction is required ...')
#     #     return corrections
#
#     def validate(self):
#         validators = self.validators if self.validators is not None else []
#         for validator in validators:
#             if not validator.check():
#                 raise RuntimeError('Validator invalidate results ...')
#
#     def apply_corrections(self, setup_fw, run_fw, corrections):
#         spec = copy.deepcopy(run_fw.spec)
#         modder = Modder()
#         for correction in corrections:
#             actions = correction['actions']
#             for action in actions:
#                 if action['action_type'] == 'modify_object':
#                     if action['object']['source'] == 'fw_spec':
#                         myobject = spec[action['object']['key']]
#                     else:
#                         raise NotImplementedError('Object source "{}" not implemented in '
#                                                   'CheckTask'.format(action['object']['source']))
#                     newobj = modder.modify_object(action['action'], myobject)
#                     spec[action['object']['key']] = newobj
#                 elif action['action_type'] == 'modify_dict':
#                     if action['dict']['source'] == 'fw_spec':
#                         mydict = spec[action['dict']['key']]
#                     else:
#                         raise NotImplementedError('Dict source "{}" not implemented in '
#                                                   'CheckTask'.format(action['dict']['source']))
#                     modder.modify(action['action'], mydict)
#                 else:
#                     raise NotImplementedError('Action type "{}" not implemented in '
#                                               'CheckTask'.format(action['action_type']))
#         # Keep track of the corrections that have been applied
#         if 'SRC_check_corrections' in spec:
#             spec['SRC_check_corrections'].extend(corrections)
#         else:
#             spec['SRC_check_corrections'] = corrections
#
#         # Update the task index
#         task_index = SRCTaskIndex.from_any(spec['SRC_task_index'])
#         task_index.increase_index()
#
#         #TODO: in the future, see whether the FW queueadapter might be replaced by the qtk_queueadapter ?
#         #      ... to be discussed with Anubhav, when the qtk queueadapter is in a qtk toolkit and not anymore
#         #          in pymatgen/io/abinit
#         #      in that case, this part will not be needed anymore ...
#         spec['_queueadapter'] = spec['qtk_queueadapter'].get_subs_dict()
#         #TODO: why do we need this again ??
#         queue_adapter_update = get_queue_adapter_update(qtk_queueadapter=spec['qtk_queueadapter'],
#                                                         corrections=corrections)
#
#         run_task = run_fw.tasks[0]
#         initialization_info = run_task.spec['initialization_info']
#         deps = run_task.deps
#         setup_task = setup_fw.tasks[0]
#
#         check_task = self
#
#         # Create the new Setup/Run/Check fireworks
#         #TODO: do we need initialization info here ?
#         #      do we need deps here ?
#         SRC_fws = createSRCFireworks(setup_task=setup_task, run_task=run_task, check_task=check_task,
#                                      spec=spec, initialization_info=initialization_info,
#                                      task_index=task_index, deps=deps)
#         wf = Workflow(fireworks=SRC_fws['fws'], links_dict=SRC_fws['links_dict'])
#         return FWAction(detours=[wf])