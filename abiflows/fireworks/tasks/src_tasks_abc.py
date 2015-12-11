
import abc
import copy
import inspect
import os

from custodian.ansible.interpreter import Modder

from fireworks.core.firework import FireTaskBase
from fireworks.core.firework import FWAction
from fireworks.core.firework import Workflow
from fireworks.core.launchpad import LaunchPad
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.utilities.fw_serializers import serialize_fw

from monty.json import MontyDecoder
from monty.serialization import loadfn

from abiflows.fireworks.utils.custodian_utils import MonitoringSRCErrorHandler
from abiflows.fireworks.utils.custodian_utils import SRCErrorHandler
from abiflows.fireworks.utils.custodian_utils import SRCValidator
from abiflows.core.mastermind_abc import ControlBarrier, ControlledItemType


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
        if self.src_type == 'setup':
            self.src_root_dir = fw_spec['_launch_dir']
        elif self.src_type in ['run', 'check']:
            self.src_root_dir = os.path.split(os.path.abspath(fw_spec['_launch_dir']))[0]
        else:
            raise ValueError('Cannot setup directories for "src_type" = "{}"'.format(self.src_type))
        self.setup_dir = os.path.join(self.src_root_dir, 'setup')
        self.run_dir = os.path.join(self.src_root_dir, 'run')
        self.check_dir = os.path.join(self.src_root_dir, 'check')
        if 'src_directories' in fw_spec:
            if (self.src_root_dir != fw_spec['src_directories']['src_root_dir'] or
                self.setup_dir != fw_spec['src_directories']['setup_dir'] or
                self.run_dir != fw_spec['src_directories']['run_dir'] or
                self.check_dir != fw_spec['src_directories']['check_dir']):
                raise ValueError('src_directories in fw_spec do not match actual SRC directories ...')
        if create_dirs:
            os.makedirs(self.setup_dir)
            os.makedirs(self.run_dir)
            os.makedirs(self.check_dir)

    @property
    def src_directories(self):
        return {'src_root_dir': self.src_root_dir,
                'setup_dir': self.setup_dir,
                'run_dir': self.run_dir,
                'check_dir': self.check_dir
                }


@explicit_serialize
class SetupTask(FireTaskBase, SRCTaskMixin):

    src_type = 'setup'

    def run_task(self, fw_spec):
        # The Setup and Run have to run on the same worker
        #TODO: Check if this works ... I think it should ... is it clean ? ... Should'nt we put that in
        #      the SRC factory function instead ?
        #TODO: Be carefull here about preserver fworker when we recreate a new SRC trio ...
        fw_spec['_preserve_fworker'] = True
        fw_spec['_pass_job_info'] = True
        # Set up and create the directory tree of the Setup/Run/Check trio
        self.setup_directories(fw_spec=fw_spec, create_dirs=True)
        # Move to the setup directory
        os.chdir(self.setup_dir)
        # Make the file transfers from another worker if needed
        self.file_transfers(fw_spec=fw_spec)
        # Setup the parameters for the run (number of cpus, time, memory, openmp, ...)
        self.setup_run_parameters(fw_spec=fw_spec)
        # Prepare run (make links to output files from previous tasks, write input files, create the directory
        # tree of the program, ...)
        self.prepare_run(fw_spec=fw_spec)

        update_spec = {'src_directories': self.src_directories}
        return FWAction(update_spec=update_spec)

    @abc.abstractmethod
    def setup_run_parameters(self, fw_spec):
        pass

    @abc.abstractmethod
    def file_transfers(self, fw_spec):
        pass

    @abc.abstractmethod
    def prepare_run(self, fw_spec):
        pass


@explicit_serialize
class RunTask(FireTaskBase, SRCTaskMixin):

    src_type = 'run'

    def __init__(self, monitoring_handlers=None):
        # TODO: evaluate the possibility to use
        self.set_monitoring_handlers(monitoring_handlers=monitoring_handlers)

    def set_monitoring_handlers(self, monitoring_handlers):
        if monitoring_handlers is None:
            self.monitoring_handlers = []
        elif issubclass(monitoring_handlers, MonitoringSRCErrorHandler):
            self.monitoring_handlers = [monitoring_handlers]
        elif isinstance(monitoring_handlers, (list, tuple)):
            self.monitoring_handlers = []
            for mh in monitoring_handlers:
                if not issubclass(mh, MonitoringSRCErrorHandler):
                    raise TypeError('One of items in "monitoring_handlers" does not derive from '
                                    'MonitoringSRCErrorHandler')
                self.monitoring_handlers.append(mh)
        else:
            raise TypeError('The monitoring_handlers argument is neither None, nor a MonitoringSRCErrorHandler, '
                            'nor a list/tuple')

    def run_task(self, fw_spec):
        # The Run and Check tasks have to run on the same worker
        fw_spec['_preserve_fworker'] = True
        fw_spec['_pass_job_info'] = True
        #TODO: do something here with the monitoring handlers ... should stop the RunTask but the correction should be
        #      applied in check !
        self.config(fw_spec=fw_spec)
        self.run(fw_spec=fw_spec)
        self.postrun(fw_spec=fw_spec)

        #TODO: the directory is passed thanks to _pass_job_info. Should we pass anything else ?
        return FWAction(stored_data=None, exit=False, update_spec=None, mod_spec=None,
                        additions=None, detours=None,
                        defuse_children=False)

    @abc.abstractmethod
    def config(self, fw_spec):
        pass

    @abc.abstractmethod
    def run(self, fw_spec):
            pass

    @abc.abstractmethod
    def postrun(self, fw_spec):
        pass


@explicit_serialize
class ControlTask(FireTaskBase, SRCTaskMixin):
    src_type = 'control'

    def __init__(self, control_barrier, max_restarts=10):
        self.control_barrier = control_barrier
        self.max_restarts = max_restarts

    def run_task(self, fw_spec):
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
                self.control_barrier.set_controlled_item_type(ControlledItemType.task_aborted())
            else:
                self.control_barrier.set_controlled_item_type(ControlledItemType.task_completed())
        elif run_fw.state == 'FIZZLED':
            self.control_barrier.set_controlled_item_type(ControlledItemType.task_failed())
        else:
            raise RuntimeError('The state of the Run Firework is "{}" '
                               'while it should be COMPLETED or FIZZLED'.format(run_fw.state))

        # Get the keyword_arguments to be passed to the process method of the control_barrier
        #TODO: how to do that kind of automatically ??
        kwargs = {'queue_adapter': run_fw.spec['qtk_queueadapter']}
        control_report = self.control_barrier.process(**kwargs)

        if control_report.finalized:
            # If everything is ok, update the spec of the children
            stored_data = {'finalized': True}
            update_spec = {}
            mod_spec = []
            for task_type, task_info in fw_spec['previous_fws'].items():
                mod_spec.append({'_push_all': {'previous_fws->'+task_type: task_info}})
            return FWAction(stored_data=stored_data)



        # Validate the results if no error was found
        self.validate()

        # If everything is ok, update the spec of the children
        stored_data = {}
        update_spec = {}
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
            check_fw_id = self.fw_id
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
            check_fw_id = fw_dict['fw_id']
        # Check that this CheckTask has only one parent firework
        this_lzy_wf = lp.get_wf_by_fw_id_lzyfw(check_fw_id)
        parents_fw_ids = this_lzy_wf.links.parent_links[check_fw_id]
        if len(parents_fw_ids) != 1:
            raise ValueError('CheckTask\'s Firework should have exactly one parent firework')
        # Get the Run Firework and its state
        run_fw = lp.get_fw_by_id(fw_id=run_fw_id)
        run_is_fizzled = '_fizzled_parents' in fw_spec
        if run_is_fizzled and not run_fw.state == 'FIZZLED':
            raise ValueError('CheckTask has "_fizzled_parents" key but parent Run firework is not fizzled ...')
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
        cb = ControlBarrier(controllers=controllers)
        return cls(control_barrier=cb, max_restarts=max_restarts)


@explicit_serialize
class CheckTask(FireTaskBase, SRCTaskMixin):

    src_type = 'check'

    def __init__(self, handlers=None, validators=None, max_restarts=10):
        self.set_handlers(handlers=handlers)
        self.set_validators(validators=validators)

        self.max_restarts = max_restarts

    def set_handlers(self, handlers):
        if handlers is None:
            self.handlers = []
        elif issubclass(handlers, SRCErrorHandler):
            self.handlers = [handlers]
        elif isinstance(handlers, (list, tuple)):
            self.handlers = []
            for handler in handlers:
                if not issubclass(handler, SRCErrorHandler):
                    raise TypeError('One of items in "handlers" does not derive from '
                                    'SRCErrorHandler')
                self.handlers.append(handler)
        else:
            raise TypeError('The handlers argument is neither None, nor a SRCErrorHandler, '
                            'nor a list/tuple')
        # Check that there is only one FIRST and one LAST handler (PRIORITY_FIRST and PRIORITY_LAST)
        # and sort handlers by their priority
        if self.handlers is not None:
            h_priorities = [h.handler_priority for h in self.handlers]
            nhfirst = h_priorities.count(SRCErrorHandler.PRIORITY_FIRST)
            nhlast = h_priorities.count(SRCErrorHandler.PRIORITY_LAST)
            if nhfirst > 1 or nhlast > 1:
                raise ValueError('More than one first or last handler :\n'
                                 ' - nfirst : {:d}\n - nlast : {:d}'.format(nhfirst,
                                                                            nhlast))
            self.handlers = sorted([h for h in self.handlers if h.allow_completed],
                                         key=lambda x: x.handler_priority)

    def set_validators(self, validators):
        if validators is None:
            self.validators = []
        elif issubclass(validators, SRCValidator):
            self.validators = [validators]
        elif isinstance(validators, (list, tuple)):
            self.validators = []
            for validator in validators:
                if not issubclass(validators, SRCValidator):
                    raise TypeError('One of items in "validators" does not derive from '
                                    'SRCValidator')
                self.validators.append(validator)
        else:
            raise TypeError('The validators argument is neither None, nor a SRCValidator, '
                            'nor a list/tuple')
        # Check that there is only one FIRST and one LAST validator (PRIORITY_FIRST and PRIORITY_LAST)
        # and sort validators by their priority
        if self.validators is not None:
            v_priorities = [v.validator_priority for v in self.validators]
            nvfirst = v_priorities.count(SRCValidator.PRIORITY_FIRST)
            nvlast = v_priorities.count(SRCValidator.PRIORITY_LAST)
            if nvfirst > 1 or nvlast > 1:
                raise ValueError('More than one first or last validator :\n'
                                 ' - nfirst : {:d}\n - nlast : {:d}'.format(nvfirst,
                                                                            nvlast))
            self.validators = sorted([v for v in self.validators],
                                         key=lambda x: x.validator_priority)

    def run_task(self, fw_spec):
        # Get the setup and run fireworks
        setup_and_run_fws = self.get_setup_and_run_fw(fw_spec=fw_spec)
        setup_fw = setup_and_run_fws['setup_fw']
        run_fw = setup_and_run_fws['run_fw']
        # Get the handlers
        handlers = self.get_handlers(run_fw=run_fw)

        # Check/detect errors and get the corrections
        corrections = self.get_corrections(fw_spec=fw_spec, run_fw=run_fw, handlers=handlers)

        # In case of a fizzled parent, at least one correction is needed !
        if run_fw.state == 'FIZZLED' and len(corrections) == 0:
            # TODO: should we do something else here ? like return a FWAction with defuse_childrens = True ??
            raise RuntimeError('No corrections found for fizzled firework ...')

        # If some errors were found, apply the corrections and return the FWAction
        if len(corrections) > 0:
            fw_action = self.apply_corrections(setup_fw=setup_fw, run_fw=run_fw, corrections=corrections)
            return fw_action

        # Validate the results if no error was found
        self.validate()

        # If everything is ok, update the spec of the children
        stored_data = {}
        update_spec = {}
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
            check_fw_id = self.fw_id
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
            check_fw_id = fw_dict['fw_id']
        # Check that this CheckTask has only one parent firework
        this_lzy_wf = lp.get_wf_by_fw_id_lzyfw(check_fw_id)
        parents_fw_ids = this_lzy_wf.links.parent_links[check_fw_id]
        if len(parents_fw_ids) != 1:
            raise ValueError('CheckTask\'s Firework should have exactly one parent firework')
        # Get the Run Firework and its state
        run_fw = lp.get_fw_by_id(fw_id=run_fw_id)
        run_is_fizzled = '_fizzled_parents' in fw_spec
        if run_is_fizzled and not run_fw.state == 'FIZZLED':
            raise ValueError('CheckTask has "_fizzled_parents" key but parent Run firework is not fizzled ...')
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

    def get_handlers(self, run_fw):
        if run_fw.state == 'FIZZLED':
            handlers = [h for h in self.handlers if h.allow_fizzled]
        elif run_fw.state == 'COMPLETED':
            handlers = [h for h in self.handlers if h.allow_completed]
        else:
            raise ValueError('Run firework is neither FIZZLED nor COMPLETED ...')
        return handlers

    def get_corrections(self, fw_spec, run_fw, handlers):
        #TODO: we should add here something about the corrections that have already been applied and that cannot
        #      be applied anymore ...
        corrections = []
        for handler in handlers:
            # Set needed data for the handlers (the spec of this check task/fw and the fw that has to be checked)
            handler.src_setup(fw_spec=fw_spec, fw_to_check=run_fw)
            if handler.check():
                corrections.append(handler.correct())
                if handler.skip_remaining_handlers:
                    break
        return corrections

    # def get_corrections(self, fw_spec, run_fw, handlers):
    #     #TODO: we should add here something about the corrections that have already been applied and that cannot
    #     #      be applied anymore ...
    #     corrections = []
    #     for handler in handlers:
    #         # Set needed data for the handlers (the spec of this check task/fw and the fw that has to be checked)
    #         handler.src_setup(fw_spec=fw_spec, fw_to_check=run_fw)
    #         if handler.check():
    #             # TODO: add something whether we have a possible correction here or not ? has_correction() in handler ?
    #             if handler.has_correction():
    #                 corrections.append(handler.correct())
    #                 if handler.skip_remaining_handlers:
    #                     break
    #             else:
    #                 if handler.correction_is_required():
    #                     raise ValueError('No correction found for error while correction is required ...')
    #     return corrections

    def validate(self):
        validators = self.validators if self.validators is not None else []
        for validator in validators:
            if not validator.check():
                raise RuntimeError('Validator invalidate results ...')

    def apply_corrections(self, setup_fw, run_fw, corrections):
        spec = copy.deepcopy(run_fw.spec)
        modder = Modder()
        for correction in corrections:
            actions = correction['actions']
            for action in actions:
                if action['action_type'] == 'modify_object':
                    if action['object']['source'] == 'fw_spec':
                        myobject = spec[action['object']['key']]
                    else:
                        raise NotImplementedError('Object source "{}" not implemented in '
                                                  'CheckTask'.format(action['object']['source']))
                    newobj = modder.modify_object(action['action'], myobject)
                    spec[action['object']['key']] = newobj
                elif action['action_type'] == 'modify_dict':
                    if action['dict']['source'] == 'fw_spec':
                        mydict = spec[action['dict']['key']]
                    else:
                        raise NotImplementedError('Dict source "{}" not implemented in '
                                                  'CheckTask'.format(action['dict']['source']))
                    modder.modify(action['action'], mydict)
                else:
                    raise NotImplementedError('Action type "{}" not implemented in '
                                              'CheckTask'.format(action['action_type']))
        # Keep track of the corrections that have been applied
        if 'SRC_check_corrections' in spec:
            spec['SRC_check_corrections'].extend(corrections)
        else:
            spec['SRC_check_corrections'] = corrections

        # Update the task index
        task_index = SRCTaskIndex.from_any(spec['SRC_task_index'])
        task_index.increase_index()

        #TODO: in the future, see whether the FW queueadapter might be replaced by the qtk_queueadapter ?
        #      ... to be discussed with Anubhav, when the qtk queueadapter is in a qtk toolkit and not anymore
        #          in pymatgen/io/abinit
        #      in that case, this part will not be needed anymore ...
        spec['_queueadapter'] = spec['qtk_queueadapter'].get_subs_dict()
        #TODO: why do we need this again ??
        queue_adapter_update = get_queue_adapter_update(qtk_queueadapter=spec['qtk_queueadapter'],
                                                        corrections=corrections)

        run_task = run_fw.tasks[0]
        initialization_info = run_task.spec['initialization_info']
        deps = run_task.deps
        setup_task = setup_fw.tasks[0]

        check_task = self

        # Create the new Setup/Run/Check fireworks
        #TODO: do we need initialization info here ?
        #      do we need deps here ?
        SRC_fws = createSRCFireworks(setup_task=setup_task, run_task=run_task, check_task=check_task,
                                     spec=spec, initialization_info=initialization_info,
                                     task_index=task_index, deps=deps)
        wf = Workflow(fireworks=SRC_fws['fws'], links_dict=SRC_fws['links_dict'])
        return FWAction(detours=[wf])


def createSRCFireworks(setup_task, run_task, check_task, spec=None, initialization_info=None,
                       task_index=None, deps=None):
    spec = copy.deepcopy(spec)
    if task_index is not None:
        src_task_index = SRCTaskIndex.from_any(task_index)
    else:
        src_task_index = SRCTaskIndex.from_task(run_task)
    setup_spec = copy.deepcopy(spec)
    setup_spec['SRC_task_index'] = src_task_index
    pass


class SRCTaskIndex(object):
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
        self.index += 1

    def __str__(self):
        return '_'.join([self.task_type, str(self.index)])

    @property
    def setup_str(self):
        return '_'.join(['setup', self.__str__()])

    @property
    def run_str(self):
        return '_'.join(['run', self.__str__()])

    @property
    def check_str(self):
        return '_'.join(['check', self.__str__()])

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
            if sp[0] not in ['setup', 'run', 'check']:
                raise ValueError('SRC_task_index_string should start with "setup", "run" or "check" when 3 parts are '
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