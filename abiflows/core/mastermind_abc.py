# coding: utf-8
"""
Abstract base classes for controllers, monitors, and other possible tools/utils/objects used to manage, correct,
monitor and check tasks, events, results, objects, ...
"""
from __future__ import print_function, division, unicode_literals, absolute_import

import abc
import fnmatch
import logging
import os
import shutil
import traceback

from six import add_metaclass
from monty.json import MontyDecoder
from monty.json import MSONable


logger = logging.getLogger(__name__)


PRIORITIES = {'PRIORITY_HIGHEST': 1000,
              'PRIORITY_VERY_HIGH': 875,
              'PRIORITY_HIGH': 750,
              'PRIORITY_MEDIUM_HIGH': 625,
              'PRIORITY_MEDIUM': 500,
              'PRIORITY_MEDIUM_LOW': 375,
              'PRIORITY_LOW': 250,
              'PRIORITY_VERY_LOW': 125,
              'PRIORITY_LOWEST': 0}

PRIORITY_HIGHEST = PRIORITIES['PRIORITY_HIGHEST']
PRIORITY_VERY_HIGH = PRIORITIES['PRIORITY_VERY_HIGH']
PRIORITY_HIGH = PRIORITIES['PRIORITY_HIGH']
PRIORITY_MEDIUM_HIGH = PRIORITIES['PRIORITY_MEDIUM_HIGH']
PRIORITY_MEDIUM = PRIORITIES['PRIORITY_MEDIUM']
PRIORITY_MEDIUM_LOW = PRIORITIES['PRIORITY_MEDIUM_LOW']
PRIORITY_LOW = PRIORITIES['PRIORITY_LOW']
PRIORITY_VERY_LOW = PRIORITIES['PRIORITY_VERY_LOW']
PRIORITY_LOWEST = PRIORITIES['PRIORITY_LOWEST']


#TODO: find a good name (is barrier ok ?). This is a container of controllers ...
# class ControlStep(MSONable):
# class ControlStage(MSONable):
# class ControlStation(MSONable):
# class ControlGate(MSONable):
# class ControlProcess(MSONable):
# class ControlBarrier(MSONable):
class ControlProcedure(MSONable):

    def __init__(self, controllers, monitors=None, sorting=None):
        self.controllers = []
        self.add_controllers(controllers=controllers)
        self.controlled_item_type = None

    def set_controlled_item_type(self, controlled_item_type):
        self.controlled_item_type = controlled_item_type

    def setup_controllers(self, controlled_item_type):
        self.grouped_controllers = {}
        controllers = [controller for controller in self.controllers
                       if controlled_item_type in controller.controlled_item_types]
        for controller in controllers:
            if controller.priority in self.grouped_controllers:
                self.grouped_controllers[controller.priority].append(controller)
            else:
                self.grouped_controllers[controller.priority] = [controller]
        self.priorities = sorted(self.grouped_controllers.keys(), reverse=True)
        self._ncontrollers = sum([len(v) for v in self.grouped_controllers.values()])

    def add_controller(self, controller):
        self.add_controllers(controllers=[controller])

    def add_controllers(self, controllers):
        if isinstance(controllers, (list, tuple)):
            for controller in controllers:
                if not issubclass(controller.__class__, Controller):
                    raise ValueError('One of the controllers is not a subclass of Controller')
            self.controllers.extend(controllers)
        elif issubclass(controllers.__class__, Controller):
            self.controllers.append(controllers)
        else:
            raise ValueError('controllers should be either a list of subclasses of Controller or a single '
                             'subclass of Controller')

    def process(self, **kwargs):
        self.setup_controllers(controlled_item_type=self.controlled_item_type)
        report = ControlReport()
        if self.ncontrollers == 0:
            report.state = ControlReport.UNRECOVERABLE
            return report
        for priority in self.priorities:
            skip_lower_priority = False
            #skip_other_controllers = False
            for controller in self.grouped_controllers[priority]:
                if controller._only_unfinalized and report.finalized:
                    #TODO: clean up here ... otherwise the ultimate could always be applied .....
                    continue
                controller_note = controller.process(**kwargs)
                # TODO: clean up this ...
                report.add_controller_note(controller_note=controller_note)
            #    if controller_note.state in ControllerNote.ERROR_STATES:
            #        skip_other_controllers = True
            #        break
                if report.state == ControlReport.RECOVERABLE and controller.skip_lower_priority_controllers:
                    skip_lower_priority = True
            if skip_lower_priority:
                break
            #if skip_other_controllers:
            #    break
        # if report.finalized and self.cleaner is not None:
        #     self.cleaner.clean()
        return report

    @property
    def ncontrollers(self):
        return self._ncontrollers

    @classmethod
    def from_dict(cls, d):
        dec = MontyDecoder()
        return cls(controllers=dec.process_decoded(d['controllers']))

    def as_dict(self):
        return {'@class': self.__class__.__name__,
                '@module': self.__class__.__module__,
                'controllers': [controller.as_dict() for controller in self.controllers]}


class ControlledItemType(MSONable):
    ALLOWED_CONTROL_ITEMS = {'TASK': ['READY', 'NOT_READY', 'VALID'],
                             'TASK_BEFORE_EXECUTION': ['READY', 'NOT_READY'],
                             'TASK_RUNNING': ['VALID', 'RECOVERABLE'],
                             'TASK_ABORTED': ['RECOVERABLE', 'UNRECOVERABLE'],
                             'TASK_FAILED': ['RECOVERABLE', 'UNRECOVERABLE', 'UNKNOWN_REASON'],
                             'TASK_COMPLETED': ['VALID', 'RECOVERABLE', 'UNRECOVERABLE'],
                             'FILE': ['MISSING', 'EMPTY', 'CORRUPTED', 'ERRONEOUS', 'VALID'],
                             'OBJECT': ['CORRUPTED', 'ERRONEOUS', 'VALID']}

    #     # Status of a controlled failed task
    # TASK_FAILED_UNRECOVERABLE_STATUS = 'TASK_FAILED_UNRECOVERABLE'
    # TASK_FAILED_UNKNOWN_REASON_STATUS = 'TASK_FAILED_UNKNOWN_REASON'
    # TASK_FAILED_RECOVERABLE_STATUS = 'TASK_FAILED_RECOVERABLE'
    #
    # # Status of a controlled stopped task
    # TASK_STOPPED_UNRECOVERABLE_STATUS = 'TASK_STOPPED_UNRECOVERABLE'
    # TASK_STOPPED_RECOVERABLE_STATUS = 'TASK_STOPPED_RECOVERABLE'
    #
    # # Status of a controlled completed task
    # TASK_COMPLETED_UNRECOVERABLE_STATUS = 'TASK_COMPLETED_UNRECOVERABLE'
    # TASK_COMPLETED_RECOVERABLE_STATUS = 'TASK_COMPLETED_RECOVERABLE'
    #
    # # Status of a controlled file
    # FILE_MISSING_STATUS = 'FILE_MISSING'
    # FILE_EMPTY_STATUS = 'FILE_EMPTY'
    # FILE_CORRUPTED_STATUS = 'FILE_CORRUPTED'
    # FILE_ERRONEOUS_STATUS = 'FILE_ERRONEOUS'
    # FILE_CORRECT_STATUS = 'FILE_CORRECT'
    #
    # # Status of a controlled object
    # OBJECT_CORRUPTED_STATUS = 'OBJECT_CORRUPTED'
    # OBJECT_ERRONEOUS_STATUS = 'OBJECT_ERRONEOUS'
    # OBJECT_CORRECT_STATUS = 'OBJECT_CORRECT'

    def __init__(self, item_type):
        self._set_item(item_type=item_type)

    def _set_item(self, item_type):
        if item_type not in self.ALLOWED_CONTROL_ITEMS:
            raise ValueError('"item_type" should be one of the following :'
                             ' {}'.format(', '.join(['"{}"' for ii in self.ALLOWED_CONTROL_ITEMS])))
        self._item_type = item_type

    @classmethod
    def task(cls):
        return cls(item_type='TASK')

    @classmethod
    def task_running(cls):
        return cls(item_type='TASK_RUNNING')

    @classmethod
    def task_aborted(cls):
        return cls(item_type='TASK_ABORTED')

    @classmethod
    def task_failed(cls):
        return cls(item_type='TASK_FAILED')

    @classmethod
    def task_completed(cls):
        return cls(item_type='TASK_COMPLETED')

    @classmethod
    def file(cls):
        return cls(item_type='FILE')

    @classmethod
    def object(cls):
        return cls(item_type='OBJECT')

    def __eq__(self, other):
        return self._item_type == other._item_type

    def __hash__(self):
        return self._item_type

    @classmethod
    def from_dict(cls, d):
        return cls(item_type=d['item_type'])

    def as_dict(self):
        return {'@class': self.__class__.__name__,
                '@module': self.__class__.__module__,
                'item_type': self._item_type}



@add_metaclass(abc.ABCMeta)
class Monitor:
    pass

@add_metaclass(abc.ABCMeta)
#class ControlStep(MSONable):
class Controller(MSONable):
    """
    Abstract base class for controlling a task, an event, a result, an output, an object, ...
    """

    _priority = PRIORITY_MEDIUM
    _controlled_item_types = None
    #TODO: try to make this cleaner ...
    _only_unfinalized = False

    # Types of controllers
    # Combinations of the following are possible, e.g. some controller might be a monitor, a handler, a manager and a
    #  validator at the same time

    # 2. Handler : Whether this controller is supposed to handle errors
    #              States that handlers can specify in their note : "NOTHING_FOUND", "ERROR_UNRECOVERABLE",
    #                                                               "ERROR_NOFIX", "ERROR_FIXSTOP", "ERROR_FIXCONTINUE"
    # is_handler = False

    # 4. Validator : Whether this controller is used to validate the results/tasks/...
    #                States that validators can specify in their note : "OK", "NOT_OK"
    # is_validator = False
    # NB: - The distinction between handler and manager is thin and is actually more up to the user. At this moment,
    #        they both are at the exact same level in the implementation ...

    can_validate = False

    def __init__(self):
        pass

    @abc.abstractmethod
    def from_dict(cls, d):
        pass

    @abc.abstractmethod
    def as_dict(self):
        pass

    @abc.abstractmethod
    def process(self, **kwargs):
        """
        Main function used to make the actual control/check of a list of inputs/outputs.
        The function should return a ControllerNote object containing the main conclusion of the controller, i.e.
        whether something important has been detected, as well as the possible actions/corrections to be done
        in order to carry on/continue/restart a task.
        """
        pass

    def set_priority(self, priority):
        if priority in PRIORITIES.keys():
            self._priority = PRIORITIES[priority]
        elif isinstance(priority, int) and 0 <= priority <= 1000:
            self._priority = priority
        else:
            raise ValueError('"priority" in set_priority should be either an integer between 0 and 1000 or '
                             'one of the following : {}'.format(', '.join([str(ii) for ii in PRIORITIES.keys()])))

    @property
    def priority(self):
        return self._priority

    @priority.setter
    def priority(self, priority):
        if priority in PRIORITIES.keys():
            self._priority = PRIORITIES[priority]
        elif isinstance(priority, int) and 0 <= priority <= 1000:
            self._priority = priority
        else:
            raise ValueError('"priority" should be either an integer between 0 and 1000 or '
                             'one of the following : {}'.format(', '.join([str(ii) for ii in PRIORITIES.keys()])))

    @property
    def controlled_item_types(self):
        return self._controlled_item_types

    @property
    def skip_remaining_controllers(self):
        return False

    @property
    def skip_lower_priority_controllers(self):
        return False

    @property
    def validated(self):
        return None


class Cleaner(MSONable):
    def __init__(self, dirs_and_patterns):
        """
        Initializes a Cleaner object used to delete files. The cleaner takes a simple list of dict described below
        :param dirs_and_patterns: list of dicts with keys "directory" and "patterns". for each item in the list, any
                                  file or directory present in the "directory" and matching one of the "patterns" will
                                  be deleted.
        Example : dirs_and_patterns = [{'directory': 'out',
                                        'patterns': ['*.log', '*.backup']},
                                       {'directory': 'tmp',
                                        'patterns': ['*']}]
                  will remove all files ending with ".log" or ".backup" in the "out" directory and all files in the
                  "tmp" directory.

        """
        for dir_and_patterns in dirs_and_patterns:
            if os.path.isabs(dir_and_patterns['directory']):
                raise ValueError('The definition of the directories to clean should be as relative paths and not '
                                 'absolute paths')
        self.dirs_and_patterns = dirs_and_patterns

    def clean(self, root_directory):
        if not os.path.isabs(root_directory):
            raise ValueError('The root directory to clean should be defined with an absolute path')
        deleted_files = []
        for dir_and_patterns in self.dirs_and_patterns:
            directory = os.path.join(root_directory, dir_and_patterns['directory'])
            if os.path.isdir(directory):
                for file in os.listdir(directory):
                    for patt in dir_and_patterns['patterns']:
                        if fnmatch.fnmatch(file, patt):
                            fp = os.path.join(directory, file)
                            try:
                                if os.path.isfile(fp):
                                    os.unlink(fp)
                                elif os.path.isdir(fp):
                                    shutil.rmtree(fp)
                                deleted_files.append(fp)
                                break
                            except:
                                logger.warning("Couldn't delete {}: {}".format(fp, traceback.format_exc()))
        pass

    # def delete_files(d, exts=None):
    #     deleted_files = []
    #     if os.path.isdir(d):
    #         for f in os.listdir(d):
    #             if exts is None or "*" in exts or any(ext in f for ext in exts):
    #                 fp = os.path.join(d, f)
    #                 try:
    #                     if os.path.isfile(fp):
    #                         os.unlink(fp)
    #                     elif os.path.isdir(fp):
    #                         shutil.rmtree(fp)
    #                     deleted_files.append(fp)
    #                 except:
    #                     logger.warning("Couldn't delete {}: {}".format(fp, traceback.format_exc()))
    #
    #     return deleted_files


@add_metaclass(abc.ABCMeta)
class Manager(MSONable):
    def __init__(self):
        pass

    @abc.abstractmethod
    def manage(self):
        pass

# class ControllerStatement(MSONable):
# class ControllerAccount(MSONable):
# class ControllerRecord(MSONable):
class ControllerNote(MSONable):

    # Special state returned by a controller that is supposed to be "more important" than others and can say whether
    #  a given task is completely finalized and does not need further restarts/changes/... In that case, it is very
    #  clear that the task is completed.
    EVERYTHING_OK = 'EVERYTHING_OK'
    # State of a controlled task specifying that nothing was detected by the controller
    NOTHING_FOUND = 'NOTHING_FOUND'
    # State of a controlled task specifying that some error(s) was (were) detected by the controller and this (these)
    #  error(s) is unrecoverable. In that case, it is very clear that nothing can be done.
    ERROR_UNRECOVERABLE = 'ERROR_UNRECOVERABLE'
    # State of a controlled task specifying that some error(s) was (were) detected by the controller but the controller
    #  could not fix that error. In that case, it is possible that some other controller could fix the situation
    ERROR_NOFIX = 'ERROR_NOFIX'
    # State of a controlled task specifying that some error(s) was (were) detected by the controller and the controller
    #  adviced some action to fix the error. No other controller should be applied.
    ERROR_RECOVERABLE = 'ERROR_RECOVERABLE'
    # # State of a controlled task specifying that some error(s) was (were) detected by the controller and the controller
    # #  adviced some action to fix the error. Other controllers (if any) might still be applied.
    # ERROR_FIXCONTINUE = 'ERROR_FIXCONTINUE'

    STATES = [EVERYTHING_OK, NOTHING_FOUND, ERROR_UNRECOVERABLE, ERROR_NOFIX, ERROR_RECOVERABLE]
    ERROR_STATES = [ERROR_UNRECOVERABLE, ERROR_NOFIX, ERROR_RECOVERABLE]

    #TODO consider using increasing integers as values, so that we can take the lowest as a general value of the
    # restart
    # Restart from the very beginning of the calculation. Every intermediate result should be neglected
    RESTART_FROM_SCRATCH = "RESTART_FROM_SCRATCH"

    # Restart from the current step but don't make use previous informations
    RESET_RESTART = "RESET_RESTART"

    # Restart with the maximum amount of information available
    SIMPLE_RESTART = "SIMPLE_RESTART"

    def __init__(self, controller, state=None, problems=None, actions=None, restart=None, is_valid=None):
        self.controller = controller
        self.state = state
        self.problems = problems
        self.actions = actions
        self.restart = restart
        self.is_valid = is_valid

    @property
    def actions(self):
        return self._actions

    @actions.setter
    def actions(self, actions):
        if actions is None:
            actions = {}
        self._actions = actions

    def add_problem(self, problem):
        if self.problems is None:
            self.problems = [problem]
        else:
            self.problems.append(problem)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        if state is None:
            self._state = None
        elif state in self.STATES:
            self._state = state
        else:
            raise ValueError('"state" in ControllerNote should be one of the following : '
                             '{}'.format(', '.join(self.STATES)))

    @property
    def has_errors_recoverable(self):
        return True

    @property
    def has_errors_unrecoverable(self):
        return True

    @property
    def is_recoverable(self):
        return True

    @property
    def has_errors(self):
        return True

    @property
    def is_valid(self):
        if not self.controller.can_validate:
            raise ValueError('This ControllerNote comes from a controller that cannot validate !')
        return self._is_valid

    @is_valid.setter
    def is_valid(self, is_valid):
        self._is_valid = is_valid

    def set_restart(self, restart):
        self.restart = restart

    def simple_restart(self):
        self.restart = self.SIMPLE_RESTART

    def reset_restart(self):
        self.restart = self.RESET_RESTART

    def restart_from_scratch(self):
        self.restart = self.RESTART_FROM_SCRATCH

    @classmethod
    def from_dict(cls, d):
        dec = MontyDecoder()
        if 'is_valid' in d:
            return cls(controller=dec.process_decoded(d['controller']),
                       state=d['state'], problems=d['problems'],
                       actions=dec.process_decoded(d['actions']),
                       restart=d['restart'], is_valid=d['is_valid'])
        else:
            return cls(controller=dec.process_decoded(d['controller']),
                       state=d['state'], problems=d['problems'],
                       actions=dec.process_decoded(d['actions']),
                       restart=d['restart'])

    def as_dict(self):
        dd = {'@class': self.__class__.__name__,
              '@module': self.__class__.__module__,
              'controller': self.controller.as_dict(),
              'state': self.state,
              'problems': self.problems,
              'actions': {key: action.as_dict() for key, action in self.actions.items()},
              'restart': self.restart}
        if self.controller.can_validate:
            dd['is_valid'] = self.is_valid
        return dd


class ControlReport(MSONable):

    # Status of an SRC trio that is completely finalized
    FINALIZED = 'FINALIZED'
    # Status of an SRC trio that has one or more errors that is (are) unrecoverable
    UNRECOVERABLE = 'UNRECOVERABLE'
    # Status of an SRC trio that has one or more errors that is (are) recoverable
    RECOVERABLE = 'RECOVERABLE'
    # Status of an SRC trio that is ongoing (for controllers taking care of non-error-related stuff, e.g. convergence,
    #  accuracy goal achieved with multiple-steps such as relaxation with low ecut then high ecut, ...)
    ONGOING = 'ONGOING'
    NONE = 'NONE'

    STATES = [FINALIZED, UNRECOVERABLE, RECOVERABLE, ONGOING, NONE]

    def __init__(self, controller_notes=None):
        self.controller_notes = []
        if controller_notes is not None:
            self.add_controller_notes(controller_notes)
        self.update_state_from_controller_notes()

    def add_controller_notes(self, controller_notes):
        self.controller_notes.extend(controller_notes)
        self.update_state_from_controller_notes()

    def add_controller_note(self, controller_note):
        self.controller_notes.append(controller_note)
        self.update_state_from_controller_notes()

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        if state not in self.STATES:
            raise ValueError('"state" in ControlReport is "{}" should be one of the following : '
                             '{}'.format(str(state), ', '.join(self.STATES)))
        self._state = state

    def update_state_from_controller_notes(self):
        if len(self.controller_notes) == 0:
            self.state = self.NONE
        elif any([cn.state == ControllerNote.ERROR_RECOVERABLE for cn in self.controller_notes]):
            self.state = self.RECOVERABLE
        elif any([cn.state == ControllerNote.ERROR_UNRECOVERABLE for cn in self.controller_notes]):
            self.state = self.UNRECOVERABLE
        elif (len([cn for cn in self.controller_notes if cn.controller.can_validate]) and
                  all([cn.is_valid for cn in self.controller_notes if cn.controller.can_validate])):
            self.state = self.FINALIZED
        else:
            # If no controller says it is recoverable/unrecoverable/valid/... then we set it to unrecoverable ?
            self.state = self.UNRECOVERABLE

    @property
    def finalized(self):
        return self.state == self.FINALIZED

    @property
    def unrecoverable(self):
        return self.state == self.UNRECOVERABLE

    @property
    def actions(self):
        # TODO: should check whether actions are compatible here ... right now, two different controllers are not
        #       allowed to modify the same object ...
        actions = {}
        for cn in self.controller_notes:
            for target, action in cn.actions.items():
                if target in actions:
                    raise ValueError('Two controllers give actions to the same object!')
                actions[target] = action
        return actions

    @property
    def restart_info(self):
        if any([cn.restart == ControllerNote.RESTART_FROM_SCRATCH for cn in self.controller_notes]):
            return ControllerNote.RESTART_FROM_SCRATCH
        if any([cn.restart == ControllerNote.RESET_RESTART for cn in self.controller_notes]):
            return ControllerNote.RESET_RESTART
        return ControllerNote.SIMPLE_RESTART

    @classmethod
    def from_dict(cls, d):
        return cls(controller_notes=[ControllerNote.from_dict(cndict) for cndict in d['controller_notes']])

    def as_dict(self):
        return {'@class': self.__class__.__name__,
                '@module': self.__class__.__module__,
                'controller_notes': [cn.as_dict() for cn in self.controller_notes]}


#TODO: should this be MSONable ? Is that even possible with a callable object in self ?
#class Instruction(MSONable):
#class Directive(MSONable):
class Action(MSONable):

    def __init__(self, callable, **kwargs):
        self.callable = callable
        self.kwargs = kwargs

    def apply(self, object):
        if len(self.kwargs) == 0:
            self.callable(object)
        else:
            self.callable(object, **self.kwargs)

    @classmethod
    def from_dict(cls, d):
        import importlib
        mod = importlib.import_module(d['callable']['module'])
        class_ = getattr(mod, d['callable']['class'])
        callable = getattr(class_, d['callable']['func_name'])
        return cls(callable=callable)

    def as_dict(self):
        #FIXME this part is not compatible with python3 and unbound methods do not exist anymore
        return {'@class': self.__class__.__name__,
                '@module': self.__class__.__module__,
                'callable': {'module': self.callable.im_class.__module__,
                             'class': self.callable.im_class.__name__,
                             'func_name': self.callable.im_func.func_name}
                }

    @classmethod
    def from_string(cls, callable_string, **kwargs):
        #TODO: do this ?
        import importlib
        mod_name, func_name = callable_string.rsplit('.',1)
        mod = importlib.import_module(mod_name)
        callable = getattr(mod, func_name)
        cls(callable=callable, **kwargs)