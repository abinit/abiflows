# coding: utf-8
"""
Abstract base classes for controllers, monitors, and other possible tools/utils/objects used to manage, correct,
monitor and check tasks, events, results, objects, ...
"""

import abc
import copy
import logging

from monty.json import MSONable


logger = logging.getLogger(__name__)


PRIORITIES = {'PRIORITY_FIRST': 0,
              'PRIORITY_VERY_HIGH': 1,
              'PRIORITY_HIGH': 2,
              'PRIORITY_MEDIUM': 3,
              'PRIORITY_LOW': 4,
              'PRIORITY_VERY_LOW': 5,
              'PRIORITY_LAST': 6}


class Controller(MSONable):
    """
    Abstract base class for controlling a task, an event, a result, an output, an object, ...
    """
    PRIORITY_FIRST = PRIORITIES['PRIORITY_FIRST']
    PRIORITY_VERY_HIGH = PRIORITIES['PRIORITY_VERY_HIGH']
    PRIORITY_HIGH = PRIORITIES['PRIORITY_HIGH']
    PRIORITY_MEDIUM = PRIORITIES['PRIORITY_MEDIUM']
    PRIORITY_LOW = PRIORITIES['PRIORITY_LOW']
    PRIORITY_VERY_LOW = PRIORITIES['PRIORITY_VERY_LOW']
    PRIORITY_LAST = PRIORITIES['PRIORITY_LAST']

    _priority = PRIORITY_MEDIUM

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
        The function should return a ControlReport object containing the main conclusion of the controller, i.e.
         whether something important has been detected, as well as the possible actions/corrections to be done
         in order to continue/restart a task.
        """
        pass

    def set_priority(self, priority):
        if priority in PRIORITIES.keys():
            self._priority = PRIORITIES[priority]
        elif priority in PRIORITIES.values():
            self._priority = priority
        else:
            raise ValueError('"priority" in set_priority should be one of the following : '
                             '{}, {}'.format(', '.join([str(ii) for ii in PRIORITIES.keys()]),
                                             ', '.join([str(ii) for ii in PRIORITIES.values()])))

    @property
    def priority(self):
        return self._priority

    @property
    def skip_remaining_controllers(self):
        return False


class ControlReport(MSONable):

    STATES = {'FAILED_UNRECOVERABLE': 0,
              'FAILED_UNKNOWN_REASON': 1,
              'FAILED_RECOVERABLE': 2,
              'TO_BE_CONTINUED': 0,
              'FINALIZED': 10}

    FAILED_UNRECOVERABLE = STATES['FAILED_UNRECOVERABLE']
    FAILED_UNKNOWN_REASON = STATES['FAILED_UNKNOWN_REASON']
    FAILED_RECOVERABLE = STATES['FAILED_RECOVERABLE']
    TO_BE_CONTINUED = STATES['TO_BE_CONTINUED']
    FINALIZED = STATES['FINALIZED']

    def __init__(self, controller, state, problems=None, actions=None):
        self.controller = controller
        self.set_state(state)
        self.problems = problems
        self.actions = actions

    def set_state(self, state):
        if state in self.STATES.keys():
            self._state = self.STATES[state]
        elif state in self.STATES.values():
            self._state = state
        else:
            raise ValueError('"state" in set_state should be one of the following : '
                             '{}, {}'.format(', '.join([str(ii) for ii in self.STATES.keys()]),
                                             ', '.join([str(ii) for ii in self.STATES.values()])))

    @property
    def state(self):
        return self._state

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('"other" in __add__ method of ControlReport should be another ControlReport instance')
        state = min(self.state, other.state)
        actions = copy.deepcopy(self.actions)
        actions.extend(other.actions)
        return ControlReport(state=state, actions=actions)


#TODO: check if we need a different class ?
class Monitor(Controller):
    """
    Abstract base class for monitoring the execution of a given task
    """

