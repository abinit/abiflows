# coding: utf-8
"""
Task history related objects
"""
from __future__ import print_function, division, unicode_literals, absolute_import

import collections
import traceback
import logging

from monty.json import MontyDecoder, jsanitize, MSONable
from pymatgen.util.serialization import pmg_serialize


logger = logging.getLogger(__name__)


class TaskHistory(collections.deque, MSONable):
    """
    History class for tracking the creation and actions performed during a task.
    The objects provided should be PGMSONable, thus dicts, lists or PGMSONable objects.
    The expected items are dictionaries, transformations, corrections, autoparal, restart/reset
    and initializations (when factories will be handled).
    Possibly, the first item should contain information about the starting point of the task.
    This object will be forwarded during task restarts and resets, in order to keep track of the full history of the
    task.
    """

    @pmg_serialize
    def as_dict(self):
        items = [i.as_dict() if hasattr(i, "as_dict") else i for i in self]
        return dict(items=items)

    @classmethod
    def from_dict(cls, d):
        dec = MontyDecoder()
        return cls([dec.process_decoded(i) for i in d['items']])

    def log_initialization(self, task, initialization_info=None):
        details = {'task_class': task.__class__.__name__}
        if initialization_info:
            details['initialization_info'] = initialization_info
        self.append(TaskEvent(TaskEvent.INITIALIZED, details=details))

    def log_corrections(self, corrections):
        self.append(TaskEvent(TaskEvent.CORRECTIONS, corrections))

    def log_restart(self, restart_info, local_restart=False):
        self.append(TaskEvent(TaskEvent.RESTART, details=dict(restart_info=restart_info, local_restart=local_restart)))

    def log_autoparal(self, optconf):
        self.append(TaskEvent(TaskEvent.AUTOPARAL, details={'optconf': optconf}))

    def log_unconverged(self):
        self.append(TaskEvent(TaskEvent.UNCONVERGED))

    def log_finalized(self, final_input=None):
        details = dict(total_run_time=self.get_total_run_time())
        if final_input:
            details['final_input'] = final_input
        self.append(TaskEvent(TaskEvent.FINALIZED, details=details))

    def log_converge_params(self, unconverged_params, abiinput):
        params={}
        for param, new_value in unconverged_params.items():
            params[param] = dict(old_value=abiinput.get(param, 'Default'), new_value=new_value)
        self.append(TaskEvent(TaskEvent.UNCONVERGED_PARAMS, details={'params': params}))

    def log_error(self, exc):
        tb = traceback.format_exc()
        event_details = dict(stacktrace=tb)
        # If the exception is serializable, save its details
        try:
            exception_details = exc.to_dict()
        except AttributeError:
            exception_details = None
        except BaseException as e:
            logger.error("Exception couldn't be serialized: {} ".format(e))
            exception_details = None
        if exception_details:
            event_details['exception_details'] = exception_details
        self.append(TaskEvent(TaskEvent.ERROR, details=event_details))

    def log_abinit_stop(self, run_time=None):
        self.append(TaskEvent(TaskEvent.ABINIT_STOP, details={'run_time': run_time}))


    def get_events_by_types(self, types):
        """
        Return the events in history of the selected types. types can be a single type or a list
        """

        types = types if isinstance(types, (list, tuple)) else [types]

        events = [e for e in self if e.event_type in types]

        return events

    def get_total_run_time(self):
        """
        Calculates total run time based summing the run times saved in the abinit stop event.
        """

        total_run_time = 0
        for te in self.get_events_by_types(TaskEvent.ABINIT_STOP):
            run_time = te.details.get('run_time', None)
            if run_time:
                total_run_time += run_time

        return total_run_time


class TaskEvent(MSONable):
    """
    Object used to categorize the events in the TaskHistory.
    """

    INITIALIZED = 'initialized'
    CORRECTIONS = 'corrections'
    RESTART = 'restart'
    AUTOPARAL = 'autoparal'
    UNCONVERGED = 'unconverged'
    FINALIZED = 'finalized'
    UNCONVERGED_PARAMS = 'unconverged parameters'
    ERROR = 'error'
    ABINIT_STOP = 'abinit stop'

    def __init__(self, event_type, details=None):
        self.event_type = event_type
        self.details = details

    @pmg_serialize
    def as_dict(self):
        d = dict(event_type=self.event_type)
        if self.details:
            d['details'] = jsanitize(self.details, strict=True)

        return d

    @classmethod
    def from_dict(cls, d):
        dec = MontyDecoder()
        details = dec.process_decoded(d['details']) if 'details' in d else None
        return cls(event_type=d['event_type'], details=details)

