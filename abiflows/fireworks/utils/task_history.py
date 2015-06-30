# coding: utf-8
"""
Task history related objects
"""
from __future__ import print_function, division, unicode_literals

from monty.json import MontyDecoder
from pymatgen.serializers.json_coders import PMGSONable, pmg_serialize
import collections
import traceback

class TaskHistory(collections.deque, PMGSONable):
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
        self.append(TaskEvent('initialized', details=details))

    def log_corrections(self, corrections):
        self.append(TaskEvent('corrections', corrections))

    def log_restart(self, restart_info, local_restart=False):
        self.append(TaskEvent('restart', details=dict(restart_info=restart_info, local_restart=local_restart)))

    def log_autoparal(self, optconf):
        self.append(TaskEvent('autoparal', details={'optconf': optconf}))

    def log_unconverged(self):
        self.append(TaskEvent('unconverged'))

    def log_concluded(self):
        self.append(TaskEvent('concluded'))

    def log_finalized(self, final_input=None):
        self.append(TaskEvent('finalized', details={'final_input': final_input} if final_input else  None))

    def log_converge_params(self, unconverged_params, abiinput):
        params={}
        for param, new_value in unconverged_params.items():
            params[param] = dict(old_value=abiinput.get(param, 'Default'), new_value=new_value)
        self.append(TaskEvent('unconverged parameters', details={'params': params}))

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
        self.append(TaskEvent('error', details=event_details))


class TaskEvent(PMGSONable):

    def __init__(self, event_type, details=None):
        self.event_type = event_type
        self.details = details

    @pmg_serialize
    def as_dict(self):
        d = dict(event_type=self.event_type)
        if self.details:
            if hasattr(self.details, "as_dict"):
                d['details'] = self.details.as_dict()
            elif isinstance(self.details, (list, tuple)):
                d['details'] = [i.as_dict() if hasattr(i, "as_dict") else i for i in self.details]
            elif isinstance(self.details, dict):
                d['details'] = {k: v.as_dict() if hasattr(v, "as_dict") else v for k, v in self.details.items()}
            else:
                d['details'] = self.details
        return d

    @classmethod
    def from_dict(cls, d):
        dec = MontyDecoder()
        details = dec.process_decoded(d['details']) if 'details' in d else None
        return cls(event_type=d['event_type'], details=details)

