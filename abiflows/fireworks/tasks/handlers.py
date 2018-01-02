# coding: utf-8
"""
Error handlers and validators
"""
from __future__ import print_function, division, unicode_literals, absolute_import

import logging
import os

from abiflows.fireworks.utils.custodian_utils import SRCErrorHandler
from pymatgen.io.abinit.scheduler_error_parsers import MemoryCancelError
from pymatgen.io.abinit.scheduler_error_parsers import MasterProcessMemoryCancelError
from pymatgen.io.abinit.scheduler_error_parsers import SlaveProcessMemoryCancelError
from pymatgen.io.abinit.scheduler_error_parsers import TimeCancelError
from pymatgen.io.abinit.qadapters import QueueAdapter


logger = logging.getLogger(__name__)


class AbinitHandler(SRCErrorHandler):
    """
    General handler for abinit's critical events handlers.
    """

    def __init__(self, job_rundir='.', critical_events=None, queue_adapter=None):
        """
        Initializes the handler with the directory where the job was run.

        Args:
            job_rundir: Directory where the job was run.
        """
        super(AbinitHandler, self).__init__()
        self.job_rundir = job_rundir
        self.critical_events = critical_events

        self.src_fw = False

    def as_dict(self):
        return {'@class': self.__class__.__name__,
                '@module': self.__class__.__module__,
                'job_rundir': self.job_rundir
                }

    @classmethod
    def from_dict(cls, d):
        return cls(job_rundir=d['job_rundir'])

    @property
    def allow_fizzled(self):
        return False

    @property
    def allow_completed(self):
        return True

    @property
    def handler_priority(self):
        return self.PRIORITY_MEDIUM

    @property
    def skip_remaining_handlers(self):
        return True

    def setup(self):
        if 'SRCScheme' in self.fw_to_check.spec and self.fw_to_check.spec['SRCScheme']:
            self.src_fw = True
        else:
            self.src_fw = False
        self.job_rundir = self.fw_to_check.launches[-1].launch_dir

    def check(self):
        abinit_task = self.fw_to_check.tasks[0]
        self.report = None
        try:
            self.report = abinit_task.get_event_report()
        except Exception as exc:
            msg = "%s exception while parsing event_report:\n%s" % (self, exc)
            logger.critical(msg)

        if self.report is not None:
            # Run has completed, check for critical events (convergence, ...)
            if self.report.run_completed:
                self.events = self.report.filter_types(abinit_task.CRITICAL_EVENTS)
                if self.events:
                    return True
                else:
                    # Calculation has converged
                    # Check if there are custom parameters that should be converged
                    unconverged_params, reset_restart = abinit_task.check_parameters_convergence(self.fw_to_check.spec)
                    if unconverged_params:
                        return True
                    else:
                        return False
            # Abinit run failed to complete
            # Check if the errors can be handled
            if self.report.errors:
                return True
        return True

    def has_corrections(self):
        return True

    def correct(self):
        if self.src_fw:
            if len(self.fw_to_check.tasks) > 1:
                raise ValueError('More than 1 task found in fizzled firework, not yet supported')
            abinit_input_update = {'iscf': 2}
            return {'errors': [self.__class__.__name__],
                    'actions': [{'action_type': 'modify_object',
                                 'object': {'source': 'fw_spec', 'key': 'abinit_input'},
                                 'action': {'_set': abinit_input_update}}]}
        else:
            raise NotImplementedError('This handler cannot be used without the SRC scheme')


class WalltimeHandler(SRCErrorHandler):
    """
    Handler for walltime infringements of the resource manager.
    """

    def __init__(self, job_rundir='.', qout_file='queue.qout', qerr_file='queue.qerr', queue_adapter=None,
                 max_timelimit=None, timelimit_increase=None):
        """
        Initializes the handler with the directory where the job was run, the standard output and error files
        of the queue manager and the queue adapter used.

        Args:
            job_rundir: Directory where the job was run.
            qout_file: Standard output file of the queue manager.
            qerr_file: Standard error file of the queue manager.
            queue_adapter: Queue adapter used to submit the job.
            max_timelimit: Maximum timelimit (in seconds) allowed by the resource manager for the queue.
            timelimit_increase: Amount of time (in seconds) to increase the timelimit.
        """
        super(WalltimeHandler, self).__init__()
        self.job_rundir = job_rundir
        self.qout_file = qout_file
        self.qerr_file = qerr_file
        self.queue_adapter = queue_adapter
        self.setup_filepaths()
        self.max_timelimit = max_timelimit
        self.timelimit_increase = timelimit_increase

        self.src_fw = False

    def setup_filepaths(self):
        self.qout_filepath = os.path.join(self.job_rundir, self.qout_file)
        self.qerr_filepath = os.path.join(self.job_rundir, self.qerr_file)

    def as_dict(self):
        return {'@class': self.__class__.__name__,
                '@module': self.__class__.__module__,
                'job_rundir': self.job_rundir,
                'qout_file': self.qout_file,
                'qerr_file': self.qerr_file,
                'queue_adapter': self.queue_adapter.as_dict() if self.queue_adapter is not None else None,
                'max_timelimit': self.max_timelimit,
                'timelimit_increase': self.timelimit_increase
                }

    @classmethod
    def from_dict(cls, d):
        qa = QueueAdapter.from_dict(d['queue_adapter']) if d['queue_adapter'] is not None else None
        return cls(job_rundir=d['job_rundir'], qout_file=d['qout_file'], qerr_file=d['qerr_file'], queue_adapter=qa,
                   max_timelimit=d['max_timelimit'],
                   timelimit_increase=d['timelimit_increase'])

    @property
    def allow_fizzled(self):
        return True

    @property
    def allow_completed(self):
        return False

    @property
    def handler_priority(self):
        return self.PRIORITY_VERY_LOW

    @property
    def skip_remaining_handlers(self):
        return True

    def setup(self):
        if 'SRCScheme' in self.fw_to_check.spec and self.fw_to_check.spec['SRCScheme']:
            self.src_fw = True
        else:
            self.src_fw = False
        self.job_rundir = self.fw_to_check.launches[-1].launch_dir
        self.setup_filepaths()
        self.queue_adapter = self.fw_to_check.spec['qtk_queueadapter']

    def check(self):

        # Analyze the stderr and stdout files of the resource manager system.
        qerr_info = None
        qout_info = None
        if os.path.exists(self.qerr_filepath):
            with open(self.qerr_filepath, "r") as f:
                qerr_info = f.read()
        if os.path.exists(self.qout_filepath):
            with open(self.qout_filepath, "r") as f:
                qout_info = f.read()

        self.timelimit_error = None
        self.queue_errors = None
        if qerr_info or qout_info:
            from pymatgen.io.abinit.scheduler_error_parsers import get_parser
            qtype = self.queue_adapter.QTYPE
            scheduler_parser = get_parser(qtype, err_file=self.qerr_filepath,
                                          out_file=self.qout_filepath)

            if scheduler_parser is None:
                raise ValueError('Cannot find scheduler_parser for qtype {}'.format(qtype))

            scheduler_parser.parse()
            self.queue_errors = scheduler_parser.errors

            for error in self.queue_errors:
                if isinstance(error, TimeCancelError):
                    logger.debug('found timelimit error.')
                    self.timelimit_error = error
                    return True
        return False

    def correct(self):
        if self.src_fw:
            if len(self.fw_to_check.tasks) > 1:
                raise ValueError('More than 1 task found in "memory-fizzled" firework, not yet supported')
            logger.debug('adding SRC detour')
            # Information about the update of the memory (master overhead or base mem per proc) in the queue adapter
            queue_adapter_update = {}
            # When max_timelimit is not set, automatically take the hard timelimit of the queue
            if self.max_timelimit is None:
                max_timelimit = self.queue_adapter.timelimit_hard
            else:
                max_timelimit = self.max_timelimit
            # When timelimit_increase is not set, automatically take a tenth of the hard timelimit of the queue
            if self.timelimit_increase is None:
                timelimit_increase = self.queue_adapter.timelimit_hard / 10
            else:
                timelimit_increase = self.timelimit_increase
            if isinstance(self.timelimit_error, TimeCancelError):
                old_timelimit = self.queue_adapter.timelimit
                if old_timelimit == max_timelimit:
                    raise ValueError('Cannot increase beyond maximum timelimit ({:d} seconds) set in WalltimeHandler.'
                                     'Hard time limit of '
                                     'the queue is {:d} seconds'.format(max_timelimit,
                                                                        self.queue_adapter.timelimit_hard))
                new_timelimit = old_timelimit + timelimit_increase
                # If the new timelimit exceeds the max timelimit, just put it to the max timelimit
                if new_timelimit > max_timelimit:
                    new_timelimit = max_timelimit
                queue_adapter_update['timelimit'] = new_timelimit
            else:
                raise ValueError('Should not be here ...')
            return {'errors': [self.__class__.__name__],
                    'actions': [{'action_type': 'modify_object',
                                 'object': {'source': 'fw_spec', 'key': 'qtk_queueadapter'},
                                 'action': {'_set': queue_adapter_update}}]}
        else:
            raise NotImplementedError('This handler cannot be used without the SRC scheme')

    def has_corrections(self):
        return True


class MemoryHandler(SRCErrorHandler):
    """
    Handler for memory infringements of the resource manager. The handler should be able to handle the possible
    overhead of the master process.
    """

    def __init__(self, job_rundir='.', qout_file='queue.qout', qerr_file='queue.qerr', queue_adapter=None,
                 max_mem_per_proc_mb=8000, mem_per_proc_increase_mb=1000,
                 max_master_mem_overhead_mb=8000, master_mem_overhead_increase_mb=1000):
        """
        Initializes the handler with the directory where the job was run, the standard output and error files
        of the queue manager and the queue adapter used.

        Args:
            job_rundir: Directory where the job was run.
            qout_file: Standard output file of the queue manager.
            qerr_file: Standard error file of the queue manager.
            queue_adapter: Queue adapter used to submit the job.
            max_mem_per_proc_mb: Maximum memory per process in megabytes.
            mem_per_proc_increase_mb: Amount of memory to increase the memory per process in megabytes.
            max_master_mem_overhead_mb: Maximum overhead memory for the master process in megabytes.
            master_mem_overhead_increase_mb: Amount of memory to increase the overhead memory for the master process
                                             in megabytes.
        """
        super(MemoryHandler, self).__init__()
        self.job_rundir = job_rundir
        self.qout_file = qout_file
        self.qerr_file = qerr_file
        self.queue_adapter = queue_adapter
        self.setup_filepaths()
        self.max_mem_per_proc_mb = max_mem_per_proc_mb
        self.mem_per_proc_increase_mb = mem_per_proc_increase_mb
        self.max_master_mem_overhead_mb = max_master_mem_overhead_mb
        self.master_mem_overhead_increase_mb = master_mem_overhead_increase_mb

        self.src_fw = False

    def setup_filepaths(self):
        self.qout_filepath = os.path.join(self.job_rundir, self.qout_file)
        self.qerr_filepath = os.path.join(self.job_rundir, self.qerr_file)

    def as_dict(self):
        return {'@class': self.__class__.__name__,
                '@module': self.__class__.__module__,
                'job_rundir': self.job_rundir,
                'qout_file': self.qout_file,
                'qerr_file': self.qerr_file,
                'queue_adapter': self.queue_adapter.as_dict() if self.queue_adapter is not None else None,
                'max_mem_per_proc_mb': self.max_mem_per_proc_mb,
                'mem_per_proc_increase_mb': self.mem_per_proc_increase_mb,
                'max_master_mem_overhead_mb': self.max_master_mem_overhead_mb,
                'master_mem_overhead_increase_mb': self.master_mem_overhead_increase_mb
                }

    @classmethod
    def from_dict(cls, d):
        qa = QueueAdapter.from_dict(d['queue_adapter']) if d['queue_adapter'] is not None else None
        return cls(job_rundir=d['job_rundir'], qout_file=d['qout_file'], qerr_file=d['qerr_file'], queue_adapter=qa,
                   max_mem_per_proc_mb=d['max_mem_per_proc_mb'],
                   mem_per_proc_increase_mb=d['mem_per_proc_increase_mb'],
                   max_master_mem_overhead_mb=d['max_master_mem_overhead_mb'],
                   master_mem_overhead_increase_mb=d['master_mem_overhead_increase_mb'])

    @property
    def allow_fizzled(self):
        return True

    @property
    def allow_completed(self):
        return False

    @property
    def handler_priority(self):
        return self.PRIORITY_VERY_LOW

    @property
    def skip_remaining_handlers(self):
        return True

    def setup(self):
        if 'SRCScheme' in self.fw_to_check.spec and self.fw_to_check.spec['SRCScheme']:
            self.src_fw = True
        else:
            self.src_fw = False
        self.job_rundir = self.fw_to_check.launches[-1].launch_dir
        self.setup_filepaths()
        self.queue_adapter = self.fw_to_check.spec['qtk_queueadapter']

    def check(self):

        # Analyze the stderr and stdout files of the resource manager system.
        qerr_info = None
        qout_info = None
        if os.path.exists(self.qerr_filepath):
            with open(self.qerr_filepath, "r") as f:
                qerr_info = f.read()
        if os.path.exists(self.qout_filepath):
            with open(self.qout_filepath, "r") as f:
                qout_info = f.read()

        self.memory_error = None
        self.queue_errors = None
        if qerr_info or qout_info:
            from pymatgen.io.abinit.scheduler_error_parsers import get_parser
            qtype = self.queue_adapter.QTYPE
            scheduler_parser = get_parser(qtype, err_file=self.qerr_filepath,
                                          out_file=self.qout_filepath)

            if scheduler_parser is None:
                raise ValueError('Cannot find scheduler_parser for qtype {}'.format(qtype))

            scheduler_parser.parse()
            self.queue_errors = scheduler_parser.errors

            #TODO: handle the cases where it is Master or Slave here ... ?
            for error in self.queue_errors:
                if isinstance(error, MemoryCancelError):
                    logger.debug('found memory error.')
                    self.memory_error = error
                    return True
                if isinstance(error, MasterProcessMemoryCancelError):
                    logger.debug('found master memory error.')
                    self.memory_error = error
                    return True
                if isinstance(error, SlaveProcessMemoryCancelError):
                    logger.debug('found slave memory error.')
                    self.memory_error = error
                    return True
        return False

    def correct(self):
        if self.src_fw:
            if len(self.fw_to_check.tasks) > 1:
                raise ValueError('More than 1 task found in "memory-fizzled" firework, not yet supported')
            logger.debug('adding SRC detour')
            # Information about the update of the memory (master overhead or base mem per proc) in the queue adapter
            queue_adapter_update = {}
            if isinstance(self.memory_error, (MemoryCancelError, SlaveProcessMemoryCancelError)):
                old_mem_per_proc = self.queue_adapter.mem_per_proc
                new_mem_per_proc = old_mem_per_proc + self.mem_per_proc_increase_mb
                queue_adapter_update['mem_per_proc'] = new_mem_per_proc
            elif isinstance(self.memory_error, MasterProcessMemoryCancelError):
                old_mem_overhead = self.queue_adapter.master_mem_overhead
                new_mem_overhead = old_mem_overhead + self.master_mem_overhead_increase_mb
                if new_mem_overhead > self.max_master_mem_overhead_mb:
                    raise ValueError('New master memory overhead {:d} is larger than '
                                     'max master memory overhead {:d}'.format(new_mem_overhead,
                                                                              self.max_master_mem_overhead_mb))
                queue_adapter_update['master_mem_overhead'] = new_mem_overhead
            else:
                raise ValueError('Should not be here ...')
            return {'errors': [self.__class__.__name__],
                    'actions': [{'action_type': 'modify_object',
                                 'object': {'source': 'fw_spec', 'key': 'qtk_queueadapter'},
                                 'action': {'_set': queue_adapter_update}}]}
        else:
            raise NotImplementedError('This handler cannot be used without the SRC scheme')

    def has_corrections(self):
        return True


class UltimateMemoryHandler(MemoryHandler):
    """
    Handler for infringements of the resource manager. If no memory error is found,
    """

    def __init__(self, job_rundir='.', qout_file='queue.qout', qerr_file='queue.qerr', queue_adapter=None,
                 max_mem_per_proc_mb=8000, mem_per_proc_increase_mb=1000,
                 max_master_mem_overhead_mb=8000, master_mem_overhead_increase_mb=1000):
        """
        Initializes the handler with the directory where the job was run, the standard output and error files
        of the queue manager and the queue adapter used.

        Args:
            job_rundir: Directory where the job was run.
            qout_file: Standard output file of the queue manager.
            qerr_file: Standard error file of the queue manager.
            queue_adapter: Queue adapter used to submit the job.
            max_mem_per_proc_mb: Maximum memory per process in megabytes.
            mem_per_proc_increase_mb: Amount of memory to increase the memory per process in megabytes.
            max_master_mem_overhead_mb: Maximum overhead memory for the master process in megabytes.
            master_mem_overhead_increase_mb: Amount of memory to increase the overhead memory for the master process
                                             in megabytes.
        """
        super(UltimateMemoryHandler, self).__init__()

    @property
    def handler_priority(self):
        return self.PRIORITY_LAST

    def check(self):
        mem_check = super(UltimateMemoryHandler, self).check()
        if mem_check:
            raise ValueError('This error should have been caught by a standard MemoryHandler ...')
        #TODO: Do we have some check that we can do here ?
        return True

    def correct(self):
        if self.src_fw:
            if len(self.fw_to_check.tasks) > 1:
                raise ValueError('More than 1 task found in "memory-fizzled" firework, not yet supported')
            if self.memory_error is not None:
                raise ValueError('This error should have been caught by a standard MemoryHandler ...')
            if self.queue_errors is not None and len(self.queue_errors) > 0:
                raise ValueError('Queue errors were found ... these should be handled properly by another handler')
            # Information about the update of the memory (base mem per proc) in the queue adapter
            queue_adapter_update = {}
            old_mem_per_proc = self.queue_adapter.mem_per_proc
            new_mem_per_proc = old_mem_per_proc + self.mem_per_proc_increase_mb
            queue_adapter_update['mem_per_proc'] = new_mem_per_proc
            return {'errors': [self.__class__.__name__],
                    'actions': [{'action_type': 'modify_object',
                                 'object': {'source': 'fw_spec', 'key': 'qtk_queueadapter'},
                                 'action': {'_set': queue_adapter_update}}]}
        else:
            raise NotImplementedError('This handler cannot be used without the SRC scheme')