# coding: utf-8
"""
Error handlers and validators
"""

from abiflows.fireworks.utils.custodian_utils import SRCErrorHandler, SRCValidator
from pymatgen.io.abinit.scheduler_error_parsers import MemoryCancelError
from pymatgen.io.abinit.scheduler_error_parsers import MasterProcessMemoryCancelError
from pymatgen.io.abinit.scheduler_error_parsers import SlaveProcessMemoryCancelError
import logging
import os


logger = logging.getLogger(__name__)


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
        self.qout_filepath = os.path.join(job_rundir, qout_file)
        self.qerr_filepath = os.path.join(job_rundir, qerr_file)
        self.max_mem_per_proc_mb = max_mem_per_proc_mb
        self.mem_per_proc_increase_mb = mem_per_proc_increase_mb
        self.max_master_mem_overhead_mb = max_master_mem_overhead_mb
        self.master_mem_overhead_increase_mb = master_mem_overhead_increase_mb

        self.src_fw = False

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
        self.queue_adapter = self.fw_to_check.spec['qtk_queueadapter']

    @property
    def src_scheme(self):
        return self.src_fw

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


# class UltimateQueueAdapterHandler(SRCErrorHandler):
#     """
#     Handler for infringements of the resource manager. This will just restart
#     """
#
#     def __init__(self, job_rundir='.', qout_file='queue.qout', qerr_file='queue.qerr', queue_adapter=None,
#                  max_mem_per_proc_mb=8000, mem_per_proc_increase_mb=1000,
#                  max_master_mem_overhead_mb=8000, master_mem_overhead_increase_mb=1000):
#         """
#         Initializes the handler with the directory where the job was run, the standard output and error files
#         of the queue manager and the queue adapter used.
#
#         Args:
#             job_rundir: Directory where the job was run.
#             qout_file: Standard output file of the queue manager.
#             qerr_file: Standard error file of the queue manager.
#             queue_adapter: Queue adapter used to submit the job.
#             max_mem_per_proc_mb: Maximum memory per process in megabytes.
#             mem_per_proc_increase_mb: Amount of memory to increase the memory per process in megabytes.
#             max_master_mem_overhead_mb: Maximum overhead memory for the master process in megabytes.
#             master_mem_overhead_increase_mb: Amount of memory to increase the overhead memory for the master process
#                                              in megabytes.
#         """
#         super(UltimateQueueAdapterHandler).__init__()
#         self.job_rundir = job_rundir
#         self.qout_file = qout_file
#         self.qerr_file = qerr_file
#         self.queue_adapter = queue_adapter
#         self.qout_filepath = os.path.join(job_rundir, qout_file)
#         self.qerr_filepath = os.path.join(job_rundir, qerr_file)
#         self.max_mem_per_proc_mb = max_mem_per_proc_mb
#         self.mem_per_proc_increase_mb = mem_per_proc_increase_mb
#         self.max_master_mem_overhead_mb = max_master_mem_overhead_mb
#         self.master_mem_overhead_increase_mb = master_mem_overhead_increase_mb
#
#         self.src_fw = False
#
#     @property
#     def allow_fizzled(self):
#         return True
#
#     @property
#     def handler_priority(self):
#         return self.PRIORITY_LAST
#
#     def setup(self):
#         if 'SRCScheme' in self.fw_to_check.spec and self.fw_to_check.spec['SRCScheme']:
#             self.src_fw = True
#         else:
#             self.src_fw = False
#         self.job_rundir = self.fw_to_check.launches[-1].launch_dir
#         self.queue_adapter = self.fw_to_check.spec['qtk_queueadapter']
#
#     @property
#     def src_scheme(self):
#         return self.src_fw
#
#     def check(self):
#
#         # Analyze the stderr and stdout files of the resource manager system.
#         qerr_info = None
#         qout_info = None
#         if os.path.exists(self.qerr_filepath):
#             with open(self.qerr_filepath, "r") as f:
#                 qerr_info = f.read()
#         if os.path.exists(self.qout_filepath):
#             with open(self.qout_filepath, "r") as f:
#                 qout_info = f.read()
#
#         self.memory_error = None
#         if qerr_info or qout_info:
#             from pymatgen.io.abinit.scheduler_error_parsers import get_parser
#             qtype = self.queue_adapter.QTYPE
#             scheduler_parser = get_parser(qtype, err_file=self.qerr_filepath,
#                                           out_file=self.qout_filepath)
#
#             if scheduler_parser is None:
#                 raise ValueError('Cannot find scheduler_parser for qtype {}'.format(qtype))
#
#             scheduler_parser.parse()
#             self.queue_errors = scheduler_parser.errors
#
#             #TODO: handle the cases where it is Master or Slave here ... ?
#             for error in self.queue_errors:
#                 if isinstance(error, MemoryCancelError):
#                     logger.debug('found memory error.')
#                     self.memory_error = error
#                     return True
#                 if isinstance(error, MasterProcessMemoryCancelError):
#                     logger.debug('found master memory error.')
#                     self.memory_error = error
#                     return True
#                 if isinstance(error, SlaveProcessMemoryCancelError):
#                     logger.debug('found slave memory error.')
#                     self.memory_error = error
#                     return True
#         return False
#
#     def correct(self):
#         if self.src_fw:
#             if len(self.fw_to_check.tasks) > 1:
#                 raise ValueError('More than 1 task found in "memory-fizzled" firework, not yet supported')
#             logger.debug('adding SRC detour')
#             # Information about the update of the memory (master overhead or base mem per proc) in the queue adapter
#             queue_adapter_update = {}
#             if isinstance(self.memory_error, (MemoryCancelError, SlaveProcessMemoryCancelError)):
#                 old_mem_per_proc = self.queue_adapter.mem_per_proc
#                 new_mem_per_proc = old_mem_per_proc + self.mem_per_proc_increase_mb
#                 queue_adapter_update['mem_per_proc'] = new_mem_per_proc
#             elif isinstance(self.memory_error, MasterProcessMemoryCancelError):
#                 old_mem_overhead = self.queue_adapter.master_mem_overhead
#                 new_mem_overhead = old_mem_overhead + self.master_mem_overhead_increase_mb
#                 if new_mem_overhead > self.max_master_mem_overhead_mb:
#                     raise ValueError('New master memory overhead {:d} is larger than '
#                                      'max master memory overhead {:d}'.format(new_mem_overhead,
#                                                                               self.max_master_mem_overhead_mb))
#                 queue_adapter_update['master_mem_overhead'] = new_mem_overhead
#             else:
#                 raise ValueError('Should not be here ...')
#             return {'errors': [self.__class__.__name__],
#                     'dict': 'queue_adapter',
#                     'action': {'_set': queue_adapter_update}}
#         else:
#             raise NotImplementedError('This handler cannot be used without the SRC scheme')