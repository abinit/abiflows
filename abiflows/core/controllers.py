# coding: utf-8
"""
Controllers
"""
from __future__ import print_function, division, unicode_literals, absolute_import

import copy
import logging
import os

from monty.json import MontyDecoder
from pymatgen.io.abinit import events
from pymatgen.io.abinit.scheduler_error_parsers import MemoryCancelError
from pymatgen.io.abinit.scheduler_error_parsers import MasterProcessMemoryCancelError
from pymatgen.io.abinit.scheduler_error_parsers import SlaveProcessMemoryCancelError
from pymatgen.io.abinit.scheduler_error_parsers import TimeCancelError
from pymatgen.io.abinit.utils import Directory, File
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.analysis.transition_state import NEBAnalysis
from pymatgen.core.structure import Structure
from abipy.abio.inputs import AbinitInput


from abiflows.core.mastermind_abc import Action
from abiflows.core.mastermind_abc import Controller
from abiflows.core.mastermind_abc import ControllerNote
from abiflows.core.mastermind_abc import ControlledItemType
from abiflows.core.mastermind_abc import PRIORITY_HIGH
from abiflows.core.mastermind_abc import PRIORITY_VERY_LOW
from abiflows.core.mastermind_abc import PRIORITY_LOWEST


logger = logging.getLogger(__name__)

class AbinitController(Controller):
    """
    General handler for abinit's events.
    Determines whether the calculation ended correctly or not and fixes errors (including unconverged) if Abinit
    error handlers are available.
    """

    is_handler = True
    is_validator = True
    can_validate = True
    #_controlled_item_types = [ControlledItemType.task_completed(), ControlledItemType.task_failed()]
    _controlled_item_types = [ControlledItemType.task_completed()]

    def __init__(self, critical_events=None, handlers=None):
        """
        Initializes the controller with the critical events that trigger the restart and a list of ErrorHandlers

        Args:
            critical_events: List of events that trigger a restart due to unconverged calculation
            handlers: List of ErrorHandlers (pymatgen.io.abinit.events) used to handle specific events
        """
        super(AbinitController, self).__init__()

        critical_events = [] if critical_events is None else critical_events
        handlers = [] if handlers is None else handlers

        self.critical_events = critical_events if isinstance(critical_events, (list, tuple)) else [critical_events]
        self.handlers = handlers if isinstance(handlers, (list, tuple)) else [handlers]

        self.set_priority(PRIORITY_HIGH)

    def process(self, **kwargs):
        """
        Returns the ControllerNote
        """
        for kw in ['abinit_input', 'abinit_output_filepath', 'abinit_log_filepath', 'abinit_mpi_abort_filepath',
                   'abinit_outdir_path']:
            if kw not in kwargs:
                raise ValueError("kwarg {} is required to process abinit results".format(kw))
        queue_adapter = copy.deepcopy(kwargs.get('queue_adapter', None))
        abinit_input = copy.deepcopy(kwargs.get('abinit_input'))
        abinit_output_file = File(kwargs.get('abinit_output_filepath'))
        abinit_log_file = File(kwargs.get('abinit_log_filepath'))
        abinit_mpi_abort_file = File(kwargs.get('abinit_mpi_abort_filepath'))
        abinit_outdir_path = Directory(kwargs.get('abinit_outdir_path'))
        structure = kwargs.get('structure', None)

        note = ControllerNote(controller=self)
        # Initialize the actions for everything that is passed to kwargs
        actions = {}

        report = None
        try:
            report = self.get_event_report(abinit_log_file, abinit_mpi_abort_file)
        except Exception as exc:
            msg = "%s exception while parsing event_report:\n%s" % (self, exc)
            logger.critical(msg)

        # If the calculation is ok, parse the outputs
        if report is not None:
            # the calculation finished without errors
            if report.run_completed:
                # Check if the calculation converged.
                critical_events_found = report.filter_types(self.critical_events)
                if critical_events_found:
                    # self.history.log_unconverged()
                    # hook
                    # local_restart, restart_fw, stored_data = self.prepare_restart(fw_spec)
                    # num_restarts = self.restart_info.num_restarts if self.restart_info else 0
                    # if num_restarts < self.ftm.fw_policy.max_restarts:
                    #     if local_restart:
                    #         return None
                    #     else:
                    #         stored_data['final_state'] = 'Unconverged'
                    #         return FWAction(detours=restart_fw, stored_data=stored_data)
                    # else:
                    #     raise UnconvergedError(self, msg="Unconverged after {} restarts".format(num_restarts),
                    #                            abiinput=self.abiinput, restart_info=self.restart_info,
                    #                            history=self.history)
                    # Calculation did not converge. A simple restart is enough
                    note.state = ControllerNote.ERROR_RECOVERABLE
                    note.simple_restart()
                    if 'RelaxConvergenceWarning' in [e.name for e in critical_events_found]:
                        if structure is None:
                            raise ValueError('Structure should be present in the initial objects for restarting '
                                             'relaxation runs')
                        # Fake action ... TODO: make something better for the actions ...
                        actions['structure'] = Action(callable=Structure.__len__)
                    note.add_problem('Unconverged: {}'.format(', '.join(e.name for e in critical_events_found)))
                else:
                    # calculation converged
                    #TODO move to a different controler
                    # check if there are custom parameters that should be converged
                    # unconverged_params, reset_restart = self.check_parameters_convergence(fw_spec)
                    # if unconverged_params:
                    #     self.history.log_converge_params(unconverged_params, self.abiinput)
                    #     self.abiinput.set_vars(**unconverged_params)
                    #     local_restart, restart_fw, stored_data = self.prepare_restart(fw_spec, reset=reset_restart)
                    #     num_restarts = self.restart_info.num_restarts if self.restart_info else 0
                    #     if num_restarts < self.ftm.fw_policy.max_restarts:
                    #         if local_restart:
                    #             return None
                    #         else:
                    #             stored_data['final_state'] = 'Unconverged_parameters'
                    #             return FWAction(detours=restart_fw, stored_data=stored_data)
                    #     else:
                    #         raise UnconvergedParametersError(self, abiinput=self.abiinput,
                    #                                          restart_info=self.restart_info, history=self.history)
                    # else:
                    #     # everything is ok. conclude the task
                    #     # hook
                    #     update_spec, mod_spec, stored_data = self.conclude_task(fw_spec)
                    #     return FWAction(stored_data=stored_data, update_spec=update_spec, mod_spec=mod_spec)
                    note.state = ControllerNote.EVERYTHING_OK
                    note.is_valid = True
            elif report.errors:
            # Abinit reported problems
            # Check if the errors could be handled
                logger.debug('Found errors in report')
                # for error in report.errors:
                #     logger.debug(str(error))
                #     try:
                #         self.abi_errors.append(error)
                #     except AttributeError:
                #         self.abi_errors = [error]

                # ABINIT errors, try to handle them
                fixed, reset, abiinput_actions = self.fix_abicritical(report=report, abiinput=abinit_input,
                                                                      queue_adapter=queue_adapter,
                                                                      outdir=abinit_outdir_path)

                if fixed:
                    note.state = ControllerNote.ERROR_RECOVERABLE
                    if reset:
                        note.reset_restart()
                    else:
                        note.simple_restart()

                    actions['abinit_input'] = abiinput_actions

                    #TODO if the queue_adapter can be modified by the handlers return it
                    # actions['queue_adapter'] = queue_adapter_actions
                else:
                    msg = "Critical events couldn't be fixed by handlers."
                    logger.info(msg)
                    note.state = ControllerNote.ERROR_UNRECOVERABLE

                for err in report.errors:
                    note.add_problem(err)

            else:
            # Calculation not completed but no errors. No fix could be applied in this controller
                note.state = ControllerNote.ERROR_UNRECOVERABLE
                note.add_problem('Abinit calculation not completed but no errors in report.')

        else:
        # report does not exist. No fix could be applied in this controller
            note.state = ControllerNote.NOTHING_FOUND
            note.add_problem('No Abinit report')

        # No errors from abinit. No fix could be applied at this stage.
        # The FW will be fizzled.
        # Try to save the stderr file for Fortran runtime errors.
        #TODO check if some cases could be handled here
        # err_msg = None
        # if self.stderr_file.exists:
        #     #TODO length should always be enough, but maybe it's worth cutting the message if it's too long
        #     err_msg = self.stderr_file.read()
        #     # It happened that the text file contained non utf-8 characters.
        #     # sanitize the text to avoid problems during database inserption
        #     err_msg.decode("utf-8", "ignore")
        # logger.error("return code {}".format(self.returncode))
        # raise AbinitRuntimeError(self, err_msg)

        note.actions = actions
        return note

    @classmethod
    def from_dict(cls, d):
        dec = MontyDecoder()
        import importlib
        critical_events = []
        for ced in d['critical_events']:
            mod = importlib.import_module(ced['module'])
            ce = getattr(mod, ced['name'])
            critical_events.append(ce)
        return cls(critical_events=critical_events,
                   # critical_events=dec.process_decoded(d['critical_events']),
                   handlers=dec.process_decoded(d['handlers']))

    def as_dict(self):
        critical_events = [{'module': ce.__module__, 'name': ce.__name__} for ce in self.critical_events]
        return {'@class': self.__class__.__name__, '@module': self.__class__.__module__,
                # 'critical_events': [ce.as_dict() for ce in self.critical_events],
                'critical_events': critical_events,
                'handlers': [er.as_dict() for er in self.handlers]
                }

    def get_event_report(self, ofile, mpiabort_file):
        """
        Analyzes the main output file for possible Errors or Warnings.

        Returns:
            :class:`EventReport` instance or None if the main output file does not exist.
        """

        parser = events.EventsParser()

        if not ofile.exists:
            if not mpiabort_file.exists:
                return None
            else:
                # ABINIT abort file without log!
                abort_report = parser.parse(mpiabort_file.path)
                return abort_report

        try:
            report = parser.parse(ofile.path)

            # Add events found in the ABI_MPIABORTFILE.
            if mpiabort_file.exists:
                logger.critical("Found ABI_MPIABORTFILE!")
                abort_report = parser.parse(mpiabort_file.path)
                if len(abort_report) == 0:
                    logger.warning("ABI_MPIABORTFILE but empty")
                else:
                    if len(abort_report) != 1:
                        logger.critical("Found more than one event in ABI_MPIABORTFILE")

                    # Add it to the initial report only if it differs
                    # from the last one found in the main log file.
                    last_abort_event = abort_report[-1]
                    if report and last_abort_event != report[-1]:
                        report.append(last_abort_event)
                    else:
                        report.append(last_abort_event)

            return report

        #except parser.Error as exc:
        except Exception as exc:
            # Return a report with an error entry with info on the exception.
            logger.critical("{}: Exception while parsing ABINIT events:\n {}".format(ofile, str(exc)))
            return parser.report_exception(ofile.path, exc)

    def fix_abicritical(self, report, abiinput, outdir, queue_adapter=None):
        """
        method to fix crashes/error caused by abinit

        Returns:
            retcode: 1 if task has been fixed else 0.
            reset: True if at least one of the corrections applied requires a reset
        """
        if not self.handlers:
            logger.info('Empty list of event handlers. Cannot fix abi_critical errors')
            return 0, None, []

        done = len(self.handlers) * [0]
        corrections = []

        for event in report:
            for i, handler in enumerate(self.handlers):
                if handler.can_handle(event) and not done[i]:
                    logger.info("handler", handler, "will try to fix", event)
                    try:
                        #TODO pass the queueadapter to the handlers? the output should be modified in that case
                        c = handler.handle_input_event(abiinput, outdir, event)
                        if c:
                            done[i] += 1
                            corrections.append(c)

                    except Exception as exc:
                        logger.critical(str(exc))

        if corrections:
            reset = any(c.reset for c in corrections)
            # self.history.log_corrections(corrections)
            # convert the actions applied on the input to Actions
            actions = []
            for c in corrections:
                # remove vars as a first action, in case incopatible variables have been set.
                if '_pop' in c.actions:
                    actions.append(Action(AbinitInput.remove_vars(c.actions['_pop'])))
                if '_set' in c.actions:
                    actions.append(Action(AbinitInput.set_vars(c.actions['_set'])))
                if '_update' in c.actions:
                    actions.append(Action(AbinitInput.set_vars(c.actions['_update'])))
                if '_change_structure' in c.actions:
                    actions.append(Action(AbinitInput.set_structure(c.actions['_change_structure'])))

            return 1, reset, actions

        logger.info('We encountered AbiCritical events that could not be fixed')
        return 0, None, []

    @classmethod
    def from_helper(cls, helper):
        return cls(critical_events=helper.CRITICAL_EVENTS, handlers=None)


class QueueControllerMixin(object):

    def get_queue_errors(self, **kwargs):
        # Get the file paths for the stderr and stdout of the resource manager system, as well as the queue_adapter
        qerr_filepath = kwargs.get('qerr_filepath', None)
        qout_filepath = kwargs.get('qout_filepath', None)
        queue_adapter = kwargs.get('queue_adapter', None)
        memory_policy = kwargs.get('memory_policy', None)
        #TODO: deal with the memory policy in the scheduler parser (whether it is vmem or mem in PBS for example ...)
        if 'queue_adapter' is None:
            raise ValueError('WalltimeController should have access to the queue_adapter')
        if 'qerr_filepath' is None:
            raise ValueError('WalltimeController should have access to the qerr_filepath')
        if 'qout_filepath' is None:
            raise ValueError('WalltimeController should have access to the qout_filepath')
        # Analyze the stderr and stdout files of the resource manager system.
        qerr_info = None
        qout_info = None
        if qerr_filepath is not None and os.path.exists(qerr_filepath):
            with open(qerr_filepath, "r") as f:
                qerr_info = f.read()
        if qout_filepath is not None and os.path.exists(qout_filepath):
            with open(qout_filepath, "r") as f:
                qout_info = f.read()
        if qerr_info or qout_info:
            from pymatgen.io.abinit.scheduler_error_parsers import get_parser
            qtype = queue_adapter.QTYPE
            scheduler_parser = get_parser(qtype, err_file=qerr_filepath,
                                          out_file=qout_filepath)

            if scheduler_parser is None:
                raise ValueError('Cannot find scheduler_parser for qtype {}'.format(qtype))

            scheduler_parser.parse()
            queue_errors = scheduler_parser.errors
            return queue_errors
        else:
            return None


class WalltimeController(Controller, QueueControllerMixin):
    """
    Controller for walltime infringements of the resource manager.
    """

    is_handler = True
    _controlled_item_types = [ControlledItemType.task_failed()]

    def __init__(self, max_timelimit=None, timelimit_increase=None):
        """
        Initializes the handler with the directory where the job was run, the standard output and error files
        of the queue manager and the queue adapter used.

        Args:
            max_timelimit: Maximum timelimit (in seconds).
            timelimit_increase: Amount of time (in seconds) to increase the timelimit.
        """
        super(WalltimeController, self).__init__()
        self.max_timelimit = max_timelimit
        self.timelimit_increase = timelimit_increase
        self.priority = PRIORITY_VERY_LOW

    def as_dict(self):
        return {'@class': self.__class__.__name__,
                '@module': self.__class__.__module__,
                'max_timelimit': self.max_timelimit,
                'timelimit_increase': self.timelimit_increase
                }

    @classmethod
    def from_dict(cls, d):
        return cls(max_timelimit=d['max_timelimit'],
                   timelimit_increase=d['timelimit_increase'])

    @property
    def skip_remaining_handlers(self):
        return True

    @property
    def skip_lower_priority_controllers(self):
        return True

    def process(self, **kwargs):
        # Get the Queue Adapter and the queue errors
        queue_adapter = kwargs.get('queue_adapter', None)
        if queue_adapter is None:
            raise ValueError('No queue adapter passed to the WalltimeController')
        queue_errors = self.get_queue_errors(**kwargs)

        # Create the Controller Note and the actions
        note = ControllerNote(controller=self)
        actions = {}

        # No errors found
        if not queue_errors:
            note.state = ControllerNote.NOTHING_FOUND
            return note

        # Get the timelimit error if there is one
        timelimit_error = None
        for error in queue_errors:
            if isinstance(error, TimeCancelError):
                logger.debug('found timelimit error.')
                timelimit_error = error

        # No timelimit error found
        if timelimit_error is None:
            note.state = ControllerNote.NOTHING_FOUND
            return note

        # Setup the new timelimit
        note.add_problem('Task has been stopped due to timelimit')
        if self.max_timelimit is None:
            max_timelimit = queue_adapter.timelimit_hard
        else:
            max_timelimit = self.max_timelimit
        # When timelimit_increase is not set, automatically take a tenth of the hard timelimit of the queue
        if self.timelimit_increase is None:
            timelimit_increase = queue_adapter.timelimit_hard / 10
        else:
            timelimit_increase = self.timelimit_increase
        old_timelimit = queue_adapter.timelimit
        if old_timelimit == max_timelimit:
            note.state = ControllerNote.ERROR_UNRECOVERABLE
            note.add_problem('Maximum timelimit has been reached, cannot increase further')
            return note
        new_timelimit = old_timelimit + timelimit_increase
        # If the new timelimit exceeds the max timelimit, just put it to the max timelimit
        if new_timelimit > max_timelimit:
            new_timelimit = max_timelimit

        # Set the actions to be performed, the state and the type of restart
        actions['queue_adapter'] = Action(callable=queue_adapter.__class__.set_timelimit,
                                          timelimit=new_timelimit)
        note.state = ControllerNote.ERROR_RECOVERABLE
        note.actions = actions
        note.reset_restart()
        return note


class MemoryController(Controller, QueueControllerMixin):
    """
    Controller for memory infringements of the resource manager. The handler should be able to handle the possible
    overhead of the master process.
    """

    is_handler = True
    #_controlled_item_types = [ControlledItemType.task_failed()]
    _controlled_item_types = [ControlledItemType.task_completed(), ControlledItemType.task_failed()]

#
#     def __init__(self, job_rundir='.', qout_file='queue.qout', qerr_file='queue.qerr', queue_adapter=None,
#                  max_mem_per_proc_mb=8000, mem_per_proc_increase_mb=1000,
#                  max_master_mem_overhead_mb=8000, master_mem_overhead_increase_mb=1000):

    def __init__(self, max_mem_per_proc_mb=8000, mem_per_proc_increase_mb=1000,
                       max_master_mem_overhead_mb=8000, master_mem_overhead_increase_mb=1000,
                       memory_policy='physical_memory'):
        """
        Initializes the handler with the directory where the job was run, the standard output and error files
        of the queue manager and the queue adapter used.

        Args:
            max_mem_per_proc_mb: Maximum memory per process in megabytes.
            mem_per_proc_increase_mb: Amount of memory to increase the memory per process in megabytes.
            max_master_mem_overhead_mb: Maximum overhead memory for the master process in megabytes.
            master_mem_overhead_increase_mb: Amount of memory to increase the overhead memory for the master process
                                             in megabytes.
            memory_policy: Policy for the memory (some weird clusters sometimes use the virtual memory to stop jobs
                           that overcome some virtual memory limit)
        """
        super(MemoryController, self).__init__()
        self.max_mem_per_proc_mb = max_mem_per_proc_mb
        self.mem_per_proc_increase_mb = mem_per_proc_increase_mb
        self.max_master_mem_overhead_mb = max_master_mem_overhead_mb
        self.master_mem_overhead_increase_mb = master_mem_overhead_increase_mb
        self.memory_policy = memory_policy
        self.priority = PRIORITY_VERY_LOW

    @property
    def memory_policy(self):
        return self._memory_policy

    @memory_policy.setter
    def memory_policy(self, memory_policy):
        if memory_policy not in ['physical_memory', 'virtual_memory']:
            raise ValueError('Memory policy is "{}" in MemoryController while itshould be either "physical_memory"'
                             'or "virtual_memory"'.format(memory_policy))
        self._memory_policy = memory_policy

    def as_dict(self):
        return {'@class': self.__class__.__name__,
                '@module': self.__class__.__module__,
                'max_mem_per_proc_mb': self.max_mem_per_proc_mb,
                'mem_per_proc_increase_mb': self.mem_per_proc_increase_mb,
                'max_master_mem_overhead_mb': self.max_master_mem_overhead_mb,
                'master_mem_overhead_increase_mb': self.master_mem_overhead_increase_mb,
                'memory_policy': self.memory_policy
                }

    @classmethod
    def from_dict(cls, d):
        return cls(max_mem_per_proc_mb=d['max_mem_per_proc_mb'],
                   mem_per_proc_increase_mb=d['mem_per_proc_increase_mb'],
                   max_master_mem_overhead_mb=d['max_master_mem_overhead_mb'],
                   master_mem_overhead_increase_mb=d['master_mem_overhead_increase_mb'],
                   memory_policy=d['memory_policy'])

    @property
    def skip_remaining_handlers(self):
        return True

    @property
    def skip_lower_priority_controllers(self):
        return True

    def process(self, **kwargs):
        # Get the Queue Adapter and the queue errors
        queue_adapter = kwargs.get('queue_adapter', None)
        if queue_adapter is None:
            raise ValueError('No queue adapter passed to the MemoryController')
        queue_errors = self.get_queue_errors(memory_policy=self.memory_policy, **kwargs)

        # Create the Controller Note and the actions
        note = ControllerNote(controller=self)
        actions = {}

        # No errors found
        if not queue_errors:
            note.state = ControllerNote.NOTHING_FOUND
            return note

        # Get the memory error if there is one
        memory_error = None
        master_memory_error = None
        slave_memory_error = None
        for error in queue_errors:
            if isinstance(error, MemoryCancelError):
                logger.debug('found memory error.')
                memory_error = error
            elif isinstance(error, MasterProcessMemoryCancelError):
                logger.debug('found master memory error.')
                master_memory_error = error
            elif isinstance(error, SlaveProcessMemoryCancelError):
                logger.debug('found slave memory error.')
                slave_memory_error = error

        # No memory error found
        if all(err is None for err in [memory_error, master_memory_error, slave_memory_error]):
            note.state = ControllerNote.NOTHING_FOUND
            return note

        # TODO: allow the possibility to have multiple actions here ? If both the master and the slave gets the error ?
        if memory_error or slave_memory_error:
            note.add_problem('Task has been stopped due to memory infringement'
                             '{}'.format('' if memory_error else ' by a slave process'))
            old_mem_per_proc = queue_adapter.mem_per_proc
            if old_mem_per_proc == self.max_mem_per_proc_mb:
                note.state = ControllerNote.ERROR_UNRECOVERABLE
                note.add_problem('Maximum mem_per_proc has been reached, cannot increase further')
                return note
            new_mem_per_proc = old_mem_per_proc + self.mem_per_proc_increase_mb
            if new_mem_per_proc > self.max_mem_per_proc_mb:
                new_mem_per_proc = self.max_mem_per_proc_mb
            actions['queue_adapter'] = Action(callable=queue_adapter.__class__.set_mem_per_proc,
                                              mem_mb=new_mem_per_proc)
        if master_memory_error:
            note.add_problem('Task has been stopped due to memory infringement by the master process')
            old_mem_overhead = self.queue_adapter.master_mem_overhead
            if old_mem_overhead == self.max_master_mem_overhead_mb:
                note.state = ControllerNote.ERROR_UNRECOVERABLE
                note.add_problem('Maximum master_mem_overhead has been reached, cannot increase further')
                return note
            new_mem_overhead = old_mem_overhead + self.master_mem_overhead_increase_mb
            if new_mem_overhead > self.max_master_mem_overhead_mb:
                new_mem_overhead = self.max_master_mem_overhead_mb
            actions['queue_adapter'] = Action(callable=queue_adapter.__class__.set_master_mem_overhead,
                                              mem_mb=new_mem_overhead)

        # Set the actions to be performed, the state and the type of restart
        note.state = ControllerNote.ERROR_RECOVERABLE
        note.actions = actions
        note.reset_restart()
        return note


class AbinitZenobeSlaveMemoryController(Controller, QueueControllerMixin):
    """
    Controller for memory infringements of the resource manager. The handler should be able to handle the possible
    overhead of the master process.
    """

    is_handler = True
    #_controlled_item_types = [ControlledItemType.task_failed()]
    _controlled_item_types = [ControlledItemType.task_completed(), ControlledItemType.task_failed()]

#
#     def __init__(self, job_rundir='.', qout_file='queue.qout', qerr_file='queue.qerr', queue_adapter=None,
#                  max_mem_per_proc_mb=8000, mem_per_proc_increase_mb=1000,
#                  max_master_mem_overhead_mb=8000, master_mem_overhead_increase_mb=1000):

    def __init__(self, max_mem_per_proc_mb=8000, mem_per_proc_increase_mb=1000,
                       memory_policy='physical_memory'):
        """
        Initializes the handler with the directory where the job was run, the standard output and error files
        of the queue manager and the queue adapter used.

        Args:
            max_mem_per_proc_mb: Maximum memory per process in megabytes.
            mem_per_proc_increase_mb: Amount of memory to increase the memory per process in megabytes.
            memory_policy: Policy for the memory (some weird clusters sometimes use the virtual memory to stop jobs
                           that overcome some virtual memory limit)
        """
        super(AbinitZenobeSlaveMemoryController, self).__init__()
        self.max_mem_per_proc_mb = max_mem_per_proc_mb
        self.mem_per_proc_increase_mb = mem_per_proc_increase_mb
        self.memory_policy = memory_policy
        self.priority = PRIORITY_VERY_LOW

    @property
    def memory_policy(self):
        return self._memory_policy

    @memory_policy.setter
    def memory_policy(self, memory_policy):
        if memory_policy not in ['physical_memory', 'virtual_memory']:
            raise ValueError('Memory policy is "{}" in MemoryController while it should be either "physical_memory"'
                             'or "virtual_memory"'.format(memory_policy))
        self._memory_policy = memory_policy

    def as_dict(self):
        return {'@class': self.__class__.__name__,
                '@module': self.__class__.__module__,
                'max_mem_per_proc_mb': self.max_mem_per_proc_mb,
                'mem_per_proc_increase_mb': self.mem_per_proc_increase_mb,
                'memory_policy': self.memory_policy
                }

    @classmethod
    def from_dict(cls, d):
        return cls(max_mem_per_proc_mb=d['max_mem_per_proc_mb'],
                   mem_per_proc_increase_mb=d['mem_per_proc_increase_mb'],
                   memory_policy=d['memory_policy'])

    @property
    def skip_remaining_handlers(self):
        return True

    @property
    def skip_lower_priority_controllers(self):
        return True

    def process(self, **kwargs):
        # Get the Queue Adapter and the queue errors
        queue_adapter = kwargs.get('queue_adapter', None)
        if queue_adapter is None:
            raise ValueError('No queue adapter passed to the MemoryController')
        run_err_filepath = kwargs.get('abinit_err_filepath', None)
        qerr_filepath = kwargs.get('qerr_filepath', None)
        qout_filepath = kwargs.get('qout_filepath', None)

        found = False
        # Try to find "dapl_conn_rc" in run.err
        if run_err_filepath:
            f = open(run_err_filepath, 'r')
            run_err = f.readlines()
            f.close()

            for line in run_err:
                if 'dapl_conn_rc' in line:
                    found = True
                    break

        # Try to find "dapl_conn_rc" in queue.qerr
        if not found:
            if run_err_filepath:
                f = open(qerr_filepath, 'r')
                qerr = f.readlines()
                f.close()

                for line in qerr:
                    if 'dapl_conn_rc' in line:
                        found = True
                        break

        # Try to find "dapl_conn_rc" in queue.qout
        if not found:
            if run_err_filepath:
                f = open(qout_filepath, 'r')
                qout = f.readlines()
                f.close()

                for line in qout:
                    if 'dapl_conn_rc' in line:
                        found = True
                        break

        # Create the Controller Note and the actions
        note = ControllerNote(controller=self)
        actions = {}

        if not found:
            note.state = ControllerNote.NOTHING_FOUND
            return note

        note.add_problem('Task has been stopped due to memory infringement by a slave process '
                         '(dapl_conn_rc error on Zenobe ...)')
        old_mem_per_proc = queue_adapter.mem_per_proc
        if old_mem_per_proc == self.max_mem_per_proc_mb:
            note.state = ControllerNote.ERROR_UNRECOVERABLE
            note.add_problem('Maximum mem_per_proc has been reached, cannot increase further')
            return note
        new_mem_per_proc = old_mem_per_proc + self.mem_per_proc_increase_mb
        if new_mem_per_proc > self.max_mem_per_proc_mb:
            new_mem_per_proc = self.max_mem_per_proc_mb
        actions['queue_adapter'] = Action(callable=queue_adapter.__class__.set_mem_per_proc,
                                          mem_mb=new_mem_per_proc)

        # Set the actions to be performed, the state and the type of restart
        note.state = ControllerNote.ERROR_RECOVERABLE
        note.actions = actions
        note.reset_restart()
        return note


    # def process(self, **kwargs):
    #     # Create the Controller Note
    #     note = ControllerNote(controller=self)
    #     # Get the file paths for the stderr and stdout of the resource manager system, as well as the queue_adapter
    #     qerr_filepath = kwargs.get('qerr_filepath', None)
    #     qout_filepath = kwargs.get('qout_filepath', None)
    #     queue_adapter = kwargs.get('queue_adapter', None)
    #     if 'queue_adapter' is None:
    #         raise ValueError('WalltimeController should have access to the queue_adapter')
    #     if 'qerr_filepath' is None:
    #         raise ValueError('WalltimeController should have access to the qerr_filepath')
    #     if 'qout_filepath' is None:
    #         raise ValueError('WalltimeController should have access to the qout_filepath')
    #     # Initialize the actions for everything that is passed to kwargs
    #     actions = {}
    #     # Analyze the stderr and stdout files of the resource manager system.
    #     qerr_info = None
    #     qout_info = None
    #     if qerr_filepath is not None and os.path.exists(qerr_filepath):
    #         with open(qerr_filepath, "r") as f:
    #             qerr_info = f.read()
    #     if qout_filepath is not None and os.path.exists(qout_filepath):
    #         with open(qout_filepath, "r") as f:
    #             qout_info = f.read()
    #     if qerr_info or qout_info:
    #         from pymatgen.io.abinit.scheduler_error_parsers import get_parser
    #         qtype = queue_adapter.QTYPE
    #         scheduler_parser = get_parser(qtype, err_file=qerr_filepath,
    #                                       out_file=qout_filepath)
    #
    #         if scheduler_parser is None:
    #             raise ValueError('Cannot find scheduler_parser for qtype {}'.format(qtype))
    #
    #         scheduler_parser.parse()
    #         queue_errors = scheduler_parser.errors
    #
    #         # Get the timelimit error if there is one
    #         timelimit_error = None
    #         for error in queue_errors:
    #             if isinstance(error, TimeCancelError):
    #                 logger.debug('found timelimit error.')
    #                 timelimit_error = error
    #         if timelimit_error is None:
    #             note.state = ControllerNote.NOTHING_FOUND
    #             return note
    #
    #         if self.max_timelimit is None:
    #             max_timelimit = queue_adapter.timelimit_hard
    #         else:
    #             max_timelimit = self.max_timelimit
    #         # When timelimit_increase is not set, automatically take a tenth of the hard timelimit of the queue
    #         if self.timelimit_increase is None:
    #             timelimit_increase = queue_adapter.timelimit_hard / 10
    #         else:
    #             timelimit_increase = self.timelimit_increase
    #         old_timelimit = queue_adapter.timelimit
    #         if old_timelimit == max_timelimit:
    #                 # raise ValueError('Cannot increase beyond maximum timelimit ({:d} seconds) set in '
    #                 #                  'WalltimeController. Hard time limit of '
    #                 #                  'the queue is {:d} seconds'.format(max_timelimit,
    #                 #                                                     queue_adapter.timelimit_hard))
    #             note.state = ControllerNote.ERROR_UNRECOVERABLE
    #             return note
    #         new_timelimit = old_timelimit + timelimit_increase
    #         # If the new timelimit exceeds the max timelimit, just put it to the max timelimit
    #         if new_timelimit > max_timelimit:
    #             new_timelimit = max_timelimit
    #         actions['queue_adapter'] = Action(callable=queue_adapter.__class__.set_timelimit,
    #                                           timelimit=new_timelimit)
    #         note.state = ControllerNote.ERROR_RECOVERABLE
    #     else:
    #         note.state = ControllerNote.NOTHING_FOUND
    #     note.actions = actions
    #     note.simple_restart()
    #     return note


class UltimateMemoryController(Controller, QueueControllerMixin):
    """
    Controller for memory infringements of the resource manager. The handler should be able to handle the possible
    overhead of the master process.
    """

    is_handler = True
    #_controlled_item_types = [ControlledItemType.task_failed()]
    _controlled_item_types = [ControlledItemType.task_completed(), ControlledItemType.task_failed()]
    _only_unfinalized = True

#
#     def __init__(self, job_rundir='.', qout_file='queue.qout', qerr_file='queue.qerr', queue_adapter=None,
#                  max_mem_per_proc_mb=8000, mem_per_proc_increase_mb=1000,
#                  max_master_mem_overhead_mb=8000, master_mem_overhead_increase_mb=1000):

    def __init__(self, max_mem_per_proc_mb=8000, mem_per_proc_increase_mb=1000,
                       max_master_mem_overhead_mb=8000, master_mem_overhead_increase_mb=1000,
                       memory_policy='physical_memory'):
        """
        Initializes the handler with the directory where the job was run, the standard output and error files
        of the queue manager and the queue adapter used.

        Args:
            max_mem_per_proc_mb: Maximum memory per process in megabytes.
            mem_per_proc_increase_mb: Amount of memory to increase the memory per process in megabytes.
            max_master_mem_overhead_mb: Maximum overhead memory for the master process in megabytes.
            master_mem_overhead_increase_mb: Amount of memory to increase the overhead memory for the master process
                                             in megabytes.
            memory_policy: Policy for the memory (some weird clusters sometimes use the virtual memory to stop jobs
                           that overcome some virtual memory limit)
        """
        super(UltimateMemoryController, self).__init__()
        self.max_mem_per_proc_mb = max_mem_per_proc_mb
        self.mem_per_proc_increase_mb = mem_per_proc_increase_mb
        self.max_master_mem_overhead_mb = max_master_mem_overhead_mb
        self.master_mem_overhead_increase_mb = master_mem_overhead_increase_mb
        self.memory_policy = memory_policy
        self.priority = PRIORITY_LOWEST

    @property
    def memory_policy(self):
        return self._memory_policy

    @memory_policy.setter
    def memory_policy(self, memory_policy):
        if memory_policy not in ['physical_memory', 'virtual_memory']:
            raise ValueError('Memory policy is "{}" in UltimateMemoryController while it should be either '
                             '"physical_memory" or "virtual_memory"'.format(memory_policy))
        self._memory_policy = memory_policy

    def as_dict(self):
        return {'@class': self.__class__.__name__,
                '@module': self.__class__.__module__,
                'max_mem_per_proc_mb': self.max_mem_per_proc_mb,
                'mem_per_proc_increase_mb': self.mem_per_proc_increase_mb,
                'max_master_mem_overhead_mb': self.max_master_mem_overhead_mb,
                'master_mem_overhead_increase_mb': self.master_mem_overhead_increase_mb,
                'memory_policy': self.memory_policy
                }

    @classmethod
    def from_dict(cls, d):
        return cls(max_mem_per_proc_mb=d['max_mem_per_proc_mb'],
                   mem_per_proc_increase_mb=d['mem_per_proc_increase_mb'],
                   max_master_mem_overhead_mb=d['max_master_mem_overhead_mb'],
                   master_mem_overhead_increase_mb=d['master_mem_overhead_increase_mb'],
                   memory_policy=d['memory_policy'])

    @property
    def skip_remaining_handlers(self):
        return True

    @property
    def skip_lower_priority_controllers(self):
        return True

    def process(self, **kwargs):
        # Get the Queue Adapter and the queue errors
        queue_adapter = kwargs.get('queue_adapter', None)
        if queue_adapter is None:
            raise ValueError('No queue adapter passed to the UltimateMemoryController')
        queue_errors = self.get_queue_errors(memory_policy=self.memory_policy, **kwargs)

        # Create the Controller Note and the actions
        note = ControllerNote(controller=self)
        actions = {}

        # No errors found
        if queue_errors:
            raise ValueError('Error should have been caught by another controller ...')

        # No memory error found
        note.add_problem('Task has been stopped and we assume it is because of memory infringement by a slave process')
        old_mem_per_proc = queue_adapter.mem_per_proc
        if old_mem_per_proc == self.max_mem_per_proc_mb:
            note.state = ControllerNote.ERROR_UNRECOVERABLE
            note.add_problem('Maximum mem_per_proc has been reached, cannot increase further')
            return note
        new_mem_per_proc = old_mem_per_proc + self.mem_per_proc_increase_mb
        if new_mem_per_proc > self.max_mem_per_proc_mb:
            new_mem_per_proc = self.max_mem_per_proc_mb
        actions['queue_adapter'] = Action(callable=queue_adapter.__class__.set_mem_per_proc,
                                          mem_mb=new_mem_per_proc)

        # Set the actions to be performed, the state and the type of restart
        note.state = ControllerNote.ERROR_RECOVERABLE
        note.actions = actions
        note.reset_restart()
        return note


class SimpleValidatorController(Controller):
    """
    Simple validator controller to be applied after all other controllers (PRIORITY_LOWEST).
    This validator controller can be used when no "real" validator exists, but just handlers/monitors
    and that we suppose that if nothing is found by the handlers/monitors, then it means that it is ok.
    """

    can_validate = True
    _controlled_item_types = [ControlledItemType.task_completed()]

    def __init__(self):
        super(SimpleValidatorController, self).__init__()
        self.priority = PRIORITY_LOWEST

    def as_dict(self):
        return {'@class': self.__class__.__name__,
                '@module': self.__class__.__module__}

    @classmethod
    def from_dict(cls, d):
        return cls()

    @property
    def skip_remaining_handlers(self):
        return True

    @property
    def skip_lower_priority_controllers(self):
        return True

    def process(self, **kwargs):
        # Create the Controller Note
        note = ControllerNote(controller=self)
        note.state = ControllerNote.EVERYTHING_OK
        note.is_valid = True
        return note

    @property
    def validated(self):
        return True


class VaspXMLValidatorController(Controller):
    """
    Checks that a valid vasprun.xml was generated
    """
    can_validate = True
    _controlled_item_types = [ControlledItemType.task_completed()]

    def __init__(self):
        super(VaspXMLValidatorController, self).__init__()
        self.priority = PRIORITY_LOWEST

    def as_dict(self):
        return {'@class': self.__class__.__name__,
                '@module': self.__class__.__module__}

    @classmethod
    def from_dict(cls, d):
        return cls()

    @property
    def skip_remaining_handlers(self):
        return True

    @property
    def skip_lower_priority_controllers(self):
        return True

    def process(self, **kwargs):
        # Create the Controller Note
        if 'vasprun_xml_file' not in kwargs:
            raise ValueError('kwarg "vasprun_xml_file" is required to validate vasprun.xml file')
        vasprun_xml_file = kwargs['vasprun_xml_file']
        note = ControllerNote(controller=self)
        note.state = ControllerNote.EVERYTHING_OK
        note.is_valid = True
        try:
            Vasprun(vasprun_xml_file)
        except:
            note.state = ControllerNote.ERROR_NOFIX
            note.is_valid = False
        return note

    @property
    def validated(self):
        return True

class VaspNEBValidatorController(Controller):
    """
    Checks that a valid vasprun.xml was generated
    """
    can_validate = True
    _controlled_item_types = [ControlledItemType.task_completed()]

    def __init__(self):
        super(VaspNEBValidatorController, self).__init__()
        self.priority = PRIORITY_LOWEST

    def as_dict(self):
        return {'@class': self.__class__.__name__,
                '@module': self.__class__.__module__}

    @classmethod
    def from_dict(cls, d):
        return cls()

    @property
    def skip_remaining_handlers(self):
        return True

    @property
    def skip_lower_priority_controllers(self):
        return True

    def process(self, **kwargs):
        # Create the Controller Note
        if 'run_dir' not in kwargs:
            raise ValueError('kwarg "run_dir" is required to validate NEB vasp calculations')
        if 'additional_vasp_wf_info' not in kwargs:
            raise ValueError('kwarg "additional_vasp_wf_info" is required to validate NEB vasp calculations')
        if 'terminal_start_run_dir' not in kwargs['additional_vasp_wf_info']:
            raise ValueError('"terminal_start_run_dir" has to be in additional_vasp_wf_info is required '
                             'to validate NEB vasp calculations')
        if 'terminal_end_run_dir' not in kwargs['additional_vasp_wf_info']:
            raise ValueError('"terminal_end_run_dir" has to be in additional_vasp_wf_info is required '
                             'to validate NEB vasp calculations')
        run_dir = kwargs['run_dir']
        note = ControllerNote(controller=self)
        note.state = ControllerNote.EVERYTHING_OK
        note.is_valid = True
        terminal_dirs = (kwargs['additional_vasp_wf_info']['terminal_start_run_dir'],
                         kwargs['additional_vasp_wf_info']['terminal_end_run_dir'])
        try:
            NEBAnalysis.from_dir(run_dir, relaxation_dirs=terminal_dirs)
        except:
            note.state = ControllerNote.ERROR_NOFIX
            note.is_valid = False
        return note

    @property
    def validated(self):
        return True

# logger = logging.getLogger(__name__)
#
#
# class AbinitHandler(SRCErrorHandler):
#     """
#     General handler for abinit's critical events handlers.
#     """
#
#     def __init__(self, job_rundir='.', critical_events=None, queue_adapter=None):
#         """
#         Initializes the handler with the directory where the job was run.
#
#         Args:
#             job_rundir: Directory where the job was run.
#         """
#         super(AbinitHandler, self).__init__()
#         self.job_rundir = job_rundir
#         self.critical_events = critical_events
#
#         self.src_fw = False
#
#     def as_dict(self):
#         return {'@class': self.__class__.__name__,
#                 '@module': self.__class__.__module__,
#                 'job_rundir': self.job_rundir
#                 }
#
#     @classmethod
#     def from_dict(cls, d):
#         return cls(job_rundir=d['job_rundir'])
#
#     @property
#     def allow_fizzled(self):
#         return False
#
#     @property
#     def allow_completed(self):
#         return True
#
#     @property
#     def handler_priority(self):
#         return self.PRIORITY_MEDIUM
#
#     @property
#     def skip_remaining_handlers(self):
#         return True
#
#     def setup(self):
#         if 'SRCScheme' in self.fw_to_check.spec and self.fw_to_check.spec['SRCScheme']:
#             self.src_fw = True
#         else:
#             self.src_fw = False
#         self.job_rundir = self.fw_to_check.launches[-1].launch_dir
#
#     def check(self):
#         abinit_task = self.fw_to_check.tasks[0]
#         self.report = None
#         try:
#             self.report = abinit_task.get_event_report()
#         except Exception as exc:
#             msg = "%s exception while parsing event_report:\n%s" % (self, exc)
#             logger.critical(msg)
#
#         if self.report is not None:
#             # Run has completed, check for critical events (convergence, ...)
#             if self.report.run_completed:
#                 self.events = self.report.filter_types(abinit_task.CRITICAL_EVENTS)
#                 if self.events:
#                     return True
#                 else:
#                     # Calculation has converged
#                     # Check if there are custom parameters that should be converged
#                     unconverged_params, reset_restart = abinit_task.check_parameters_convergence(self.fw_to_check.spec)
#                     if unconverged_params:
#                         return True
#                     else:
#                         return False
#             # Abinit run failed to complete
#             # Check if the errors can be handled
#             if self.report.errors:
#                 return True
#         return True
#
#     def has_corrections(self):
#         return True
#
#     def correct(self):
#         if self.src_fw:
#             if len(self.fw_to_check.tasks) > 1:
#                 raise ValueError('More than 1 task found in fizzled firework, not yet supported')
#             abinit_input_update = {'iscf': 2}
#             return {'errors': [self.__class__.__name__],
#                     'actions': [{'action_type': 'modify_object',
#                                  'object': {'source': 'fw_spec', 'key': 'abinit_input'},
#                                  'action': {'_set': abinit_input_update}}]}
#         else:
#             raise NotImplementedError('This handler cannot be used without the SRC scheme')
#
#
# class WalltimeHandler(SRCErrorHandler):
#     """
#     Handler for walltime infringements of the resource manager.
#     """
#
#     def __init__(self, job_rundir='.', qout_file='queue.qout', qerr_file='queue.qerr', queue_adapter=None,
#                  max_timelimit=None, timelimit_increase=None):
#         """
#         Initializes the handler with the directory where the job was run, the standard output and error files
#         of the queue manager and the queue adapter used.
#
#         Args:
#             job_rundir: Directory where the job was run.
#             qout_file: Standard output file of the queue manager.
#             qerr_file: Standard error file of the queue manager.
#             queue_adapter: Queue adapter used to submit the job.
#             max_timelimit: Maximum timelimit (in seconds) allowed by the resource manager for the queue.
#             timelimit_increase: Amount of time (in seconds) to increase the timelimit.
#         """
#         super(WalltimeHandler, self).__init__()
#         self.job_rundir = job_rundir
#         self.qout_file = qout_file
#         self.qerr_file = qerr_file
#         self.queue_adapter = queue_adapter
#         self.setup_filepaths()
#         self.max_timelimit = max_timelimit
#         self.timelimit_increase = timelimit_increase
#
#         self.src_fw = False
#
#     def setup_filepaths(self):
#         self.qout_filepath = os.path.join(self.job_rundir, self.qout_file)
#         self.qerr_filepath = os.path.join(self.job_rundir, self.qerr_file)
#
#     def as_dict(self):
#         return {'@class': self.__class__.__name__,
#                 '@module': self.__class__.__module__,
#                 'job_rundir': self.job_rundir,
#                 'qout_file': self.qout_file,
#                 'qerr_file': self.qerr_file,
#                 'queue_adapter': self.queue_adapter.as_dict() if self.queue_adapter is not None else None,
#                 'max_timelimit': self.max_timelimit,
#                 'timelimit_increase': self.timelimit_increase
#                 }
#
#     @classmethod
#     def from_dict(cls, d):
#         qa = QueueAdapter.from_dict(d['queue_adapter']) if d['queue_adapter'] is not None else None
#         return cls(job_rundir=d['job_rundir'], qout_file=d['qout_file'], qerr_file=d['qerr_file'], queue_adapter=qa,
#                    max_timelimit=d['max_timelimit'],
#                    timelimit_increase=d['timelimit_increase'])
#
#     @property
#     def allow_fizzled(self):
#         return True
#
#     @property
#     def allow_completed(self):
#         return False
#
#     @property
#     def handler_priority(self):
#         return self.PRIORITY_VERY_LOW
#
#     @property
#     def skip_remaining_handlers(self):
#         return True
#
#     def setup(self):
#         if 'SRCScheme' in self.fw_to_check.spec and self.fw_to_check.spec['SRCScheme']:
#             self.src_fw = True
#         else:
#             self.src_fw = False
#         self.job_rundir = self.fw_to_check.launches[-1].launch_dir
#         self.setup_filepaths()
#         self.queue_adapter = self.fw_to_check.spec['qtk_queueadapter']
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
#         self.timelimit_error = None
#         self.queue_errors = None
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
#             for error in self.queue_errors:
#                 if isinstance(error, TimeCancelError):
#                     logger.debug('found timelimit error.')
#                     self.timelimit_error = error
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
#             # When max_timelimit is not set, automatically take the hard timelimit of the queue
#             if self.max_timelimit is None:
#                 max_timelimit = self.queue_adapter.timelimit_hard
#             else:
#                 max_timelimit = self.max_timelimit
#             # When timelimit_increase is not set, automatically take a tenth of the hard timelimit of the queue
#             if self.timelimit_increase is None:
#                 timelimit_increase = self.queue_adapter.timelimit_hard / 10
#             else:
#                 timelimit_increase = self.timelimit_increase
#             if isinstance(self.timelimit_error, TimeCancelError):
#                 old_timelimit = self.queue_adapter.timelimit
#                 if old_timelimit == max_timelimit:
#                     raise ValueError('Cannot increase beyond maximum timelimit ({:d} seconds) set in WalltimeHandler.'
#                                      'Hard time limit of '
#                                      'the queue is {:d} seconds'.format(max_timelimit,
#                                                                         self.queue_adapter.timelimit_hard))
#                 new_timelimit = old_timelimit + timelimit_increase
#                 # If the new timelimit exceeds the max timelimit, just put it to the max timelimit
#                 if new_timelimit > max_timelimit:
#                     new_timelimit = max_timelimit
#                 queue_adapter_update['timelimit'] = new_timelimit
#             else:
#                 raise ValueError('Should not be here ...')
#             return {'errors': [self.__class__.__name__],
#                     'actions': [{'action_type': 'modify_object',
#                                  'object': {'source': 'fw_spec', 'key': 'qtk_queueadapter'},
#                                  'action': {'_set': queue_adapter_update}}]}
#         else:
#             raise NotImplementedError('This handler cannot be used without the SRC scheme')
#
#     def has_corrections(self):
#         return True
#
#
# class MemoryHandler(SRCErrorHandler):
#     """
#     Handler for memory infringements of the resource manager. The handler should be able to handle the possible
#     overhead of the master process.
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
#         super(MemoryHandler, self).__init__()
#         self.job_rundir = job_rundir
#         self.qout_file = qout_file
#         self.qerr_file = qerr_file
#         self.queue_adapter = queue_adapter
#         self.setup_filepaths()
#         self.max_mem_per_proc_mb = max_mem_per_proc_mb
#         self.mem_per_proc_increase_mb = mem_per_proc_increase_mb
#         self.max_master_mem_overhead_mb = max_master_mem_overhead_mb
#         self.master_mem_overhead_increase_mb = master_mem_overhead_increase_mb
#
#         self.src_fw = False
#
#     def setup_filepaths(self):
#         self.qout_filepath = os.path.join(self.job_rundir, self.qout_file)
#         self.qerr_filepath = os.path.join(self.job_rundir, self.qerr_file)
#
#     def as_dict(self):
#         return {'@class': self.__class__.__name__,
#                 '@module': self.__class__.__module__,
#                 'job_rundir': self.job_rundir,
#                 'qout_file': self.qout_file,
#                 'qerr_file': self.qerr_file,
#                 'queue_adapter': self.queue_adapter.as_dict() if self.queue_adapter is not None else None,
#                 'max_mem_per_proc_mb': self.max_mem_per_proc_mb,
#                 'mem_per_proc_increase_mb': self.mem_per_proc_increase_mb,
#                 'max_master_mem_overhead_mb': self.max_master_mem_overhead_mb,
#                 'master_mem_overhead_increase_mb': self.master_mem_overhead_increase_mb
#                 }
#
#     @classmethod
#     def from_dict(cls, d):
#         qa = QueueAdapter.from_dict(d['queue_adapter']) if d['queue_adapter'] is not None else None
#         return cls(job_rundir=d['job_rundir'], qout_file=d['qout_file'], qerr_file=d['qerr_file'], queue_adapter=qa,
#                    max_mem_per_proc_mb=d['max_mem_per_proc_mb'],
#                    mem_per_proc_increase_mb=d['mem_per_proc_increase_mb'],
#                    max_master_mem_overhead_mb=d['max_master_mem_overhead_mb'],
#                    master_mem_overhead_increase_mb=d['master_mem_overhead_increase_mb'])
#
#     @property
#     def allow_fizzled(self):
#         return True
#
#     @property
#     def allow_completed(self):
#         return False
#
#     @property
#     def handler_priority(self):
#         return self.PRIORITY_VERY_LOW
#
#     @property
#     def skip_remaining_handlers(self):
#         return True
#
#     def setup(self):
#         if 'SRCScheme' in self.fw_to_check.spec and self.fw_to_check.spec['SRCScheme']:
#             self.src_fw = True
#         else:
#             self.src_fw = False
#         self.job_rundir = self.fw_to_check.launches[-1].launch_dir
#         self.setup_filepaths()
#         self.queue_adapter = self.fw_to_check.spec['qtk_queueadapter']
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
#         self.queue_errors = None
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
#                     'actions': [{'action_type': 'modify_object',
#                                  'object': {'source': 'fw_spec', 'key': 'qtk_queueadapter'},
#                                  'action': {'_set': queue_adapter_update}}]}
#         else:
#             raise NotImplementedError('This handler cannot be used without the SRC scheme')
#
#     def has_corrections(self):
#         return True
#
#
# class UltimateMemoryHandler(MemoryHandler):
#     """
#     Handler for infringements of the resource manager. If no memory error is found,
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
#         super(UltimateMemoryHandler, self).__init__()
#
#     @property
#     def handler_priority(self):
#         return self.PRIORITY_LAST
#
#     def check(self):
#         mem_check = super(UltimateMemoryHandler, self).check()
#         if mem_check:
#             raise ValueError('This error should have been caught by a standard MemoryHandler ...')
#         #TODO: Do we have some check that we can do here ?
#         return True
#
#     def correct(self):
#         if self.src_fw:
#             if len(self.fw_to_check.tasks) > 1:
#                 raise ValueError('More than 1 task found in "memory-fizzled" firework, not yet supported')
#             if self.memory_error is not None:
#                 raise ValueError('This error should have been caught by a standard MemoryHandler ...')
#             if self.queue_errors is not None and len(self.queue_errors) > 0:
#                 raise ValueError('Queue errors were found ... these should be handled properly by another handler')
#             # Information about the update of the memory (base mem per proc) in the queue adapter
#             queue_adapter_update = {}
#             old_mem_per_proc = self.queue_adapter.mem_per_proc
#             new_mem_per_proc = old_mem_per_proc + self.mem_per_proc_increase_mb
#             queue_adapter_update['mem_per_proc'] = new_mem_per_proc
#             return {'errors': [self.__class__.__name__],
#                     'actions': [{'action_type': 'modify_object',
#                                  'object': {'source': 'fw_spec', 'key': 'qtk_queueadapter'},
#                                  'action': {'_set': queue_adapter_update}}]}
#         else:
#             raise NotImplementedError('This handler cannot be used without the SRC scheme')