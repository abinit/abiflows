# coding: utf-8
"""
Task classes for Fireworks.
"""
from __future__ import print_function, division, unicode_literals

import inspect
import subprocess
import logging
import collections
import time
import shutil
import json
import threading
import glob
import os
import errno
import numpy as np
from fireworks.core.firework import Firework, FireTaskBase, FWAction, Workflow
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.utilities.fw_serializers import serialize_fw
from collections import namedtuple, defaultdict
from abiflows.fireworks.utils.task_history import TaskHistory
from pymatgen.io.abinit.utils import Directory, File
from pymatgen.io.abinit import events, tasks
from pymatgen.io.abinit.utils import irdvars_for_ext
from pymatgen.io.abinit.wrappers import Mrgddb
from pymatgen.serializers.json_coders import json_pretty_dump, pmg_serialize
from monty.json import MontyEncoder, MontyDecoder, MSONable
from abipy.abio.factories import InputFactory
from abipy.abio.inputs import AbinitInput
from abipy.dfpt.ddb import ElasticComplianceTensor
from abipy.abio.input_tags import *

from abipy.core import Structure

from abiflows.fireworks.tasks.abinit_common import TMPDIR_NAME, OUTDIR_NAME, INDIR_NAME, STDERR_FILE_NAME, \
    LOG_FILE_NAME, FILES_FILE_NAME, OUTPUT_FILE_NAME, INPUT_FILE_NAME, MPIABORTFILE, DUMMY_FILENAME, \
    ELPHON_OUTPUT_FILE_NAME, DDK_FILES_FILE_NAME, HISTORY_JSON
from abiflows.fireworks.utils.fw_utils import FWTaskManager
from abiflows.fireworks.utils.task_history import TaskHistory
from abiflows.fireworks.tasks.utility_tasks import SRCFireworks

logger = logging.getLogger(__name__)


# files and folders names

class BasicTaskMixin(object):
    task_type = ""

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

    def get_fw_task_manager(self, fw_spec):
        if 'ftm_file' in fw_spec:
            ftm = FWTaskManager.from_file(fw_spec['ftm_file'])
        else:
            ftm = FWTaskManager.from_user_config()
        ftm.update_fw_policy(fw_spec.get('fw_policy', {}))
        return ftm

    def run_autoparal(self, abiinput, autoparal_dir, ftm, clean_up='move'):
        """
        Runs the autoparal using AbinitInput abiget_autoparal_pconfs method.
        The information are retrieved from the FWTaskManager that should be present and contain the standard
        abipy TaskManager, that provides information about the queue adapters.
        No check is performed on the autoparal_dir. If there is a possibility of overwriting output data due to
        reuse of the same folder, it should be handled by the caller.
        """
        manager = ftm.task_manager
        if not manager:
            msg = 'No task manager available: autoparal could not be performed.'
            logger.error(msg)
            raise InitializationError(msg)
        pconfs = abiinput.abiget_autoparal_pconfs(max_ncpus=manager.max_cores, workdir=autoparal_dir,
                                                       manager=manager)
        optconf = manager.select_qadapter(pconfs)
        qadapter_spec = manager.qadapter.get_subs_dict()

        d = pconfs.as_dict()
        d["optimal_conf"] = optconf
        json_pretty_dump(d, os.path.join(autoparal_dir, "autoparal.json"))

        # Method to clean the output files
        def safe_rm(name):
            try:
                path = os.path.join(autoparal_dir, name)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except OSError:
                pass

        # Method to rename the output files
        def safe_mv(name):
            try:
                autoparal_backup_dir = os.path.join(autoparal_dir, 'autoparal_backup')
                if not os.path.exists(autoparal_backup_dir):
                    os.makedirs(autoparal_backup_dir)
                current_path = os.path.join(autoparal_dir, name)
                newpath = os.path.join(autoparal_backup_dir, name)
                if not os.path.exists(newpath):
                    shutil.move(current_path, newpath)
                else:
                    raise ValueError('Autoparal backup file already exists for the file "{}"'.format(name))
            except OSError:
                pass

        # clean up useless files. The output files should removed also to avoid abinit renaming the out file in
        # case the main run will be performed in the same dir

        if clean_up == 'move':
            to_be_moved = [OUTPUT_FILE_NAME, LOG_FILE_NAME, STDERR_FILE_NAME]
            for r in to_be_moved:
                safe_mv(r)
            to_be_removed = [TMPDIR_NAME, OUTDIR_NAME, INDIR_NAME]
            for r in to_be_removed:
                safe_rm(r)
        elif clean_up == 'remove':
            to_be_removed = [OUTPUT_FILE_NAME, LOG_FILE_NAME, STDERR_FILE_NAME, TMPDIR_NAME, OUTDIR_NAME, INDIR_NAME]
            for r in to_be_removed:
                safe_rm(r)

        return optconf, qadapter_spec, manager.qadapter

    def get_final_mod_spec(self, fw_spec):
        return [{'_push': {'previous_fws->'+self.task_type: self.current_task_info(fw_spec)}}]
        # if 'previous_fws' in fw_spec:
        #     prev_fws = fw_spec['previous_fws'].copy()
        # else:
        #     prev_fws = {}
        # prev_fws[self.task_type] = [self.current_task_info(fw_spec)]
        # return [{'_set': {'previous_fws': prev_fws}}]

    def set_logger(self):
        # Set a logger for pymatgen.io.abinit and abipy
        log_handler = logging.FileHandler('abipy.log')
        log_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
        logging.getLogger('pymatgen.io.abinit').addHandler(log_handler)
        logging.getLogger('abipy').addHandler(log_handler)
        logging.getLogger('abiflows').addHandler(log_handler)

    def link_ext(self, ext, source_dir, strict=True):
        source = os.path.join(source_dir, self.prefix.odata + "_" + ext)
        logger.info("Need path {} with ext {}".format(source, ext))
        dest = os.path.join(self.workdir, self.prefix.idata + "_" + ext)
        if not os.path.exists(source):
            # Try netcdf file. TODO: this case should be treated in a cleaner way.
            source += "-etsf.nc"
            if os.path.exists(source): dest += "-etsf.nc"
        if not os.path.exists(source):
            if strict:
                msg = "{} is needed by this task but it does not exist".format(source)
                logger.error(msg)
                raise InitializationError(msg)
            else:
                return None

        # Link path to dest if dest link does not exist.
        # else check that it points to the expected file.
        logger.info("Linking path {} --> {}".format(source, dest))
        if not os.path.exists(dest) or not strict:
            if self.ftm.fw_policy.copy_deps:
                shutil.copyfile(source, dest)
            else:
                os.symlink(source, dest)
            return dest
        else:
            # check links but only if we haven't performed the restart.
            # in this case, indeed we may have replaced the file pointer with the
            # previous output file of the present task.
            if not self.ftm.fw_policy.copy_deps and os.path.realpath(dest) != source and not self.restart_info:
                msg = "dest {} does not point to path {}".format(dest, source)
                logger.error(msg)
                raise InitializationError(msg)

    def link_ddk(self, source_dir):
        # handle the custom DDK extension on its own
        # accept more than one DDK file in the outdir: multiple perturbations are allowed in a
        # single calculation
        outdata_dir = Directory(os.path.join(source_dir, OUTDIR_NAME))
        ddks = []
        for f in outdata_dir.list_filepaths():
            if f.endswith('_DDK'):
                ddks.append(f)
        if not ddks:
            msg = "DDK is needed by this task but it does not exist"
            logger.error(msg)
            raise InitializationError(msg)
        exts = [os.path.basename(ddk).split('_')[-2] for ddk in ddks]
        ddk_files = []
        for ext in exts:
            ddk_files.append(self.link_ext(ext, source_dir))

        return ddk_files

    #from Task
    # Prefixes for Abinit (input, output, temporary) files.
    Prefix = collections.namedtuple("Prefix", "idata odata tdata")
    pj = os.path.join

    prefix = Prefix(pj("indata", "in"), pj("outdata", "out"), pj("tmpdata", "tmp"))
    del Prefix, pj


@explicit_serialize
class AbiFireTask(BasicTaskMixin, FireTaskBase):

    # List of `AbinitEvent` subclasses that are tested in the check_status method.
    # Subclasses should provide their own list if they need to check the converge status.
    CRITICAL_EVENTS = [
    ]

    def __init__(self, abiinput, restart_info=None, handlers=[], is_autoparal=None, deps=None, history=[],
                 use_SRC_scheme=False, task_type=None):
        """
        Basic __init__, subclasses are supposed to define the same input parameters, add their own and call super for
        the basic ones. The input parameter should be stored as attributes of the instance for serialization and
        for inspection.
        """
        self.abiinput = abiinput
        self.restart_info = restart_info

        self.handlers = handlers or [cls() for cls in events.get_event_handler_classes()]
        self.is_autoparal = is_autoparal
        self.use_SRC_scheme = use_SRC_scheme

        #TODO: rationalize this and check whether this might create problems due to the fact that if task_type is None,
        #      self.task_type is the class variable (actually self.task_type refers to self.__class__.task_type) while
        #      if task_type is specified, self.task_type is an instance variable and is potentially different from
        #      self.__class__.task_type !
        if task_type is not None:
            self.task_type = task_type

        # deps are transformed to be a list or a dict of lists
        if isinstance(deps, dict):
            deps = dict(deps)
            for k, v in deps.items():
                if not isinstance(v, (list, tuple)):
                    deps[k] = [v]
        elif deps and not isinstance(deps, (list, tuple)):
            deps = [deps]
        self.deps = deps

        # create a copy
        self.history = TaskHistory(history)

    #from Task
    def set_workdir(self, workdir):
        """Set the working directory."""

        self.workdir = os.path.abspath(workdir)

        # Files required for the execution.
        self.input_file = File(os.path.join(self.workdir, INPUT_FILE_NAME))
        self.output_file = File(os.path.join(self.workdir, OUTPUT_FILE_NAME))
        self.files_file = File(os.path.join(self.workdir, FILES_FILE_NAME))
        self.log_file = File(os.path.join(self.workdir, LOG_FILE_NAME))
        self.stderr_file = File(os.path.join(self.workdir, STDERR_FILE_NAME))

        # This file is produce by Abinit if nprocs > 1 and MPI_ABORT.
        self.mpiabort_file = File(os.path.join(self.workdir, MPIABORTFILE))

        # Directories with input|output|temporary data.
        self.indir = Directory(os.path.join(self.workdir, INDIR_NAME))
        self.outdir = Directory(os.path.join(self.workdir, OUTDIR_NAME))
        self.tmpdir = Directory(os.path.join(self.workdir, TMPDIR_NAME))

    #from abitask
    def rename_outputs(self):
        """
        If rerunning in the same folder, we rename the outputs according to the abipy convention:
        Abinit has the very *bad* habit of changing the file extension by appending the characters in [A,B ..., Z]
        to the output file, and this breaks a lot of code that relies of the use of a unique file extension.
        Here we fix this issue by renaming run.abo to run.abo_[number] if the output file "run.abo" already
        exists.
        This is applied both if the calculation is rerun from scratch or with a restart in the same folder.
        """

        files_to_rename = [self.output_file.path, self.log_file.path]

        if not any(os.path.isfile(f) for f in files_to_rename):
            return

        file_paths = [f for file_path in files_to_rename for f in glob.glob(file_path+'*')]
        nums = [int(f) for f in [f.split("_")[-1] for f in file_paths] if f.isdigit()]
        new_index = (max(nums) if nums else 0) +1

        for f in files_to_rename:
            try:
                new_path = f + '_' + str(new_index)
                os.rename(f, new_path)
                logger.info("Renamed %s to %s" % (f, new_path))
            except OSError as exc:
                logger.warning("couldn't rename {} to {} : {} ".format(f, new_path, str(exc)))

    #from AbintTask
    @property
    def filesfile_string(self):
        """String with the list of files and prefixes needed to execute ABINIT."""
        lines = []
        app = lines.append
        pj = os.path.join

        app(self.input_file.path)                 # Path to the input file
        app(self.output_file.path)                # Path to the output file
        app(pj(self.workdir, self.prefix.idata))  # Prefix for input data
        app(pj(self.workdir, self.prefix.odata))  # Prefix for output data
        app(pj(self.workdir, self.prefix.tdata))  # Prefix for temporary data

        # Paths to the pseudopotential files.
        # Note that here the pseudos **must** be sorted according to znucl.
        # Here we reorder the pseudos if the order is wrong.
        ord_pseudos = []
        znucl = self.abiinput.structure.to_abivars()["znucl"]

        for z in znucl:
            for p in self.abiinput.pseudos:
                if p.Z == z:
                    ord_pseudos.append(p)
                    break
            else:
                raise ValueError("Cannot find pseudo with znucl %s in pseudos:\n%s" % (z, self.pseudos))

        for pseudo in ord_pseudos:
            app(pseudo.path)

        return "\n".join(lines)

    def config_run(self, fw_spec):
        """
        Configure the job for the run:
        - set up logging system
        - sets and creates the directories and the input files needed to run the task.
        - sets dependencies and data from previous run (both the case of a restart in the same folder as the previous
          FW and the case of a creation of a new folder).
        """

        # rename outputs if rerunning in the same dir
        self.rename_outputs()

        # Copy the appropriate dependencies in the in dir
        #TODO it should be clarified if this should stay here or in setup_task().
        self.resolve_deps(fw_spec)

        # if it's the restart of a previous task, perform specific task updates.
        # perform these updates before writing the input, but after creating the dirs.
        if self.restart_info:
            #TODO add if it is a local restart or not
            self.history.log_restart(self.restart_info)
            self.restart()

        # Write files file and input file.
        if not self.files_file.exists:
            self.files_file.write(self.filesfile_string)

        self.input_file.write(str(self.abiinput))

    def run_abinit(self, fw_spec):
        """
        executes abinit and waits for the end of the process.
        the mpirun and abinit commands are retrived from the mpirun_cmd and abinit_cmd keys in the fw_polity
        of the FWTaskManager, that can be overridden by the values in the spec.
        Note that in case of missing definition of these parameters, the values fall back to the default
        values of  mpirun_cmd and abinit_cmd: 'mpirun' and 'abinit', assuming that these are properly retrived
        from the PATH
        """

        def abinit_process():
            command = []
            #consider the case of serial execution
            if self.ftm.fw_policy.mpirun_cmd:
                command.extend(self.ftm.fw_policy.mpirun_cmd.split())
                if 'mpi_ncpus' in fw_spec:
                    command.extend(['-np', str(fw_spec['mpi_ncpus'])])
            command.append(self.ftm.fw_policy.abinit_cmd)
            with open(self.files_file.path, 'r') as stdin, open(self.log_file.path, 'w') as stdout, \
                    open(self.stderr_file.path, 'w') as stderr:
                self.process = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=stderr)

            (stdoutdata, stderrdata) = self.process.communicate()
            self.returncode = self.process.returncode

        # initialize returncode to avoid missing references in case of exception in the other thread
        self.returncode = None

        thread = threading.Thread(target=abinit_process)
        # the amount of time left plus a buffer of 2 minutes
        timeout = (self.walltime - (time.time() - self.start_time) - 120) if self.walltime else None
        start_abinit_time = time.time()
        thread.start()
        thread.join(timeout)
        self.history.log_abinit_stop(run_time=(time.time() - start_abinit_time))
        if thread.is_alive():
            self.process.terminate()
            thread.join()
            raise WalltimeError("The task couldn't be terminated within the time limit. Killed.")

    def get_event_report(self, source='log'):
        """
        Analyzes the main output file for possible Errors or Warnings.

        Returns:
            :class:`EventReport` instance or None if the main output file does not exist.
        """
        ofile = {
            "output": self.output_file,
            "log": self.log_file}[source]

        parser = events.EventsParser()

        if not ofile.exists:
            if not self.mpiabort_file.exists:
                return None
            else:
                # ABINIT abort file without log!
                abort_report = parser.parse(self.mpiabort_file.path)
                return abort_report

        try:
            report = parser.parse(ofile.path)

            # Add events found in the ABI_MPIABORTFILE.
            if self.mpiabort_file.exists:
                logger.critical("Found ABI_MPIABORTFILE!")
                abort_report = parser.parse(self.mpiabort_file.path)
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

    def task_analysis(self, fw_spec):
        """
        This function checks final status of the calculation, inspecting the output and error files.
        Raises an AbinitRuntimeError if unfixable errors are encountered.
        """

        self.report = None
        try:
            self.report = self.get_event_report()
        except Exception as exc:
            msg = "%s exception while parsing event_report:\n%s" % (self, exc)
            logger.critical(msg)

        # If the calculation is ok, parse the outputs
        if self.report is not None:
            # the calculation finished without errors
            if self.report.run_completed:
                # Check if the calculation converged.
                not_ok = self.report.filter_types(self.CRITICAL_EVENTS)
                if not_ok:
                    self.history.log_unconverged()
                    # hook
                    local_restart, restart_fw, stored_data = self.prepare_restart(fw_spec)
                    num_restarts = self.restart_info.num_restarts if self.restart_info else 0
                    if num_restarts < self.ftm.fw_policy.max_restarts:
                        if local_restart:
                            return None
                        else:
                            stored_data['final_state'] = 'Unconverged'
                            return FWAction(detours=restart_fw, stored_data=stored_data)
                    else:
                        raise UnconvergedError(self, msg="Unconverged after {} restarts".format(num_restarts),
                                               abiinput=self.abiinput, restart_info=self.restart_info,
                                               history=self.history)
                else:
                    # calculation converged
                    # check if there are custom parameters that should be converged
                    unconverged_params, reset_restart = self.check_parameters_convergence(fw_spec)
                    if unconverged_params:
                        self.history.log_converge_params(unconverged_params, self.abiinput)
                        self.abiinput.set_vars(**unconverged_params)
                        local_restart, restart_fw, stored_data = self.prepare_restart(fw_spec, reset=reset_restart)
                        num_restarts = self.restart_info.num_restarts if self.restart_info else 0
                        if num_restarts < self.ftm.fw_policy.max_restarts:
                            if local_restart:
                                return None
                            else:
                                stored_data['final_state'] = 'Unconverged_parameters'
                                return FWAction(detours=restart_fw, stored_data=stored_data)
                        else:
                            raise UnconvergedParametersError(self, abiinput=self.abiinput,
                                                             restart_info=self.restart_info, history=self.history)
                    else:
                        # everything is ok. conclude the task
                        # hook
                        update_spec, mod_spec, stored_data = self.conclude_task(fw_spec)
                        return FWAction(stored_data=stored_data, update_spec=update_spec, mod_spec=mod_spec)

            # Abinit reported problems
            # Check if the errors could be handled
            if self.report.errors:
                logger.debug('Found errors in report')
                for error in self.report.errors:
                    logger.debug(str(error))
                    try:
                        self.abi_errors.append(error)
                    except AttributeError:
                        self.abi_errors = [error]

                # ABINIT errors, try to handle them
                fixed, reset = self.fix_abicritical(fw_spec)
                if fixed:
                    local_restart, restart_fw, stored_data = self.prepare_restart(fw_spec, reset=reset)
                    if local_restart:
                        return None
                    else:
                        return FWAction(detours=restart_fw, stored_data=stored_data)
                else:
                    msg = "Critical events couldn't be fixed by handlers. return code".format(self.returncode)
                    logger.error(msg)
                    raise AbinitRuntimeError(self, "Critical events couldn't be fixed by handlers")

        # No errors from abinit. No fix could be applied at this stage.
        # The FW will be fizzled.
        # Try to save the stderr file for Fortran runtime errors.
        #TODO check if some cases could be handled here
        err_msg = None
        if self.stderr_file.exists:
            #TODO length should always be enough, but maybe it's worth cutting the message if it's too long
            err_msg = self.stderr_file.read()
            # It happened that the text file contained non utf-8 characters.
            # sanitize the text to avoid problems during database inserption
            err_msg.decode("utf-8", "ignore")
        logger.error("return code {}".format(self.returncode))
        raise AbinitRuntimeError(self, err_msg)

    def check_parameters_convergence(self, fw_spec):
        return {}, False

    def _get_init_args_and_vals(self):
        init_dict = {}
        for arg in inspect.getargspec(self.__init__).args:
            if arg != "self":
                init_dict[arg] = self.__getattribute__(arg)

        return init_dict

    def _exclude_from_spec_in_restart(self):
        return ['_tasks', '_exception_details']

    def prepare_restart(self, fw_spec, reset=False):
        if self.restart_info:
            num_restarts = self.restart_info.num_restarts + 1
        else:
            num_restarts = 1

        self.restart_info = RestartInfo(previous_dir=self.workdir, reset=reset, num_restarts=num_restarts)

        # forward all the specs of the task
        new_spec = {k: v for k, v in fw_spec.items() if k not in self._exclude_from_spec_in_restart()}

        local_restart = False
        # only restart if it is known that there is a reasonable amount of time left
        if self.ftm.fw_policy.allow_local_restart and self.walltime and self.walltime/2 > (time.time() - self.start_time):
            local_restart = True

        if self.use_SRC_scheme:
            fw_task_index = int(fw_spec['wf_task_index'].split('_')[-1])
            new_index = fw_task_index + 1
            SRC_fws = SRCFireworks(task_class=self, task_input=self.abiinput, spec=new_spec,
                                   initialization_info=fw_spec['initialization_info'],
                                   wf_task_index_prefix=fw_spec['wf_task_index_prefix'],
                                   current_task_index=new_index)
            wf = Workflow(fireworks=SRC_fws['fws'], links_dict=SRC_fws['links_dict'])
            return FWAction(detours=[wf])

        # run here the autorun, otherwise it would need a separated FW
        if self.ftm.fw_policy.autoparal:
            # in case of restarting from the same folder the autoparal subfolder can already exist
            # create a new one with increasing number
            i = 0
            while os.path.exists(os.path.join(self.workdir, "autoparal{}".format("_"+str(i) if i else ""))):
                i += 1
            autoparal_dir = os.path.join(self.workdir, "autoparal{}".format("_"+str(i) if i else ""))
            optconf, qadapter_spec, qtk_qadapter = self.run_autoparal(self.abiinput, autoparal_dir, self.ftm)
            self.history.log_autoparal(optconf)
            self.abiinput.set_vars(optconf.vars)
            # set quadapter specification.
            new_spec['_queueadapter'] = qadapter_spec
            new_spec['mpi_ncpus'] = optconf['mpi_ncpus']

            # if autoparal enabled, the number of processors should match the current number to restart in place
            local_restart = local_restart and qadapter_spec == fw_spec.get('_queueadapter', {})

        # increase the index associated with the specific task in the workflow
        if 'wf_task_index' in fw_spec:
            split = fw_spec['wf_task_index'].split('_')
            new_spec['wf_task_index'] = '{}_{:d}'.format('_'.join(split[:-1]), int(split[-1])+1)

        # new task. Construct it from the actual values of the input parameters
        restart_task = self.__class__(**self._get_init_args_and_vals())

        if self.ftm.fw_policy.rerun_same_dir:
            new_spec['_launch_dir'] = self.workdir

        # create the new FW
        new_fw = Firework([restart_task], spec=new_spec)

        # At this point the event report should be present
        stored_data = self.report.as_dict()
        stored_data['finalized'] = False
        stored_data['restarted'] = True

        return local_restart, new_fw, stored_data

    def fix_abicritical(self, fw_spec):
        """
        method to fix crashes/error caused by abinit

        Returns:
            retcode: 1 if task has been fixed else 0.
            reset: True if at least one of the corrections applied requires a reset
        """
        if not self.handlers:
            logger.info('Empty list of event handlers. Cannot fix abi_critical errors')
            return 0

        done = len(self.handlers) * [0]
        corrections = []

        for event in self.report:
            for i, handler in enumerate(self.handlers):
                if handler.can_handle(event) and not done[i]:
                    logger.info("handler", handler, "will try to fix", event)
                    try:
                        c = handler.handle_input_event(self.abiinput, self.outdir, event)
                        if c:
                            done[i] += 1
                            corrections.append(c)

                    except Exception as exc:
                        logger.critical(str(exc))

        if corrections:
            reset = any(c.reset for c in corrections)
            self.history.log_corrections(corrections)
            return 1, reset

        logger.info('We encountered AbiCritical events that could not be fixed')
        return 0, None

    def setup_task(self, fw_spec):
        self.start_time = time.time()

        self.set_logger()

        # load the FWTaskManager to get configuration parameters
        self.ftm = self.get_fw_task_manager(fw_spec)

        # set walltime, if possible
        self.walltime = None
        if self.ftm.fw_policy.walltime_command:
            try:
                p = subprocess.Popen(self.ftm.fw_policy.walltime_command, shell=True, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out, err =p.communicate()
                status = p.returncode
                if status == 0:
                    self.walltime = int(out)
                else:
                    logger.warning("Impossible to get the walltime: " + err)
            except Exception as e:
                logger.warning("Impossible to get the walltime: ", exc_info=True)

        # read autoparal policy from config if not explicitly set
        if self.is_autoparal is None:
            self.is_autoparal = self.ftm.fw_policy.autoparal

        initialization_info = fw_spec.get('initialization_info', {})

        # if the input is a factory, dynamically create the abinit input. From now on the code will expect an
        # AbinitInput and not a factory. In this case there should be either a single input coming from the previous
        # fws or a deps specifying which input use
        if isinstance(self.abiinput, InputFactory):
            initialization_info['input_factory'] = self.abiinput
            previous_input = None
            if self.abiinput.input_required:
                previous_fws = fw_spec.get('previous_fws', {})
                # check if the input source is specified
                task_type_source = None
                if isinstance(self.deps, dict):
                    try:
                        task_type_source = [tt for tt, deps in self.deps.items() if '@input' in deps][0]
                    except IndexError:
                        pass
                # if not there should be only one previous fw
                if not task_type_source:
                    if len(previous_fws) != 1:
                        msg = 'The input source cannot be identified from depenencies {}. ' \
                              'required by factory {}.'.format(self.deps, self.abiinput.__class__)
                        logger.error(msg)
                        raise InitializationError(msg)
                    task_type_source = previous_fws.keys()[0]
                # the task_type_source should contain just one task and contain the 'input' key
                if len(previous_fws[task_type_source]) != 1 or not previous_fws[task_type_source][0].get('input', None):
                    msg = 'The factory {} requires the input from previous run in the spec'.format(self.abiinput.__class__)
                    logger.error(msg)
                    raise InitializationError(msg)
                # a single input exists
                previous_input = previous_fws[task_type_source][0]['input']
                if not isinstance(previous_input, AbinitInput):
                    previous_input = AbinitInput.from_dict(previous_input)
                initialization_info['previous_input'] = previous_input

            self.abiinput = self.abiinput.build_input(previous_input)

        initialization_info['initial_input'] = self.abiinput

        # if it's the first run log the initialization of the task
        if len(self.history) == 0:
            self.history.log_initialization(self, initialization_info)

        # update data from previous run if it is not a restart
        if 'previous_fws' in fw_spec and not self.restart_info:
            self.load_previous_fws_data(fw_spec)

        self.set_workdir(workdir=os.getcwd())

        # Create dirs for input, output and tmp data.
        self.indir.makedirs()
        self.outdir.makedirs()
        self.tmpdir.makedirs()

        # check if there is a rerun of the FW with info on the exception.
        # if that's the case use the restart information stored there to continue the calculation
        exception_details = fw_spec.get('_exception_details', None)
        if exception_details:
            error_code = exception_details.get('error_code', '')
            if (self.ftm.fw_policy.continue_unconverged_on_rerun and error_code==ErrorCode.UNCONVERGED and
                    exception_details.get('abiinput', None) and exception_details.get('restart_info', None) and
                    exception_details.get('history', None)):
                self.abiinput = AbinitInput.from_dict(exception_details['abiinput'])
                self.restart_info = RestartInfo.from_dict(exception_details['restart_info'])
                self.history = TaskHistory.from_dict(exception_details['history'])

    def run_task(self, fw_spec):
        try:
            self.setup_task(fw_spec)
            if self.is_autoparal:
                return self.autoparal(fw_spec)
            else:
                # loop to allow local restart
                while True:
                    self.config_run(fw_spec)
                    self.run_abinit(fw_spec)
                    action = self.task_analysis(fw_spec)
                    if action:
                        return action
        except BaseException as exc:
            # log the error in history and reraise
            self.history.log_error(exc)
            raise
        finally:
            # Always dump the history for automatic parsing of the folders
            with open(HISTORY_JSON, "w") as f:
                json.dump(self.history, f, cls=MontyEncoder, indent=4, sort_keys=4)

    def restart(self):
        """
        Restart method. Each subclass should implement its own restart. The baseclass should be called
        for common restarting operations
        """
        pass

    def conclude_task(self, fw_spec):
        stored_data = self.report.as_dict()
        stored_data['finalized'] = True
        self.history.log_finalized(self.abiinput)
        stored_data['history'] = self.history.as_dict()
        update_spec = {}
        mod_spec = self.get_final_mod_spec(fw_spec)
        return update_spec, mod_spec, stored_data

    def current_task_info(self, fw_spec):
        return dict(dir=self.workdir, input=self.abiinput)

    def autoparal(self, fw_spec):
        # Copy the appropriate dependencies in the in dir. needed in some cases
        self.resolve_deps(fw_spec)

        optconf, qadapter_spec, qtk_qadapter = self.run_autoparal(self.abiinput, os.path.abspath('.'), self.ftm)
        if self.use_SRC_scheme:
            if 'current_memory_per_proc_mb' in fw_spec and fw_spec['current_memory_per_proc_mb'] is not None:
                qtk_qadapter.set_mem_per_proc(fw_spec['current_memory_per_proc_mb'])
            encoder = MontyEncoder()
            update_spec = None
            if 'previous_fws' in fw_spec:
                update_spec ={'previous_fws': fw_spec['previous_fws']}
            return FWAction(mod_spec={'_set': {'_queueadapter': qtk_qadapter.get_subs_dict(),
                                               'mpi_ncpus': optconf['mpi_ncpus'],
                                               'optconf': optconf, 'qtk_queueadapter': qtk_qadapter}},
                            update_spec=update_spec)
        self.history.log_autoparal(optconf)
        self.abiinput.set_vars(optconf.vars)

        task = self.__class__(**self._get_init_args_and_vals())
        task.is_autoparal = False
        # forward all the specs of the task
        new_spec = {k: v for k, v in fw_spec.items() if k != '_tasks'}
        # set quadapter specification. Note that mpi_ncpus may be different from ntasks
        new_spec['_queueadapter'] = qadapter_spec
        new_spec['mpi_ncpus'] = optconf['mpi_ncpus']
        if 'wf_task_index' in fw_spec:
            split = fw_spec['wf_task_index'].split('_')
            new_spec['wf_task_index'] = '{}_{:d}'.format('_'.join(split[:-1]), 1)
        new_fw = Firework([task], new_spec)

        return FWAction(detours=new_fw)

    def resolve_deps_per_task_type(self, previous_tasks, deps_list):
        for previous_task in previous_tasks:
            for d in deps_list:
                if d.startswith('@structure'):
                    if 'structure' not in previous_task:
                        msg = "previous_fws does not contain the structure."
                        logger.error(msg)
                        raise InitializationError(msg)
                    self.abiinput.set_structure(previous_task['structure'])
                elif not d.startswith('@'):
                    source_dir = previous_task['dir']
                    self.abiinput.set_vars(irdvars_for_ext(d))
                    if d == "DDK":
                        self.link_ddk(source_dir)
                    else:
                        self.link_ext(d, source_dir)

    def resolve_deps(self, fw_spec):
        """
        Method to link the required deps for the current FW.
        Note that different cases are handled here depending whether the current FW is a restart or not and whether
        the rerun is performed in the same folder or not.
        In case of restart the safest choice is to link the deps of the previous FW, so that if they have been
        updated in the meanwhile we are taking the correct one.
        TODO: this last case sounds quite unlikely and should be tested
        """

        # If no deps, nothing to do here
        if not self.deps:
            return

        if not self.restart_info:
            # If this is the first run of the task, the informations are taken from the 'previous_fws',
            # that should be present.
            previous_fws = fw_spec.get('previous_fws', None)
            if previous_fws is None:
                msg = "No previous_fws data. Needed for dependecies {}.".format(str(self.deps))
                logger.error(msg)
                raise InitializationError(msg)

            if isinstance(self.deps, (list, tuple)):
                # check that there is only one previous_fws
                if len(previous_fws) != 1 or len(previous_fws.values()[0]) != 1:
                    msg = "previous_fws does not contain a single reference. " \
                          "Specify the dependency for {}.".format(str(self.deps))
                    logger.error(msg)
                    raise InitializationError(msg)

                self.resolve_deps_per_task_type(previous_fws.values()[0], self.deps)

            else:
                # deps should be a dict
                for task_type, deps_list in self.deps.items():
                    if task_type not in previous_fws:
                        msg = "No previous_fws data for task type {}.".format(task_type)
                        logger.error(msg)
                        raise InitializationError(msg)
                    if len(previous_fws[task_type]) < 1:
                        msg = "Previous_fws does not contain any reference for task type {}, " \
                              "needed in reference {}. ".format(task_type, str(self.deps))
                        logger.error(msg)
                        raise InitializationError(msg)
                    elif len(previous_fws[task_type]) > 1:
                        msg = "Previous_fws contains more than a single reference for task type {}, " \
                              "needed in reference {}. Risk of overwriting.".format(task_type, str(self.deps))
                        logger.warning(msg)

                    self.resolve_deps_per_task_type(previous_fws[task_type], deps_list)

        else:
            # If it is a restart, link the one from the previous task.
            # If it's in the same dir, it is assumed that the dependencies have been corretly resolved in the previous
            # run. So do nothing
            if self.restart_info.previous_dir == self.workdir:
                logger.info('rerunning in the same dir, no action on the deps')
                return

            #just link everything from the indata folder of the previous run. files needed for restart will be overwritten
            prev_indata = os.path.join(self.restart_info.previous_dir, INDIR_NAME)
            for f in os.listdir(prev_indata):
                # if the target is already a link, link to the source to avoid many nested levels of linking
                source = os.path.join(prev_indata, f)
                if os.path.islink(source):
                    source = os.readlink(source)
                os.symlink(source, os.path.join(self.workdir, INDIR_NAME, f))

    def load_previous_fws_data(self, fw_spec):
        pass

    def out_to_in(self, out_file):
        """
        links or copies, according to the fw_policy, the output file to the input data directory of this task
        and rename the file so that ABINIT will read it as an input data file.

        Returns:
            The absolute path of the new file in the indata directory.
        """
        in_file = os.path.basename(out_file).replace("out", "in", 1)
        dest = os.path.join(self.indir.path, in_file)

        if os.path.exists(dest) and not os.path.islink(dest):
            logger.warning("Will overwrite %s with %s" % (dest, out_file))

        # if rerunning in the same folder the file should be moved anyway
        if self.ftm.fw_policy.copy_deps or self.workdir == self.restart_info.previous_dir:
            shutil.copyfile(out_file, dest)
        else:
            # if dest already exists should be overwritten. see also resolve_deps and config_run
            try:
                os.symlink(out_file, dest)
            except OSError, e:
                if e.errno == errno.EEXIST:
                    os.remove(dest)
                    os.symlink(out_file, dest)
                else:
                    raise e

        return dest

    def in_to_in(self, in_file):
        """
        copies the input file to the input of a previous task to the data directory of this task

        Returns:
            The absolute path of the new file in the indata directory.
        """
        dest = os.path.join(self.indir.path, os.path.basename(in_file))

        if os.path.exists(dest) and not os.path.islink(dest):
            logger.warning("Will overwrite %s with %s" % (dest, in_file))

        # os.rename(out_file, dest)
        shutil.copy(in_file, dest)

        return dest

    def remove_restart_vars(self, exts):
        if not isinstance(exts, (list, tuple)):
            exts = [exts]

        remove_vars = [v for e in exts for v in irdvars_for_ext(e).keys()]
        self.abiinput.remove_vars(remove_vars, strict=False)
        logger.info("Removing variables {} from input".format(remove_vars))

##############################
# Specific tasks
##############################


class GsFWTask(AbiFireTask):
    # from GsTask
    @property
    def gsr_path(self):
        """Absolute path of the GSR file. Empty string if file is not present."""
        # Lazy property to avoid multiple calls to has_abiext.
        try:
            return self._gsr_path
        except AttributeError:
            path = self.outdir.has_abiext("GSR")
            if path: self._gsr_path = path
            return path

    def open_gsr(self):
        """
        Open the GSR file located in the in self.outdir.
        Returns :class:`GsrFile` object, raise a PostProcessError exception if file could not be found or file is not readable.
        """
        gsr_path = self.gsr_path
        if not gsr_path:
            msg = "No GSR file available for task {} in {}".format(self, self.outdir)
            logger.critical(msg)
            raise PostProcessError(msg)

        # Open the GSR file.
        from abipy.electrons.gsr import GsrFile
        try:
            return GsrFile(gsr_path)
        except Exception as exc:
            msg = "Exception while reading GSR file at %s:\n%s" % (gsr_path, str(exc))
            logger.critical(msg)
            raise PostProcessError(msg)


@explicit_serialize
class ScfFWTask(GsFWTask):
    task_type = "scf"

    CRITICAL_EVENTS = [
        events.ScfConvergenceWarning,
    ]

    def restart(self):
        """SCF calculations can be restarted if we have either the WFK file or the DEN file."""
        # Prefer WFK over DEN files since we can reuse the wavefunctions.
        if self.restart_info.reset:
            # remove non reset keys that may have been added in a previous restart
            self.remove_restart_vars(["WFK", "DEN"])
        else:
            for ext in ("WFK", "DEN"):
                restart_file = self.restart_info.prev_outdir.has_abiext(ext)
                irdvars = irdvars_for_ext(ext)
                if restart_file: break
            else:
                msg = "Cannot find WFK or DEN file to restart from."
                logger.error(msg)
                raise RestartError(msg)

            # Move out --> in.
            self.out_to_in(restart_file)

            # Add the appropriate variable for restarting.
            self.abiinput.set_vars(irdvars)


@explicit_serialize
class NscfFWTask(GsFWTask):
    task_type = "nscf"

    CRITICAL_EVENTS = [
        events.NscfConvergenceWarning,
    ]

    def restart(self):
        """NSCF calculations can be restarted only if we have the WFK file."""
        if self.restart_info.reset:
            # remove non reset keys that may have been added in a previous restart
            self.remove_restart_vars(["WFK"])
        else:
            ext = "WFK"
            restart_file = self.restart_info.prev_outdir.has_abiext(ext)
            if not restart_file:
                msg = "Cannot find the WFK file to restart from."
                logger.error(msg)
                raise AbinitRuntimeError(msg)

            # Move out --> in.
            self.out_to_in(restart_file)

            # Add the appropriate variable for restarting.
            irdvars = irdvars_for_ext(ext)
            self.abiinput.set_vars(irdvars)


@explicit_serialize
class RelaxFWTask(GsFWTask):
    task_type = "relax"

    CRITICAL_EVENTS = [
        events.RelaxConvergenceWarning,
    ]

    def get_final_structure(self):
        """Read the final structure from the GSR file."""
        try:
            with self.open_gsr() as gsr:
                return gsr.structure
        except AttributeError:
            msg = "Cannot find the GSR file with the final structure to restart from."
            logger.error(msg)
            raise PostProcessError(msg)

    def prepare_restart(self, fw_spec, reset=False):
        self.abiinput.set_structure(self.get_final_structure())

        return super(RelaxFWTask, self).prepare_restart(fw_spec, reset)

    def restart(self):
        """
        Restart the structural relaxation.

        See original RelaxTask for more details
        """

        if self.restart_info.reset:
            # remove non reset keys that may have been added in a previous restart
            self.remove_restart_vars(["WFK", "DEN"])
        else:
            # for optcell > 0 it may fail to restart if paral_kgb == 0. Do not use DEN or WFK in this case
            #FIXME fix when Matteo makes the restart possible for paral_kgb == 0
            paral_kgb = self.abiinput.get('paral_kgb', 0)
            optcell = self.abiinput.get('optcell', 0)

            if optcell == 0 or paral_kgb == 1:
                restart_file = None

                # Try to restart from the WFK file if possible.
                # FIXME: This part has been disabled because WFK=IO is a mess if paral_kgb == 1
                # This is also the reason why I wrote my own MPI-IO code for the GW part!
                wfk_file = self.restart_info.prev_outdir.has_abiext("WFK")
                if False and wfk_file:
                    irdvars = irdvars_for_ext("WFK")
                    restart_file = self.out_to_in(wfk_file)

                # Fallback to DEN file. Note that here we look for out_DEN instead of out_TIM?_DEN
                # This happens when the previous run completed and task.on_done has been performed.
                # ********************************************************************************
                # Note that it's possible to have an undected error if we have multiple restarts
                # and the last relax died badly. In this case indeed out_DEN is the file produced
                # by the last run that has executed on_done.
                # ********************************************************************************
                if restart_file is None:
                    out_den = self.restart_info.prev_outdir.path_in("out_DEN")
                    if os.path.exists(out_den):
                        irdvars = irdvars_for_ext("DEN")
                        restart_file = self.out_to_in(out_den)

                if restart_file is None:
                    # Try to restart from the last TIM?_DEN file.
                    # This should happen if the previous run didn't complete in clean way.
                    # Find the last TIM?_DEN file.
                    last_timden = self.restart_info.prev_outdir.find_last_timden_file()
                    if last_timden is not None:
                        ofile = self.restart_info.prev_outdir.path_in("out_DEN")
                        os.rename(last_timden.path, ofile)
                        restart_file = self.out_to_in(ofile)
                        irdvars = irdvars_for_ext("DEN")

                if restart_file is None:
                    # Don't raise RestartError as the structure has been updated
                    logger.warning("Cannot find the WFK|DEN|TIM?_DEN file to restart from.")
                else:
                    # Add the appropriate variable for restarting.
                    self.abiinput.set_vars(irdvars)
                    logger.info("Will restart from %s", restart_file)

    def current_task_info(self, fw_spec):
        d = super(RelaxFWTask, self).current_task_info(fw_spec)
        d['structure'] = self.get_final_structure()
        return d

    # def conclude_task(self, fw_spec):
    #     update_spec, mod_spec, stored_data = super(RelaxFWTask, self).conclude_task(fw_spec)
    #     update_spec['previous_run']['structure'] = self.get_final_structure()
    #     return update_spec, mod_spec, stored_data


@explicit_serialize
class HybridFWTask(GsFWTask):
    task_type = "hybrid"

    CRITICAL_EVENTS = [
    ]


@explicit_serialize
class DfptTask(AbiFireTask):
    """

    """
    CRITICAL_EVENTS = [
        events.ScfConvergenceWarning,
    ]
    task_type = "dfpt"

    def restart(self):
        """
        Phonon calculations can be restarted only if we have the 1WF file or the 1DEN file.
        from which we can read the first-order wavefunctions or the first order density.
        Prefer 1WF over 1DEN since we can reuse the wavefunctions.
        Try to handle an input with many perturbation calculated at the same time. link/copy all the 1WF or 1DEN files
        """
        # Abinit adds the idir-ipert index at the end of the file and this breaks the extension
        # e.g. out_1WF4, out_DEN4. find_1wf_files and find_1den_files returns the list of files found
        #TODO check for reset
        restart_files, irdvars = None, None

        # Highest priority to the 1WF file because restart is more efficient.
        wf_files = self.restart_info.prev_outdir.find_1wf_files()
        if wf_files is not None:
            restart_files = [f.path for f in wf_files]
            irdvars = irdvars_for_ext("1WF")
            # if len(wf_files) != 1:
            #     restart_files = None
            #     logger.critical("Found more than one 1WF file. Restart is ambiguous!")

        if restart_files is None:
            den_files = self.restart_info.prev_outdir.find_1den_files()
            if den_files is not None:
                restart_files = [f.path for f in den_files]
                irdvars = {"ird1den": 1}
                # if len(den_files) != 1:
                #     restart_files = None
                #     logger.critical("Found more than one 1DEN file. Restart is ambiguous!")

        if not restart_files:
            # Raise because otherwise restart is equivalent to a run from scratch --> infinite loop!
            msg = "Cannot find the 1WF|1DEN file to restart from."
            logger.error(msg)
            raise RestartError(msg)

        # Move file.
        for f in restart_files:
            self.out_to_in(f)

        # Add the appropriate variable for restarting.
        self.abiinput.set_vars(irdvars)


@explicit_serialize
class DdkTask(AbiFireTask):
    task_type = "ddk"

    def conclude_task(self, fw_spec):
        # make a link to _DDK of the 1WF file to ease the link in the dependencies
        wf_files = self.outdir.find_1wf_files()
        if not wf_files:
            raise PostProcessError("Couldn't link 1WF files.")
        for f in wf_files:
            os.symlink(f.path, f.path+'_DDK')

        return super(DdkTask, self).conclude_task(fw_spec)


# @explicit_serialize
# class Ddk1WFTask(AbiFireTask):
#     task_type = "ddk-1wf"
#
#     def conclude_task(self, fw_spec):
#         # # make the links
#         # natom = len(self.abiinput.structure)
#         # for ii in range(3*natom+1, 3*natom+4):
#         #     wf_file = self.outdir.has_abiext('1WF{:d}'.format(ii))
#         #     if not wf_file:
#         #         raise PostProcessError(self, "Couldn't link the 1WF files.")
#         #     os.symlink(wf_file, wf_file+'_DDK')
#
#         return super(Ddk1WFTask, self).conclude_task(fw_spec)


@explicit_serialize
class DdeTask(DfptTask):
    task_type = "dde"


@explicit_serialize
class PhononTask(DfptTask):
    task_type = "phonon"


@explicit_serialize
class BecTask(DfptTask):
    task_type = "bec"


@explicit_serialize
class StrainPertTask(DfptTask):
    task_type = "strain_pert"


##############################
# Convergence tasks
##############################

@explicit_serialize
class RelaxDilatmxFWTask(RelaxFWTask):
    def __init__(self, abiinput, restart_info=None, handlers=[], is_autoparal=None, deps=None, history=[],
                 target_dilatmx=1.01):
        self.target_dilatmx = target_dilatmx
        super(RelaxDilatmxFWTask, self).__init__(abiinput=abiinput, restart_info=restart_info, handlers=handlers,
                                                 is_autoparal=is_autoparal, deps=deps, history=history)

    def check_parameters_convergence(self, fw_spec):
        actual_dilatmx = self.abiinput.get('dilatmx', 1.)
        new_dilatmx = actual_dilatmx - min((actual_dilatmx-self.target_dilatmx), actual_dilatmx*0.03)
        #FIXME reset can be False with paral_kgb==1
        return {'dilatmx': new_dilatmx} if new_dilatmx != actual_dilatmx else {}, True


##############################
# Wrapper tasks
##############################


@explicit_serialize
class MergeDdbTask(BasicTaskMixin, FireTaskBase):
    task_type = "mrgddb"

    #TODO: make it possible to use "any" task and in particular, this MergeDdbTask for the SRC
    # scheme (to be rationalized)
    def __init__(self, ddb_source_task_types=None, delete_source_ddbs=True, num_ddbs=None):
        """
        ddb_source_task_type: list of task types that will be used as source for the DDB to be merged.
        The default is [PhononTask.task_type, DdeTask.task_type, BecTask.task_type]
        delete_ddbs: delete the ddb files used after the merge
        num_ddbs: number of ddbs to be merged. If set will be used to check that the correct number of ddbs have been
         passed to the task. Tha task will fizzle if the numbers do not match
        """

        if ddb_source_task_types is None:
            ddb_source_task_types = [PhononTask.task_type, DdeTask.task_type, BecTask.task_type]
        elif not isinstance(ddb_source_task_types, (list, tuple)):
            ddb_source_task_types = [ddb_source_task_types]

        self.ddb_source_task_types = ddb_source_task_types
        self.delete_source_ddbs = delete_source_ddbs
        self.num_ddbs = num_ddbs

    def get_ddb_list(self, previous_fws, task_type):
        ddb_files = []
        for t in previous_fws.get(task_type, []):
            ddb = Directory(os.path.join(t['dir'], OUTDIR_NAME)).has_abiext('DDB')
            if not ddb:
                msg = "One of the task of type {} (folder: {}) " \
                      "did not produce a DDB file!".format(task_type, t['dir'])
                raise InitializationError(msg)
            ddb_files.append(ddb)
        return ddb_files

    def get_event_report(self, ofile_name="mrgddb.stdout"):
        ofile = File(os.path.join(self.workdir, ofile_name))
        parser = events.EventsParser()

        if not ofile.exists:
            return None
        else:
            try:
                report = parser.parse(ofile.path)
                return report
            except Exception as exc:
                # Return a report with an error entry with info on the exception.
                logger.critical("{}: Exception while parsing MRGDDB events:\n {}".format(ofile, str(exc)))
                return parser.report_exception(ofile.path, exc)

    def set_workdir(self, workdir):
        self.workdir = workdir
        self.outdir = Directory(os.path.join(self.workdir, OUTDIR_NAME))

    def run_task(self, fw_spec):
        self.set_workdir(workdir=os.getcwd())
        self.outdir.makedirs()
        self.history = TaskHistory()
        try:
            ftm = self.get_fw_task_manager(fw_spec)
            if not ftm.has_task_manager():
                raise InitializationError("No task manager available: mrgddb could not be performed.")
            mrgddb = Mrgddb(manager=ftm.task_manager, executable=ftm.fw_policy.mrgddb_cmd, verbose=0)

            previous_fws = fw_spec['previous_fws']
            ddb_files = []
            for source_task_type in self.ddb_source_task_types:
                ddb_files.extend(self.get_ddb_list(previous_fws, source_task_type))

            initialization_info = fw_spec.get('initialization_info', {})
            initialization_info['ddb_files_list'] = ddb_files
            self.history.log_initialization(self, initialization_info)

            if not ddb_files:
                raise InitializationError("No DDB files to merge.")

            if self.num_ddbs is not None and self.num_ddbs != len(ddb_files):
                raise InitializationError("Wrong number of DDB files: {} DDB files have been requested, "
                                          "but {} have been linked".format(self.num_ddbs, len(ddb_files)))

            # keep the output in the outdata dir for consistency
            out_ddb = os.path.join(self.workdir, OUTDIR_NAME, "out_DDB")
            desc = "DDB file merged by %s on %s" % (self.__class__.__name__, time.asctime())

            out_ddb = mrgddb.merge(self.workdir, ddb_files, out_ddb=out_ddb, description=desc,
                                   delete_source_ddbs=self.delete_source_ddbs)

            self.report = self.get_event_report()

            if not os.path.isfile(out_ddb) or (self.report and self.report.errors):
                raise AbinitRuntimeError(self, msg="Error during mrgddb.")

            stored_data = dict(finalized=True)
            mod_spec = self.get_final_mod_spec(fw_spec)

            self.history.log_finalized()

            return FWAction(stored_data=stored_data, mod_spec=mod_spec)

        except BaseException as exc:
            # log the error in history and reraise
            self.history.log_error(exc)
            raise
        finally:
            with open(HISTORY_JSON, "w") as f:
                json.dump(self.history, f, cls=MontyEncoder, indent=4, sort_keys=4)


    def current_task_info(self, fw_spec):
        return dict(dir=self.workdir)

    @property
    def merged_ddb_path(self):
        """Absolute path of the merged DDB file. Empty string if file is not present."""
        # Lazy property to avoid multiple calls to has_abiext.
        try:
            return self._merged_ddb_path
        except AttributeError:
            path = self.outdir.has_abiext("DDB")
            if path: self._merged_ddb_path = path
            return path


@explicit_serialize
class AnaDdbTask(BasicTaskMixin, FireTaskBase):
    task_type = "anaddb"

    def __init__(self, anaddb_input, restart_info=None, handlers=[], is_autoparal=None, deps=None, history=[],
                 use_SRC_scheme=False, task_type=None):
        self.anaddb_input = anaddb_input
        self.restart_info = restart_info

        self.handlers = handlers or [cls() for cls in events.get_event_handler_classes()]
        self.is_autoparal = is_autoparal

        self.use_SRC_scheme = use_SRC_scheme

        #TODO: rationalize this and check whether this might create problems due to the fact that if task_type is None,
        #      self.task_type is the class variable (actually self.task_type refers to self.__class__.task_type) while
        #      if task_type is specified, self.task_type is an instance variable and is potentially different from
        #      self.__class__.task_type !
        if task_type is not None:
            self.task_type = task_type

        # deps are transformed to be a list or a dict of lists
        if isinstance(deps, dict):
            deps = dict(deps)
            for k, v in deps.items():
                if not isinstance(v, (list, tuple)):
                    deps[k] = [v]
        elif deps and not isinstance(deps, (list, tuple)):
            deps = [deps]
        self.deps = deps

        self.history = TaskHistory(history)

    @property
    def ec_path(self):
        """Absolute path of the GSR file. Empty string if file is not present."""
        path = self.rundir.has_abiext("EC")
        if path: self._ec_path = path
        return path

    def get_elastic_tensor(self, tensor_type='relaxed_ion'):
        """
        Open the EC file located in the in self.workdir.
        Returns :class:`ElasticConstant` object, None if file could not be found or file is not readable.
        """
        ec_path = self.ec_path
        if not ec_path:
            msg = "{} reached the conclusion but didn't produce a EC file in {}".format(self, self.workdir)
            logger.critical(msg)
            raise PostProcessError(msg)
        ec = ElasticComplianceTensor.from_ec_nc_file(ec_path, tensor_type=tensor_type)
        return ec

    def get_ddb_list(self, previous_fws, task_type):
        ddb_files = []
        for t in previous_fws.get(task_type, []):
            ddb = Directory(os.path.join(t['dir'], OUTDIR_NAME)).has_abiext('DDB')
            if not ddb:
                msg = "One of the task of type {} (folder: {}) " \
                      "did not produce a DDB file!".format(task_type, t['dir'])
    def resolve_deps_per_task_type(self, previous_tasks, deps_list):
        for previous_task in previous_tasks:
            for d in deps_list:
                source_dir = previous_task['dir']
                if d == "DDB":
                    self.ddb_filepath = self.link_ext(d, source_dir)
                elif d == "GKK":
                    self.gkk_filepath = self.link_ext(d, source_dir)
                elif d == "DDK":
                    self.ddk_filepaths.extend(self.link_ddk(source_dir))
                else:
                    logger.warning("Extensions {} is not used in anaddb and will be ignored".format(d))
                    continue

    def resolve_deps(self, fw_spec):
        #FIXME extract common method with AbinitTask
        previous_fws = fw_spec.get('previous_fws', None)
        if previous_fws is None:
            msg = "No previous_fws data. Needed for dependecies {}.".format(str(self.deps))
            logger.error(msg)
            raise InitializationError(msg)

        if isinstance(self.deps, (list, tuple)):
            # check that there is only one previous_fws
            if len(previous_fws) != 1 or len(previous_fws.values()[0]) != 1:
                msg = "previous_fws does not contain a single reference. " \
                      "Specify the dependency for {}.".format(str(self.deps))
                logger.error(msg)
                raise InitializationError(msg)

            self.resolve_deps_per_task_type(previous_fws.values()[0], self.deps)
        else:
            # deps should be a dict
            for task_type, deps_list in self.deps.items():
                if task_type not in previous_fws:
                    msg = "No previous_fws data for task type {}.".format(task_type)
                    logger.error(msg)
                    raise InitializationError(msg)
                if len(previous_fws[task_type]) < 1:
                    msg = "Previous_fws does not contain any reference for task type {}, " \
                          "needed in reference {}. ".format(task_type, str(self.deps))
                    logger.error(msg)
                    raise InitializationError(msg)
                elif len(previous_fws[task_type]) > 1:
                    msg = "Previous_fws contains more than a single reference for task type {}, " \
                          "needed in reference {}. Risk of overwriting.".format(task_type, str(self.deps))
                    logger.warning(msg)

                self.resolve_deps_per_task_type(previous_fws[task_type], deps_list)

    def run_anaddb(self, fw_spec):
        """
        executes anaddb and waits for the end of the process.
        TODO: make it general in the same way as "run_abinit"
        the mpirun command is retrived from the mpirun_cmd keys in the fw_polity
        of the FWTaskManager, that can be overridden by the values in the spec.
        Note that in case of missing definition of this parameter, the values fall back to the default
        value of  mpirun_cmd: 'mpirun', assuming that it is properly retrived
        from the PATH. By default, anaddb is retrieved from the PATH.
        """

        def anaddb_process():
            command = []
            #consider the case of serial execution
            if self.ftm.fw_policy.mpirun_cmd:
                command.append(self.ftm.fw_policy.mpirun_cmd)
                if 'mpi_ncpus' in fw_spec:
                    command.extend(['-np', str(fw_spec['mpi_ncpus'])])
            command.append(self.ftm.fw_policy.anaddb_cmd)
            with open(self.files_file.path, 'r') as stdin, open(self.log_file.path, 'w') as stdout, \
                    open(self.stderr_file.path, 'w') as stderr:
                self.process = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=stderr)

            (stdoutdata, stderrdata) = self.process.communicate()
            self.returncode = self.process.returncode

        # initialize returncode to avoid missing references in case of exception in the other thread
        self.returncode = None

        thread = threading.Thread(target=anaddb_process)
        # the amount of time left plus a buffer of 2 minutes
        timeout = (self.walltime - (time.time() - self.start_time) - 120) if self.walltime else None
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            self.process.terminate()
            thread.join()
            raise WalltimeError("The task couldn't be terminated within the time limit. Killed.")

    def setup_task(self, fw_spec):
        self.start_time = time.time()

        self.set_logger()

        # load the FWTaskManager to get configuration parameters
        self.ftm = self.get_fw_task_manager(fw_spec)

        # set walltime, if possible
        self.walltime = None
        if self.ftm.fw_policy.walltime_command:
            try:
                p = subprocess.Popen(self.ftm.fw_policy.walltime_command, shell=True, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out, err =p.communicate()
                status = p.returncode
                if status == 0:
                    self.walltime = int(out)
                else:
                    logger.warning("Impossible to get the walltime: " + err)
            except Exception as e:
                logger.warning("Impossible to get the walltime: ", exc_info=True)

        # setting working directory and files
        self.set_workdir(os.getcwd())

        self.indir.makedirs()
        self.outdir.makedirs()
        self.tmpdir.makedirs()

        self.ddb_filepath = None
        self.gkk_filepath = None
        self.ddk_filepaths = []

        self.resolve_deps(fw_spec)

        # the DDB file is needed. If not set as a dependency, look for it in all the possible sources
        #FIXME check if we can remove this case and just rely on deps
        if not self.ddb_filepath:
            previous_fws = fw_spec['previous_fws']
            for task_class in DfptTask.__subclasses__() + [DfptTask]:
                task_type = task_class.task_type
                ddb_list = []
                for previous_task in previous_fws.get(task_type, []):
                    ddb_list.append(self.link_ext("DDB", previous_task['dir'], strict=False))

            if len(ddb_list) != 1:
                raise InitializationError("Cannot find a single DDB to run...")

            self.ddb_filepath = ddb_list[0]

        if self.ddk_filepaths:
            self.ddk_files_file.write("\n".join(self.ddk_filepaths))

        # Write files file and input file.
        if not self.files_file.exists:
            self.files_file.write(self.filesfile_string)

        self.input_file.write(str(self.anaddb_input))

    def run_task(self, fw_spec):
        try:
            self.setup_task(fw_spec)
            self.run_anaddb(fw_spec)
            action = self.task_analysis(fw_spec)
        except BaseException as exc:
            # log the error in history and reraise
            self.history.log_error(exc)
            raise
        finally:
            # Always dump the history for automatic parsing of the folders
            with open(HISTORY_JSON, "w") as f:
                json.dump(self.history, f, cls=MontyEncoder, indent=4, sort_keys=4)

    def task_analysis(self, fw_spec):
        return FWAction()

    def set_workdir(self, workdir):
        """Set the working directory."""

        self.workdir = os.path.abspath(workdir)

        # Files required for the execution.
        self.input_file = File(os.path.join(self.workdir, INPUT_FILE_NAME))
        self.output_file = File(os.path.join(self.workdir, OUTPUT_FILE_NAME))
        self.files_file = File(os.path.join(self.workdir, FILES_FILE_NAME))
        self.log_file = File(os.path.join(self.workdir, LOG_FILE_NAME))
        self.stderr_file = File(os.path.join(self.workdir, STDERR_FILE_NAME))
        self.elphon_out_file = File(os.path.join(self.workdir, ELPHON_OUTPUT_FILE_NAME))
        self.ddk_files_file = File(os.path.join(self.workdir, DDK_FILES_FILE_NAME))

        # This file is produce by Abinit if nprocs > 1 and MPI_ABORT.
        self.mpiabort_file = File(os.path.join(self.workdir, MPIABORTFILE))

        # Directories with input|output|temporary data.
        self.rundir = Directory(self.workdir)
        self.indir = Directory(os.path.join(self.workdir, INDIR_NAME))
        self.outdir = Directory(os.path.join(self.workdir, OUTDIR_NAME))
        self.tmpdir = Directory(os.path.join(self.workdir, TMPDIR_NAME))

    @property
    def filesfile_string(self):
        """String with the list of files and prefixes needed to execute ABINIT."""
        lines = []
        app = lines.append

        app(self.input_file.path)                     # 1) Path of the input file
        app(self.output_file.path)                    # 2) Path of the output file
        app(self.ddb_filepath)                        # 3) Input derivative database e.g. t13.ddb.in
        app(DUMMY_FILENAME)                           # 4) Ignored
        app(self.gkk_filepath or DUMMY_FILENAME)      # 5) Input elphon matrix elements  (GKK file)
        app(self.elphon_out_file.path)                     # 6) Base name for elphon output files e.g. t13
        app(self.ddk_files_file if self.ddk_filepaths
            else DUMMY_FILENAME)                      # 7) File containing ddk filenames for elphon/transport.

        return "\n".join(lines)

    @property
    def phbst_path(self):
        """Absolute path of the merged DDB file. Empty string if file is not present."""
        # Lazy property to avoid multiple calls to has_abiext.
        try:
            return self._phbst_path
        except AttributeError:
            path = os.path.join(self.workdir, "run.abo_PHBST.nc")
            if path: self._phbst_path = path
            return path

    def open_phbst(self):
        """
        Open PHBST file produced by Anaddb and returns :class:`PhbstFile` object.
        Raise a PostProcessError exception if file could not be found or file is not readable.
        """
        from abipy.dfpt.phonons import PhbstFile
        if not self.phbst_path:
            msg = "No PHBST file available for task {} in {}".format(self, self.outdir)
            logger.critical(msg)
            raise PostProcessError(msg)

        try:
            return PhbstFile(self.phbst_path)
        except Exception as exc:
            msg = "Exception while reading GSR file at %s:\n%s" % (self.phbst_path, str(exc))
            logger.critical(msg)
            return PostProcessError(msg)


##############################
# Generation tasks
##############################

@explicit_serialize
class GeneratePhononFlowFWTask(BasicTaskMixin, FireTaskBase):
    def __init__(self, phonon_factory, previous_task_type=ScfFWTask.task_type, handlers=[], with_autoparal=None, ddb_file=None):
        self.phonon_factory = phonon_factory
        self.previous_task_type = previous_task_type
        self.handlers = handlers
        self.with_autoparal=with_autoparal
        self.ddb_file = ddb_file

    def get_fws(self, multi_inp, task_class, deps, new_spec, ftm, nscf_fws=None):
        formula = multi_inp[0].structure.composition.reduced_formula
        fws = []
        fw_deps = defaultdict(list)
        autoparal_spec = {}
        for i, inp in enumerate(multi_inp):
            new_spec = dict(new_spec)
            start_task_index = 1
            if self.with_autoparal:
                if not autoparal_spec:
                    autoparal_dir = os.path.join(os.path.abspath('.'), "autoparal_{}_{}".format(task_class.__name__, str(i)))
                    optconf, qadapter_spec, qadapter = self.run_autoparal(inp, autoparal_dir, ftm)
                    autoparal_spec['_queueadapter'] = qadapter_spec
                    autoparal_spec['mpi_ncpus'] = optconf['mpi_ncpus']
                new_spec.update(autoparal_spec)

            current_deps = dict(deps)
            parent_fw = None
            if nscf_fws:
                qpt = inp['qpt']
                for nscf_fw in nscf_fws:
                    if np.allclose(nscf_fw.tasks[0].abiinput['qpt'], qpt):
                        parent_fw = nscf_fw
                        current_deps[nscf_fw.tasks[0].task_type] = "WFQ"
                        break


            task = task_class(inp, handlers=self.handlers, deps=current_deps, is_autoparal=False)
            # this index is for the different task, each performing a different perturbation
            indexed_task_type = task_class.task_type + '_' + str(i)
            # this index is to index the restarts of the single task
            new_spec['wf_task_index'] = indexed_task_type + '_' + str(start_task_index)
            fw = Firework(task, spec=new_spec, name=(formula + '_' + indexed_task_type)[:15])
            fws.append(fw)
            if parent_fw is not None:
                fw_deps[parent_fw].append(fw)

        return fws, fw_deps

    def run_task(self, fw_spec):
        previous_input = fw_spec.get('previous_fws', {}).get(self.previous_task_type, [{}])[0].get('input', None)
        if not previous_input:
            raise InitializationError('No input file available from task of type {}'.format(self.previous_task_type))

        previous_input = AbinitInput.from_dict(previous_input)

        if self.with_autoparal is None:
            self.with_autoparal = ftm.fw_policy.autoparal

        ftm = self.get_fw_task_manager(fw_spec)
        if self.with_autoparal:
            if not ftm.has_task_manager():
                msg = 'No task manager available: autoparal could not be performed.'
                logger.error(msg)
                raise InitializationError(msg)

            # inject task manager
            tasks._USER_CONFIG_TASKMANAGER = ftm.task_manager

        ph_inputs = self.phonon_factory.build_input(previous_input)

        initialization_info = fw_spec.get('initialization_info', {})
        initialization_info['input_factory'] = self.phonon_factory.as_dict()
        new_spec = dict(previous_fws=fw_spec['previous_fws'], initialization_info=initialization_info,
                        _preserve_fworker=True)
        if '_fworker' in fw_spec:
            new_spec['_fworker'] = fw_spec['_fworker']

        ph_q_pert_inputs = ph_inputs.filter_by_tags(PH_Q_PERT)
        ddk_inputs = ph_inputs.filter_by_tags(DDK)
        dde_inputs = ph_inputs.filter_by_tags(DDE)
        bec_inputs = ph_inputs.filter_by_tags(BEC)

        nscf_inputs = ph_inputs.filter_by_tags(NSCF)

        nscf_fws = []
        if nscf_inputs is not None:
            nscf_fws, nscf_fw_deps= self.get_fws(nscf_inputs, NscfFWTask,
                                                 {self.previous_task_type: "WFK", self.previous_task_type: "DEN"}, new_spec, ftm)

        ph_fws = []
        if ph_q_pert_inputs:
            ph_q_pert_inputs.set_vars(prtwf=-1)
            ph_fws, ph_fw_deps = self.get_fws(ph_q_pert_inputs, PhononTask, {self.previous_task_type: "WFK"}, new_spec,
                                              ftm, nscf_fws)

        ddk_fws = []
        if ddk_inputs:
            ddk_fws, ddk_fw_deps = self.get_fws(ddk_inputs, DdkTask, {self.previous_task_type: "WFK"}, new_spec, ftm)

        dde_fws = []
        if dde_inputs:
            dde_inputs.set_vars(prtwf=-1)
            dde_fws, dde_fw_deps = self.get_fws(dde_inputs, DdeTask,
                                                {self.previous_task_type: "WFK", DdkTask.task_type: "DDK"}, new_spec, ftm)

        bec_fws = []
        if bec_inputs:
            bec_inputs.set_vars(prtwf=-1)
            bec_fws, bec_fw_deps = self.get_fws(bec_inputs, BecTask,
                                                {self.previous_task_type: "WFK", DdkTask.task_type: "DDK"}, new_spec, ftm)


        mrgddb_spec = dict(new_spec)
        mrgddb_spec['wf_task_index'] = 'mrgddb'
        #FIXME import here to avoid circular imports.
        from abiflows.fireworks.utils.fw_utils import get_short_single_core_spec
        qadapter_spec = get_short_single_core_spec(ftm)
        mrgddb_spec['mpi_ncpus'] = 1
        mrgddb_spec['_queueadapter'] = qadapter_spec
        # Set a higher priority to favour the end of the WF
        #TODO improve the handling of the priorities
        mrgddb_spec['_priority'] = 10
        num_ddbs_to_be_merged = len(ph_fws) + len(dde_fws) + len(bec_fws)
        mrgddb_fw = Firework(MergeDdbTask(num_ddbs=num_ddbs_to_be_merged, delete_source_ddbs=False), spec=mrgddb_spec,
                             name=ph_inputs[0].structure.composition.reduced_formula+'_mergeddb')

        fws_deps = {}

        if ddk_fws:
            for ddk_fw in ddk_fws:
                if dde_fws:
                    fws_deps[ddk_fw] = dde_fws
                if bec_fws:
                    fws_deps[ddk_fw] = bec_fws

        ddb_fws = dde_fws + ph_fws + bec_fws
        #TODO pass all the tasks to the MergeDdbTask for logging or easier retrieve of the DDK?
        for ddb_fw in ddb_fws:
            fws_deps[ddb_fw] = mrgddb_fw

        total_list_fws = ddb_fws+ddk_fws+[mrgddb_fw] + nscf_fws

        fws_deps.update(ph_fw_deps)

        ph_wf = Workflow(total_list_fws, fws_deps)

        stored_data = dict(finalized=True)

        return FWAction(stored_data=stored_data, detours=ph_wf)

##############################
# Exceptions
##############################

class ErrorCode(object):
    ERROR = 'Error'
    UNRECOVERABLE = 'Unrecoverable'
    UNCLASSIFIED = 'Unclassified'
    UNCONVERGED = 'Unconverged'
    UNCONVERGED_PARAMETERS = 'Unconverged_parameters'
    INITIALIZATION = 'Initialization'
    RESTART = 'Restart'
    POSTPROCESS = 'Postprocess'
    WALLTIME = 'Walltime'


class AbiFWError(Exception):

    def __init__(self, msg):
        super(AbiFWError, self).__init__(msg)
        self.msg = msg

    def to_dict(self):
        return dict(error_code=self.ERROR_CODE, msg=self.msg)


class AbinitRuntimeError(AbiFWError):
    """
    Exception raised for errors during Abinit calculation
    Initialized with a task, uses it to prepare a suitable error message
    """
    ERROR_CODE = ErrorCode.ERROR

    def __init__(self, task, msg=None):
        super(AbinitRuntimeError, self).__init__(msg)
        self.task = task
        self.msg = msg

    @pmg_serialize
    def to_dict(self):
        report = self.task.report
        d = {}
        d['num_errors'] = report.num_errors
        d['num_warnings'] = report.num_warnings
        if report.num_errors:
            errors = []
            for error in report.errors:
                errors.append(error.as_dict())
            d['errors'] = errors
        if report.num_warnings:
            warnings = []
            for warning in report.warnings:
                warnings.append(warning.as_dict())
            d['warnings'] = warnings
        if self.msg:
            d['error_message'] = self.msg

        d['error_code'] = self.ERROR_CODE

        return d


class UnconvergedError(AbinitRuntimeError):
    ERROR_CODE = ErrorCode.UNCONVERGED

    def __init__(self, task, msg=None, abiinput=None, restart_info=None, history=None):
        super(UnconvergedError, self).__init__(task, msg)
        self.abiinput = abiinput
        self.restart_info = restart_info
        self.history = history

    def to_dict(self):
        d = super(UnconvergedError, self).to_dict()
        d['abiinput'] = self.abiinput.as_dict() if self.abiinput else None
        d['restart_info'] = self.restart_info.as_dict() if self.restart_info else None
        d['history'] = self.history.as_dict() if self.history else None
        return d


class UnconvergedParametersError(UnconvergedError):
    ERROR_CODE = ErrorCode.UNCONVERGED_PARAMETERS


class WalltimeError(AbiFWError):
    ERROR_CODE = ErrorCode.WALLTIME


class InitializationError(AbiFWError):
    ERROR_CODE = ErrorCode.INITIALIZATION


class RestartError(InitializationError):
    ERROR_CODE = ErrorCode.RESTART


class PostProcessError(AbiFWError):
    ERROR_CODE = ErrorCode.POSTPROCESS

##############################
# Other objects
##############################


class RestartInfo(MSONable):
    """
    Object that contains the information about the restart of a task.
    """
    def __init__(self, previous_dir, reset=False, num_restarts=0):
        self.previous_dir = previous_dir
        self.reset = reset
        self.num_restarts = num_restarts

    @pmg_serialize
    def as_dict(self):
        return dict(previous_dir=self.previous_dir, reset=self.reset, num_restarts=self.num_restarts)

    @classmethod
    def from_dict(cls, d):
        return cls(previous_dir=d['previous_dir'], reset=d['reset'], num_restarts=d['num_restarts'])

    @property
    def prev_outdir(self):
        return Directory(os.path.join(self.previous_dir, OUTDIR_NAME))

    @property
    def prev_indir(self):
        return Directory(os.path.join(self.previous_dir, INDIR_NAME))
