# coding: utf-8
"""
Abinit Task classes for Fireworks.
"""
from __future__ import print_function, division, unicode_literals, absolute_import

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

from collections import namedtuple, defaultdict
from monty.json import MontyEncoder, MontyDecoder, MSONable
from pymatgen.util.serialization import json_pretty_dump, pmg_serialize
from pymatgen.analysis.elasticity import ElasticTensor
from abipy.flowtk.utils import Directory, File
from abipy.flowtk import events, tasks
from abipy.flowtk.netcdf import NetcdfReader, NO_DEFAULT
from abipy.flowtk.utils import irdvars_for_ext
from abipy.flowtk.wrappers import Mrgddb
from abipy.flowtk.qutils import time2slurm
from abipy.abio.factories import InputFactory, PiezoElasticFromGsFactory
from abipy.abio.inputs import AbinitInput
from abipy.abio.input_tags import *
from abipy.abio.outputs import OutNcFile
from abipy.core.mixins import Has_Structure
from abipy.core import Structure
from fireworks.core.firework import Firework, FireTaskBase, FWAction, Workflow
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.utilities.fw_serializers import serialize_fw
from abiflows.fireworks.utils.task_history import TaskHistory
from abiflows.fireworks.utils.fw_utils import links_dict_update
from abiflows.fireworks.utils.fw_utils import set_short_single_core_to_spec
from abiflows.fireworks.tasks.abinit_common import TMPDIR_NAME, OUTDIR_NAME, INDIR_NAME, STDERR_FILE_NAME, \
    LOG_FILE_NAME, FILES_FILE_NAME, OUTPUT_FILE_NAME, INPUT_FILE_NAME, MPIABORTFILE, DUMMY_FILENAME, \
    ELPHON_OUTPUT_FILE_NAME, DDK_FILES_FILE_NAME, HISTORY_JSON
from abiflows.fireworks.utils.fw_utils import FWTaskManager


logger = logging.getLogger(__name__)


# files and folders names


class BasicAbinitTaskMixin(object):
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
        """
        Generates an instance of FWTaskManager. First looks for a 'ftm_file' key in the spec, otherwise generates it
        with FWTaskManager.from_user_config. The configuration is updated with the keywords defined in 'fw_policy'
        in the spec.
        """

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
        AbiPy |TaskManager|, that provides information about the queue adapters.
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

    def run_fake_autoparal(self, ftm):
        """
        In cases where the autoparal is not supported a fake run autoparal can be used to set the queueadapter.
        Takes the number of processors suggested by the manager given that the paral hints contain all the
        number of processors up tu max_cores and they all have the same efficiency.
        """
        manager = ftm.task_manager
        if not manager:
            msg = 'No task manager available: autoparal could not be performed.'
            logger.error(msg)
            raise InitializationError(msg)

        # all the options have the same priority, let the qadapter decide which is preferred.
        fake_conf_list = list({'tot_ncpus': i, 'mpi_ncpus': i, 'efficiency': 1} for i in range(1, manager.max_cores+1))
        from pymatgen.io.abinit.tasks import ParalHints
        pconfs = ParalHints({}, fake_conf_list)

        optconf = manager.select_qadapter(pconfs)
        qadapter_spec = manager.qadapter.get_subs_dict()

        d = pconfs.as_dict()
        d["optimal_conf"] = optconf
        json_pretty_dump(d, os.path.join(os.getcwd(), "autoparal.json"))

        return optconf, qadapter_spec, manager.qadapter

    def get_final_mod_spec(self, fw_spec):
        """
        Generates the standard mod_spec dict for the FWAction. Pushes the information of the current task to
        the list associated with self.task_type. Requires a "current_task_info" method.
        """

        return [{'_push': {'previous_fws->'+self.task_type: self.current_task_info(fw_spec)}}]
        # if 'previous_fws' in fw_spec:
        #     prev_fws = fw_spec['previous_fws'].copy()
        # else:
        #     prev_fws = {}
        # prev_fws[self.task_type] = [self.current_task_info(fw_spec)]
        # return [{'_set': {'previous_fws': prev_fws}}]

    def set_logger(self):
        """
        Set a logger for pymatgen.io.abinit and abipy
        """

        log_handler = logging.FileHandler('abipy.log')
        log_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
        logging.getLogger('pymatgen.io.abinit').addHandler(log_handler)
        logging.getLogger('abipy').addHandler(log_handler)
        logging.getLogger('abiflows').addHandler(log_handler)

    def link_ext(self, ext, source_dir, strict=True):
        """
        Links the required files from previous runs in the indata folder.
        It will first try to link the fortran file and then the Netcdf file, if the first is not found.

        Args:
            ext: extension that should be linked
            source_dir: path to the source directory
            strict: if True an exception is raised if the file is missing.

        Returns:
            The path to the generated link. None if strict=False and the file could not be found.
        """

        source = os.path.join(source_dir, self.prefix.odata + "_" + ext)
        logger.info("Need path {} with ext {}".format(source, ext))
        dest = os.path.join(self.workdir, self.prefix.idata + "_" + ext)
        if not os.path.exists(source):
            # Try netcdf file. TODO: this case should be treated in a cleaner way.
            source += ".nc"
            if os.path.exists(source): dest += ".nc"
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
        """
        Links the DDK files rom previous runs in the indata folder.
        Accepts more than one DDK file in the source_dir: multiple perturbations are allowed in a single calculation.
        Note that the DDK extension is not generated by abinit, but should be created from the appropriate 1WF file.
        """

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

    #TODO to avoid any regression this function will only be used to link 1WF and 1DEN files.
    # This method should be more general than link_ext, but currently fails in passing all the tests.
    # After fixing the problems the more general methods should replace link_ext.
    def link_1ext(self, ext, source_dir, strict=True):
        """
        Links the 1DEN and 1WF files in the indata folder.
        It will first try to link the fortran file and then the Netcdf file, if the first is not found.
        """

        # 1DEN is used as a general reference and to trigger the correct ird variable,
        # but the real extension in DEN.
        if "1DEN" in ext:
            ext = "DEN"
        source = Directory(os.path.join(source_dir,os.path.split(self.prefix.odata)[0])).has_abiext(ext)
        if not source:
            if strict:
                msg = "output file with extension {} is needed from {} dir, " \
                      "but it does not exist".format(ext, source_dir)
                logger.error(msg)
                raise InitializationError(msg)
            else:
                return None
        logger.info("Need path {} with ext {}".format(source, ext))
        # determine the correct extension
        #TODO check if this is correct for all the possible extensions, apart from 1WF
        ext_full = source.split('_')[-1]
        dest = os.path.join(self.workdir, self.prefix.idata + "_" + ext_full)

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

    #from Task
    # Prefixes for Abinit (input, output, temporary) files.
    Prefix = collections.namedtuple("Prefix", "idata odata tdata")
    pj = os.path.join

    prefix = Prefix(pj("indata", "in"), pj("outdata", "out"), pj("tmpdata", "tmp"))
    del Prefix, pj


@explicit_serialize
class AbiFireTask(BasicAbinitTaskMixin, FireTaskBase):

    # List of `AbinitEvent` subclasses that are tested in the check_status method.
    # Subclasses should provide their own list if they need to check the converge status.
    CRITICAL_EVENTS = [
    ]

    def __init__(self, abiinput, restart_info=None, handlers=None, is_autoparal=None, deps=None, history=None,
                 task_type=None):
        """
        Basic __init__, subclasses are supposed to define the same input parameters, add their own and call super for
        the basic ones. The input parameter should be stored as attributes of the instance for serialization and
        for inspection.

        Args:
            abiinput: an |AbinitInput| or an InputFactory. Defines the input used in the run
            restart_info: an instance of RestartInfo. This should be present in case the current task is a restart.
                Contains information useful to proceed with the restart.
            handlers: list of ErrorHandlers that should be used in case of error. If None all the error handlers
                available from abipy will be used.
            is_autoparal: whether the current task is just an autoparal job or not.
            deps: the required file dependencies from previous tasks (e.g. DEN, WFK, ...). Can be a single string,
                a list or a dict of the form {task_type: list of dependecies}. The dependencies will be retrieved
                from the 'previous_tasks' key in spec.
            history: a TaskHistory or a list of items that will be stored in a TaskHistory instance.
            task_type: a string that, if not None, overrides the task_type defined in the class.
        """

        if handlers is None:
            handlers = []
        if history is None:
            history = []
        self.abiinput = abiinput
        self.restart_info = restart_info

        self.handlers = handlers or [cls() for cls in events.get_event_handler_classes()]
        self.is_autoparal = is_autoparal

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
        """
        Sets up the working directory: adds attributes for all the files and directories.
        """

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
        # self.rename_outputs()

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
        Executes abinit and waits for the end of the process.
        The mpirun and abinit commands are retrived from the mpirun_cmd and abinit_cmd keys in the fw_policy
        of the FWTaskManager, that can be overridden by the values in the spec.
        Note that in case of missing definition of these parameters, the values fall back to the default
        values of  mpirun_cmd and abinit_cmd: 'mpirun' and 'abinit', assuming that these are properly retrieved
        from the $PATH.
        """

        def abinit_process():
            command = []
            #consider the case of serial execution
            if self.ftm.fw_policy.mpirun_cmd:
                command.extend(self.ftm.fw_policy.mpirun_cmd.split())
                if 'mpi_ncpus' in fw_spec:
                    command.extend(['-n', str(fw_spec['mpi_ncpus'])])
            command.extend(self.ftm.fw_policy.abinit_cmd.split())

            if self.walltime:
                mytimelimit = self.walltime
                if mytimelimit > 240:
                    mytimelimit -= 120
                command.extend(['--timelimit', time2slurm(mytimelimit)])

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
        Analyzes the main output file for possible Errors or Warnings. Will check the presence of an MPIABORTFILE
        if not output file is found.

        Args:
            source: "output" or "log". Determine which file will be parsed.

        Returns:
            :class:`EventReport` instance or None if none of the output files exist.
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
        Sets up the restart in case of convergence not achieved (both from abinit or from the convergence of
        cofiguration parameters) or in case of errors fixable by a ErrorHandler.
        If the job is completed calls conclude_task and prepares the FWAction.
        Raises an AbinitRuntimeError if unfixable errors are encountered or if the number or restarts exceeds
        the number defined in the policy.
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
                    msg = "Critical events couldn't be fixed by handlers. return code {}".format(self.returncode)
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
            # sanitize the text to avoid problems during database insertion
            # remove decode as incompatible with python 3
            # err_msg.decode("utf-8", "ignore")
        logger.error("return code {}".format(self.returncode))
        raise AbinitRuntimeError(self, err_msg)

    def check_parameters_convergence(self, fw_spec):
        """
        Base method related to the iterative convergence of some configuration parameter.
        Specific task should overwrite this method and implement appropriate checks and updates of the
        specific parameters.

        Args:
            fw_spec: The spec

        Returns:
            (tuple): tuple containing:

                - unconverged_params(dict): The uncoverged input variables that should be updated as keys and their
                    corresponding new values as values.
                - reset (boolean): True if a reset is required in self.prepare_restart.
        """

        return {}, False

    def _get_init_args_and_vals(self):
        """
        Inspection method to extract variables and values of the arguments of __init__ that should be stored in self.
        """

        init_dict = {}
        for arg in inspect.getargspec(self.__init__).args:
            if arg != "self":
                init_dict[arg] = self.__getattribute__(arg)

        return init_dict

    def _exclude_from_spec_in_restart(self):
        """
        List of keys that should not be forwarded to the newly created firework in case of restart.
        """

        return ['_tasks', '_exception_details']

    def prepare_restart(self, fw_spec, reset=False):
        """
        Determines the required information for the restart. It will be called at the end of a task which requires
        a restart (both for an error or for the convergence of some configuration parameter).
        Sets self.restart_info with an instance of RestartInfo.

        Args:
            fw_spec: the spec
            reset: if True a reset will be set in the restart_info

        Returns:
            (tuple): tuple containing:

                - local_restart (boolean): True if the restart should be in the same folder
                - new_fw (Firework): The new firework that should be used for detour
                - stored_data (dict): Dict to be saved in the "stored_data"
        """

        if self.restart_info:
            num_restarts = self.restart_info.num_restarts + 1
        else:
            num_restarts = 0

        self.restart_info = RestartInfo(previous_dir=self.workdir, reset=reset, num_restarts=num_restarts)

        # forward all the specs of the task
        new_spec = {k: v for k, v in fw_spec.items() if k not in self._exclude_from_spec_in_restart()}

        local_restart = False
        # only restart if it is known that there is a reasonable amount of time left
        if self.ftm.fw_policy.allow_local_restart and self.walltime and self.walltime/2 > (time.time() - self.start_time):
            local_restart = True

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
        stored_data = {}
        stored_data['report'] = self.report.as_dict()
        stored_data['finalized'] = False
        stored_data['restarted'] = True

        return local_restart, new_fw, stored_data

    def fix_abicritical(self, fw_spec):
        """
        method to fix crashes/error caused by abinit

        Returns:
            (tuple): tuple containing:
                retcode (int): 1 if task has been fixed else 0.
                reset (boolean): True if at least one of the corrections applied requires a reset
        """
        if not self.handlers:
            logger.info('Empty list of event handlers. Cannot fix abi_critical errors')
            return 0

        done = len(self.handlers) * [0]
        corrections = []

        for event in self.report:
            for i, handler in enumerate(self.handlers):
                if handler.can_handle(event) and not done[i]:
                    logger.info("handler {} will try to fix {}".format(handler, event))
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
        """
        Sets up the requirements for the task:
            - sets several attributes
            - generates the input in case self.abiinput is a factory
            - makes directories
            - handles information in '_exception_details'
        """

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
                out, err = p.communicate()
                status = p.returncode
                if status == 0 and out:
                    self.walltime = int(out.decode("utf-8"))
                else:
                    logger.warning("Impossible to get the walltime: " + err.decode("utf-8"))
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
                    task_type_source = list(previous_fws.keys())[0]
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
        # assume that DECODE_MONTY=True so that a handeled exception has already been deserialized
        if exception_details and isinstance(exception_details, AbinitRuntimeError):
            error_code = exception_details.ERROR_CODE
            if (self.ftm.fw_policy.continue_unconverged_on_rerun and error_code==ErrorCode.UNCONVERGED and
                    exception_details.abiinput and exception_details.restart_info and
                    exception_details.history):
                self.abiinput = exception_details.abiinput
                self.restart_info = exception_details.restart_info
                self.history = exception_details.history

    def run_task(self, fw_spec):

        try:
            self.setup_task(fw_spec)
            if self.is_autoparal:
                return self.autoparal(fw_spec)
            else:
                # loop to allow local restart
                while True:
                    self.config_run(fw_spec)
                    # try to recover previous run
                    if not self.ftm.fw_policy.recover_previous_job or not os.path.isfile(self.output_file.path):
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
        Restart method. Each subclass should implement its own restart. It is called at the beginning of a task
        that is the restart of a previous one. The base class should be called for common restarting operations.
        """
        pass

    def conclude_task(self, fw_spec):
        """
        Performs operations that should be handled at the end of the tasks in case of successful completion.
        Subclasses can overwrite to add additional operations, but the returns should be the same and
        it is suggested to call original method with super.

        Args:
            fw_spec: the spec

        Returns:
            (tuple): tuple containing:
                update_spec (dict): dictionary that should be passed to update_spec
                mod_spec (dict): dictionary that should be passed to mod_spec
                stored_data (dict): dictionary that should be passed to stored_data
        """
        stored_data = {}
        stored_data['report'] = self.report.as_dict()
        stored_data['finalized'] = True
        self.history.log_finalized(self.abiinput)
        stored_data['history'] = self.history.as_dict()
        update_spec = {}
        mod_spec = self.get_final_mod_spec(fw_spec)
        return update_spec, mod_spec, stored_data

    def current_task_info(self, fw_spec):
        """
        A dict containing information that should be passed to subsequent tasks.
        It should contain at least the current workdir and input. Subclasses can add specific additional information.
        """

        return dict(dir=self.workdir, input=self.abiinput)

    def autoparal(self, fw_spec):
        """
        Runs the task in autoparal and creates the new Firework with the optimized configuration.

        Args:
            fw_spec: the spec

        Returns:
            The FWAction containing the detour Firework.

        """
        # Copy the appropriate dependencies in the in dir. needed in some cases
        self.resolve_deps(fw_spec)

        optconf, qadapter_spec, qtk_qadapter = self.run_autoparal(self.abiinput, os.path.abspath('.'), self.ftm)

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
        """
        Method to link the required deps for the current FW for a specific task_type.
        Sets the ird variables corresponding to the linked dependecies.

        Args:
            previous_tasks: list of previous tasks from which the dependencies should be linked
            deps_list: list of dependencies that should be linked
        """
        for previous_task in previous_tasks:
            for d in deps_list:
                if d.startswith('@structure'):
                    if 'structure' not in previous_task:
                        msg = "previous_fws does not contain the structure."
                        logger.error(msg)
                        raise InitializationError(msg)
                    self.abiinput.set_structure(previous_task['structure'])
                #FIXME out.nc is not safe. Check if needed and move to other nc files in case.
                # elif d.startswith('@outnc'):
                #     varname = d.split('.')[1]
                #     outnc_path = os.path.join(previous_task['dir'], self.prefix.odata + "_OUT.nc")
                #     outnc_file = OutNcFile(outnc_path)
                #     vars = {varname: outnc_file[varname]}
                #     self.abiinput.set_vars(vars)
                elif not d.startswith('@'):
                    source_dir = previous_task['dir']
                    self.abiinput.set_vars(irdvars_for_ext(d))
                    if d == "DDK":
                        self.link_ddk(source_dir)
                    elif d == "1WF" or d == "1DEN":
                        self.link_1ext(d, source_dir)
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
            # If it's in the same dir, it is assumed that the dependencies have been correctly resolved in the previous
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
        """
        Called if a previous_fws key is in spec and the job is not a restart. Allows to load information from previous
        tasks if needed. Subclasses can overwrite to handle specific cases.

        Args:
            fw_spec: The spec
        """
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
            except OSError as e:
                if e.errno == errno.EEXIST:
                    os.remove(dest)
                    os.symlink(out_file, dest)
                else:
                    raise e

        return dest

    def in_to_in(self, in_file):
        """
        Copies the input file to the input of a previous task to the data directory of this task

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
        """
        Removes from the current input the ird values associated with the extension.
        Useful in case of reset during a restart.
        """
        if not isinstance(exts, (list, tuple)):
            exts = [exts]

        remove_vars = [v for e in exts for v in irdvars_for_ext(e).keys()]
        self.abiinput.remove_vars(remove_vars, strict=False)
        logger.info("Removing variables {} from input".format(remove_vars))


##############################
# Specific tasks
##############################


class GsFWTask(AbiFireTask):
    """
    Base Task to handle Ground state calculation.

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: GsFWTask
    """

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
        Open the GSR.nc_ file located in the in self.outdir.
        Returns |GsrFile| object, raise a PostProcessError exception if file could not be found or file is not readable.
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
    """
    Task to handle SCF calculations

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: ScfFWTask
    """

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
    """
    Task to handle non SCF calculations

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: NscfFWTask
    """

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
                raise RestartError(msg)

            # Move out --> in.
            self.out_to_in(restart_file)

            # Add the appropriate variable for restarting.
            irdvars = irdvars_for_ext(ext)
            self.abiinput.set_vars(irdvars)


@explicit_serialize
class NscfWfqFWTask(NscfFWTask):
    """
    Task to handle non SCF calculations for the calculations of the WFQ.
    Differs from :class:`NscfFWTask` for the different restart requirements.

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: NscfWfqFWTask
    """

    task_type = "nscf_wfq"

    def restart(self):
        """
        NSCF calculations can be restarted only if we have the WFK file.
        Wfq calculations require a WFK file for restart. The produced out_WFQ
        needs to be linked as a in_WFK with appropriate irdwfk=1.
        """
        if self.restart_info.reset:
            # remove non reset keys that may have been added in a previous restart
            self.remove_restart_vars(["WFQ", "WFK"])
        else:
            ext = "WFQ"
            restart_file = self.restart_info.prev_outdir.has_abiext(ext)
            if not restart_file:
                msg = "Cannot find the WFK file to restart from."
                logger.error(msg)
                raise RestartError(msg)

            # Move out --> in.
            self.out_to_in(restart_file)

            # Add the appropriate variable for restarting.
            irdvars = irdvars_for_ext("WFK")
            self.abiinput.set_vars(irdvars)

    def out_to_in(self, out_file):
        """
        links or copies, according to the fw_policy, the output file to the input data directory of this task
        and rename the file so that ABINIT will read it as an input data file.
        In the case of Wfq calculations out_WFQ needs to be linked as a in_WFK

        Returns:
            The absolute path of the new file in the indata directory.
        """
        in_file = os.path.basename(out_file).replace("out", "in", 1)
        in_file = os.path.basename(in_file).replace("WFQ", "WFK", 1)
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
            except OSError as e:
                if e.errno == errno.EEXIST:
                    os.remove(dest)
                    os.symlink(out_file, dest)
                else:
                    raise e

        return dest

@explicit_serialize
class RelaxFWTask(GsFWTask):
    """
    Task to handle relax calculations

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: RelaxFWTask
    """

    task_type = "relax"

    CRITICAL_EVENTS = [
        events.RelaxConvergenceWarning,
    ]

    def get_final_structure(self):
        """Read the final structure from the GSR.nc_ file."""
        try:
            with self.open_gsr() as gsr:
                return gsr.structure
        except AttributeError:
            msg = "Cannot find the GSR file with the final structure to restart from."
            logger.error(msg)
            raise PostProcessError(msg)

    def prepare_restart(self, fw_spec, reset=False):
        """
        Sets the final structure in abiinput, so that the relax can continue in the detour and calls the baseclass
        method.
        """

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
                        if last_timden.path.endswith(".nc"):
                            in_file_name = ("in_DEN.nc")
                        else:
                            in_file_name = ("in_DEN")
                        restart_file = self.out_to_in_tim(last_timden.path, in_file_name)
                        irdvars = irdvars_for_ext("DEN")

                if restart_file is None:
                    # Don't raise RestartError as the structure has been updated
                    logger.warning("Cannot find the WFK|DEN|TIM?_DEN file to restart from.")
                else:
                    # Add the appropriate variable for restarting.
                    self.abiinput.set_vars(irdvars)
                    logger.info("Will restart from %s", restart_file)

    def current_task_info(self, fw_spec):
        """
        Add the final structure to the basic current_task_info
        """

        d = super(RelaxFWTask, self).current_task_info(fw_spec)
        d['structure'] = self.get_final_structure()
        return d

    # def conclude_task(self, fw_spec):
    #     update_spec, mod_spec, stored_data = super(RelaxFWTask, self).conclude_task(fw_spec)
    #     update_spec['previous_run']['structure'] = self.get_final_structure()
    #     return update_spec, mod_spec, stored_data

    @property
    def hist_nc_path(self):
        """Absolute path of the HIST.nc_ file. Empty string if file is not present."""
        # Lazy property to avoid multiple calls to has_abiext.
        try:
            return self._hist_nc_path
        except AttributeError:
            path = self.outdir.has_abiext("HIST")
            if path: self._hist_nc_path = path
            return path

    def out_to_in_tim(self, out_file, in_file):
        """
        links or copies, according to the fw_policy, the output file to the input data directory of this task
        and rename the file so that ABINIT will read it as an input data file. for the TIM file the input needs to
        be specified as depends on the specific iteration.

        Returns:
            The absolute path of the new file in the indata directory.
        """
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
            except OSError as e:
                if e.errno == errno.EEXIST:
                    os.remove(dest)
                    os.symlink(out_file, dest)
                else:
                    raise e

        return dest


@explicit_serialize
class HybridFWTask(GsFWTask):
    """
    Task to handle hybrid functional calculations based on GW.

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: HybridFWTask
    """

    task_type = "hybrid"

    CRITICAL_EVENTS = [
    ]

    @property
    def sigres_path(self):
        """Absolute path of the SIGRES.nc file. Empty string if file is not present."""
        # Lazy property to avoid multiple calls to has_abiext.
        try:
            return self._sigres_path
        except AttributeError:
            path = self.outdir.has_abiext("SIGRES")
            if path: self._sigres_path = path
            return path

    def open_sigres(self):
        """
        Open the SIGRES.nc_ file located in the in self.outdir.
        Returns |SigresFile| object, None if file could not be found or file is not readable.
        """
        sigres_path = self.sigres_path

        if not sigres_path:
            msg = "%s didn't produce a SIGRES file in %s" % (self, self.outdir)
            logger.critical(msg)
            raise PostProcessError(msg)

        # Open the SIGRES file and add its data to results.out
        from abipy.electrons.gw import SigresFile
        try:
            return SigresFile(sigres_path)
        except Exception as exc:
            msg = "Exception while reading SIGRES file at %s:\n%s" % (sigres_path, str(exc))
            logger.critical(msg)
            raise PostProcessError(msg)


@explicit_serialize
class DfptTask(AbiFireTask):
    """
    Base Task to handle DFPT calculations

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: DfptTask
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
class DdkTask(DfptTask):
    """
    Task to handle DDK calculations

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: DdkTask
    """

    task_type = "ddk"

    def conclude_task(self, fw_spec):
        """
        Extends the base class method in order to create a _DDK file from the 1WF.
        """

        # make a link to _DDK of the 1WF file to ease the link in the dependencies
        wf_files = self.outdir.find_1wf_files()
        if not wf_files:
            raise PostProcessError("Couldn't link 1WF files.")
        for f in wf_files:
            os.symlink(f.path, f.path+'_DDK')

        return super(DdkTask, self).conclude_task(fw_spec)


@explicit_serialize
class DdeTask(DfptTask):
    """
    Task to handle DDE calculations

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: DdeTask
    """

    task_type = "dde"


@explicit_serialize
class PhononTask(DfptTask):
    """
    Task to handle phonon calculations

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: PhononTask
    """

    task_type = "phonon"


@explicit_serialize
class BecTask(DfptTask):
    """
    Task to handle BEC calculations

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: BecTask
    """

    task_type = "bec"


@explicit_serialize
class StrainPertTask(DfptTask):
    """
    Task to handle strain calculations

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: StrainPertTask
    """

    task_type = "strain_pert"


@explicit_serialize
class DteTask(DfptTask):
    """
    Task to handle the third derivatives with respect to the electric field.

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: DteTask
    """

    CRITICAL_EVENTS = []

    task_type = "dte"

    # non-linear does not support autoparal. Take the suggested number of processors.
    def run_autoparal(self, abiinput, autoparal_dir, ftm, clean_up='move'):
        """
        Non-linear does not support autoparal, so this will provide a fake run of the autoparal.
        """
        return self.run_fake_autoparal(ftm)


##############################
# Convergence tasks
##############################

@explicit_serialize
class RelaxDilatmxFWTask(RelaxFWTask):
    """
    Task to handle relax calculations with iterative convergence of the dilatmx

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: DteTask
    """

    def __init__(self, abiinput, restart_info=None, handlers=None, is_autoparal=None, deps=None, history=None,
                 target_dilatmx=1.01):
        if handlers is None:
            handlers = []
        if history is None:
            history = []
        self.target_dilatmx = target_dilatmx
        super(RelaxDilatmxFWTask, self).__init__(abiinput=abiinput, restart_info=restart_info, handlers=handlers,
                                                 is_autoparal=is_autoparal, deps=deps, history=history)

    def check_parameters_convergence(self, fw_spec):
        """
        Checks if the target value for the dilatmx has been reached. If not reduces the values for the dilatmx and
        signals that a restart is needed.

        Args:
            fw_spec: the spec of the Firework
        """

        actual_dilatmx = self.abiinput.get('dilatmx', 1.)
        new_dilatmx = actual_dilatmx - min((actual_dilatmx-self.target_dilatmx), actual_dilatmx*0.03)
        #FIXME reset can be False with paral_kgb==1
        return {'dilatmx': new_dilatmx} if new_dilatmx != actual_dilatmx else {}, True


##############################
# Wrapper tasks
##############################


@explicit_serialize
class MergeDdbAbinitTask(BasicAbinitTaskMixin, FireTaskBase):
    """
    Task to handle the merge of multiple DDB files with mrgddb

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: MergeDdbAbinitTask
    """

    task_type = "mrgddb"

    def __init__(self, ddb_source_task_types=None, delete_source_ddbs=True, num_ddbs=None, task_type=None):
        """
        ddb_source_task_type: list of task types that will be used as source for the DDB to be merged.
        The default is [PhononTask.task_type, DdeTask.task_type, BecTask.task_type]
        delete_ddbs: delete the ddb files used after the merge
        num_ddbs: number of ddbs to be merged. If set will be used to check that the correct number of ddbs have been
         passed to the task. Tha task will fizzle if the numbers do not match
        """

        if ddb_source_task_types is None:
            ddb_source_task_types = [ScfFWTask.task_type, PhononTask.task_type, DdeTask.task_type, BecTask.task_type,
                                     DteTask.task_type, StrainPertTask.task_type]
        elif not isinstance(ddb_source_task_types, (list, tuple)):
            ddb_source_task_types = [ddb_source_task_types]

        self.ddb_source_task_types = ddb_source_task_types
        self.delete_source_ddbs = delete_source_ddbs
        self.num_ddbs = num_ddbs

        if task_type is not None:
            self.task_type = task_type

    def get_ddb_list(self, previous_fws, task_type):
        """
        Given a task_type that can produce DDB files and whole list of previous fireworks available here,
        gets the list of DDB files that should be merged.

        Args:
            previous_fws: list of previous fireworks
            task_type: string describing the task type

        Returns:
            The list of DDB files that should be linked
        """

        ddb_files = []
        #Check that the same directory is not passed more than once (in principle it should not be needed.
        # kept from previous version to avoid regressions)
        mydirs = []
        for t in previous_fws.get(task_type, []):
            if t['dir'] in mydirs:
                continue
            # filepaths = Directory(os.path.join(t['dir'], OUTDIR_NAME)).list_filepaths()
            # a DDB.nc is usually produced along with the text DDB file. has_abiext handles the extraction of
            # the text one, ignoring the netCDF. If has_abiext is changed the problem should be handled here.
            # ddb = self.get_ddb_from_filepaths(filepaths=filepaths)
            ddb = Directory(os.path.join(t['dir'], OUTDIR_NAME)).has_abiext('DDB')
            if not ddb:
                msg = "One of the task of type {} (folder: {}) " \
                      "did not produce a DDB file!".format(task_type, t['dir'])
                raise InitializationError(msg)
            mydirs.append(t['dir'])
            ddb_files.append(ddb)
        return ddb_files

    # def get_ddb_from_filepaths(self, filepaths):
    #     #TODO: temporary fix due to new DDB.nc in addition to DDB ... then has_abiext finds multiple multiple files ...
    #     ext = '_DDB'
    #
    #     files = []
    #     for f in filepaths:
    #         if f.endswith(ext):
    #             files.append(f)
    #
    #     if not files:
    #         return None
    #
    #     if len(files) > 1:
    #         # ABINIT users must learn that multiple datasets are bad!
    #         err_msg = "Found multiple files with the same extensions:\n %s\nPlease avoid the use of mutiple datasets!" % files
    #         raise ValueError(err_msg)
    #
    #     return files[0]

    def get_event_report(self, ofile_name="mrgddb.stdout"):
        """
        Analyzes the main output file for possible Errors or Warnings.

        Args:
            ofile_name: Name of the outpu file.

        Returns:
            :class:`EventReport` instance or None if the output file does not exist.
        """

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
        """
        Sets up the working directory: adds attributes for all the files and directories.
        """

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

            # Temporary fix ... mrgddb doesnt seem to work when I merge the GS DDB file with the Strain DDB file
            # because the info on the pseudopotentials is not in the GS DDB file ...
            #  www.welcome2quickanddirty.com (DavidWaroquiers)
            if 'PAW_datasets_description_correction' in fw_spec:
                if len(ddb_files) != 2:
                    raise ValueError('Fix is temporary and only for a number of DDBs equal to 2')
                fname_with_psp = None
                psp_lines = []

                for fname in ddb_files:
                    in_psp_info = False
                    with open(fname, 'r') as fh:
                        dd = fh.readlines()
                        for iline, line in enumerate(dd):
                            if 'No information on the potentials yet' in line:
                                break
                            if 'Description of the PAW dataset(s)' in line:
                                in_psp_info = True
                                fname_with_psp = fname
                            if in_psp_info:
                                if '**** Database of total energy derivatives ****' in line:
                                    break
                                psp_lines.append(line)
                    if fname_with_psp:
                        break

                if not fname_with_psp:
                    raise ValueError('Should have at least one DDB with the psp info ...')

                out_ddb_backup = '{}.backup'.format(out_ddb)
                shutil.move(out_ddb, out_ddb_backup)

                fw = open(out_ddb, 'w')
                with open(out_ddb_backup, 'r') as fh:
                    dd = fh.readlines()
                    just_copy = True
                    for line in dd:
                        if 'Description of the PAW dataset(s)' in line:
                            just_copy = False
                            for pspline in psp_lines:
                                fw.write(pspline)
                        if just_copy:
                            fw.write(line)
                            continue
                        if '**** Database of total energy derivatives ****' in line:
                            just_copy = True
                            fw.write(line)
                            continue
                fw.close()

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
        """
        A dict containing information that should be passed to subsequent tasks.
        In this case it contains the current workdir.
        """

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
class AnaDdbAbinitTask(BasicAbinitTaskMixin, FireTaskBase):
    """
    Task that handles the run of anaddb based on a custom input

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: AnaDdbAbinitTask
    """

    task_type = "anaddb"

    def __init__(self, anaddb_input, restart_info=None, handlers=None, is_autoparal=None, deps=None, history=None,
                 task_type=None):
        """
        Args:
            anaddb_input: |AnaddbInput| object. Defines the input used in the run
            restart_info: an instance of RestartInfo. This should be present in case the current task is a restart.
                Contains information useful to proceed with the restart.
            handlers: list of ErrorHandlers that should be used in case of error. If None all the error handlers
                available from abipy will be used.
            is_autoparal: whether the current task is just an autoparal job or not.
            deps: the required file dependencies from previous tasks (e.g. DEN, WFK, ...). Can be a single string,
                a list or a dict of the form {task_type: list of dependecies}. The dependencies will be retrieved
                from the 'previous_tasks' key in spec.
            history: a TaskHistory or a list of items that will be stored in a TaskHistory instance.
            task_type: a string that, if not None, overrides the task_type defined in the class.
        """

        if handlers is None:
            handlers = []
        if history is None:
            history = []

        self.anaddb_input = anaddb_input
        self.restart_info = restart_info

        self.handlers = handlers or [cls() for cls in events.get_event_handler_classes()]
        self.is_autoparal = is_autoparal

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
        """Absolute path of the GSR.nc_ file. Empty string if file is not present."""
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

    def resolve_deps_per_task_type(self, previous_tasks, deps_list):
        """
        Method to link the required deps for the current FW for a specific task_type.

        Args:
            previous_tasks: list of previous tasks from which the dependencies should be linked
            deps_list: list of dependencies that should be linked
        """
        ddb_dirs = []
        for previous_task in previous_tasks:
            for d in deps_list:
                source_dir = previous_task['dir']
                if d == "DDB":
                    # Check that the same directory is not passed more than once (in principle it should not be needed.
                    # kept from previous version to avoid regressions)
                    if source_dir in ddb_dirs:
                        continue
                    ddb_dirs.append(source_dir)
                    self.ddb_filepath = self.link_ext(d, source_dir)
                elif d == "GKK":
                    self.gkk_filepath = self.link_ext(d, source_dir)
                elif d == "DDK":
                    self.ddk_filepaths.extend(self.link_ddk(source_dir))
                else:
                    logger.warning("Extensions {} is not used in anaddb and will be ignored".format(d))
                    continue

    def resolve_deps(self, fw_spec):
        """
        Method to link the required deps for the current FW.
        """

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
        from the $PATH. By default, anaddb is retrieved from the PATH.
        """

        def anaddb_process():
            command = []
            #consider the case of serial execution
            if self.ftm.fw_policy.mpirun_cmd:
                command.extend(self.ftm.fw_policy.mpirun_cmd.split())
                if 'mpi_ncpus' in fw_spec:
                    command.extend(['-n', str(fw_spec['mpi_ncpus'])])
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
        """
        Sets up the requirements for the task:
            - sets several attributes
            - makes directories
            - writes input files
        """
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
        if self.returncode != 0:
            raise RuntimeError("Return code different from 0: {}".format(self.returncode))
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
        """Absolute path of the run.abo_PHBST.nc file. Empty string if file is not present."""
        # Lazy property to avoid multiple calls to has_abiext.
        try:
            return self._phbst_path
        except AttributeError:
            path = os.path.join(self.workdir, "run.abo_PHBST.nc")
            if path: self._phbst_path = path
            return path

    @property
    def phdos_path(self):
        """Absolute path of the run.abo_PHDOS.nc file. Empty string if file is not present."""
        # Lazy property to avoid multiple calls to has_abiext.
        try:
            return self._phdos_path
        except AttributeError:
            path = os.path.join(self.workdir, "run.abo_PHDOS.nc")
            if path: self._phdos_path = path
            return path

    @property
    def anaddb_nc_path(self):
        """Absolute path of the anaddb.nc file. Empty string if file is not present."""
        # Lazy property to avoid multiple calls to has_abiext.
        try:
            return self._anaddbnc_path
        except AttributeError:
            path = os.path.join(self.workdir, "anaddb.nc")
            if path: self._anaddbnc_path = path
            return path

    def open_phbst(self):
        """
        Open PHBST file produced by Anaddb and returns |PhbstFile| object.
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
            msg = "Exception while reading PHBST file at %s:\n%s" % (self.phbst_path, str(exc))
            logger.critical(msg)
            raise PostProcessError(msg)

    def open_phdos(self):
        """
        Open PHDOS file produced by Anaddb and returns |PhdosFile| object.
        Raise a PostProcessError exception if file could not be found or file is not readable.
        """
        from abipy.dfpt.phonons import PhdosFile
        if not self.phdos_path:
            msg = "No PHDOS file available for task {} in {}".format(self, self.outdir)
            logger.critical(msg)
            raise PostProcessError(msg)

        try:
            return PhdosFile(self.phdos_path)
        except Exception as exc:
            msg = "Exception while reading PHDOS file at %s:\n%s" % (self.phdos_path, str(exc))
            logger.critical(msg)
            raise PostProcessError(msg)

    def open_anaddbnc(self):
        """
        Open anaddb.nc file produced by Anaddb and returns |AnaddbNcFile| object.
        Raise a PostProcessError exception if file could not be found or file is not readable.
        """
        from abipy.dfpt.anaddbnc import AnaddbNcFile
        if not self.anaddb_nc_path:
            msg = "No anaddb.nc file available for task {} in {}".format(self, self.outdir)
            logger.critical(msg)
            raise PostProcessError(msg)

        try:
            return AnaddbNcFile(self.anaddb_nc_path)
        except Exception as exc:
            msg = "Exception while reading anaddb.nc file at %s:\n%s" % (self.anaddb_nc_path, str(exc))
            logger.critical(msg)
            raise PostProcessError(msg)


@explicit_serialize
class AutoparalTask(AbiFireTask):
    """
    Task to run the autoparal for many tasks of the same type already defined as children

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: AutoparalTask
    """

    task_type = "autoparal"

    def __init__(self, abiinput, restart_info=None, handlers=None, deps=None, history=None, is_autoparal=True,
                 task_type=None, forward_spec=False, skip_spec_keys=None):
        """
        Note that the method still takes is_autoparal as input even if this is switched to True irrespectively of the
        provided value, as the point of the task is to run autoparal. This is done to preserve the API in cases where
        automatic generation of tasks is involved.
        skip_spec_keys allows to specify a list of keys to skip when forwarding the spec: 'wf_task_index' will
        be always skipped. All the reserved keys starting with _ will always be skipped as well.
        If abiinput is None, autoparal will not run and the preferred configuration will be chosen based on
        the options set in the manager.
        #FIXME find a better solution if this model is preserved
        """
        if handlers is None:
            handlers = []
        if history is None:
            history = []
        super(AutoparalTask, self).__init__(abiinput, restart_info=restart_info, handlers=handlers, is_autoparal=True,
                                            deps=deps, history=history, task_type=task_type)
        self.forward_spec = forward_spec
        if not skip_spec_keys:
            skip_spec_keys = []
        skip_spec_keys.append('wf_task_index')
        self.skip_spec_keys = skip_spec_keys

    def autoparal(self, fw_spec):
        """
        Runs the autoparal, if an input is available. Does not return a detour, updates the children fws instead.
        """

        # Copy the appropriate dependencies in the in dir. needed in some cases
        self.resolve_deps(fw_spec)

        if self.abiinput is None:
            optconf, qadapter_spec, qtk_qadapter = self.run_fake_autoparal(self.ftm)
        else:
            optconf, qadapter_spec, qtk_qadapter = self.run_autoparal(self.abiinput, os.path.abspath('.'), self.ftm)

        self.history.log_autoparal(optconf)
        mod_spec = [{'_push_all': {'spec->_tasks->0->abiinput->abi_args': list(optconf.vars.items())}}]

        if self.forward_spec:
            # forward all the specs of the task
            new_spec = {k: v for k, v in fw_spec.items() if k.startswith('_') and k not in self.skip_spec_keys}
        else:
            new_spec = {}
        # set quadapter specification. Note that mpi_ncpus may be different from ntasks
        new_spec['_queueadapter'] = qadapter_spec
        new_spec['mpi_ncpus'] = optconf['mpi_ncpus']

        return FWAction(update_spec=new_spec, mod_spec=mod_spec)

##############################
# Generation tasks
##############################

@explicit_serialize
class GeneratePhononFlowFWAbinitTask(BasicAbinitTaskMixin, FireTaskBase):
    """
    Task that generates all the phonon perturbation based on the input of the previous ground state step

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: GeneratePhononFlowFWAbinitTask
    """

    def __init__(self, phonon_factory, previous_task_type=ScfFWTask.task_type, handlers=None, with_autoparal=None, ddb_file=None):
        if handlers is None:
            handlers = []
        self.phonon_factory = phonon_factory
        self.previous_task_type = previous_task_type
        self.handlers = handlers
        self.with_autoparal=with_autoparal
        self.ddb_file = ddb_file

    def get_fws(self, multi_inp, task_class, deps, new_spec, ftm, nscf_fws=None):
        """
        Prepares the fireworks for a specific type of calculation

        Args:
            multi_inp: |MultiDataset| with the inputs that should be run
            task_class: class of the tasks that should be generated
            deps: dict with the dependencies already set for this type of task
            new_spec: spec for the new Fireworks that will be created
            ftm: a FWTaskManager
            nscf_fws: list of NSCF fws for the calculation of WFQ files, in case they are present.
                Will be linked if needed.

        Returns:
            (tuple): tuple containing:

                - fws (list): The list of new Fireworks.
                - fw_deps (dict): The dependencies related to these fireworks. Should be used when generating
                    the workflow.
        """

        if deps is None:
            deps = {}

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
                inp.set_vars(optconf.vars)

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

        # compatibility with old DECODE_MONTY=False
        if not isinstance(previous_input, AbinitInput):
            previous_input = AbinitInput.from_dict(previous_input)

        ftm = self.get_fw_task_manager(fw_spec)

        if self.with_autoparal is None:
            self.with_autoparal = ftm.fw_policy.autoparal

        if self.with_autoparal:
            if not ftm.has_task_manager():
                msg = 'No task manager available: autoparal could not be performed.'
                logger.error(msg)
                raise InitializationError(msg)

        # inject task manager
        tasks.set_user_config_taskmanager(ftm.task_manager)

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
            nscf_fws, nscf_fw_deps= self.get_fws(nscf_inputs, NscfWfqFWTask,
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
        # add one for the scf that is linked to the mrgddbtask and will be merged as well
        num_ddbs_to_be_merged = len(ph_fws) + len(dde_fws) + len(bec_fws) + 1
        mrgddb_fw = Firework(MergeDdbAbinitTask(num_ddbs=num_ddbs_to_be_merged, delete_source_ddbs=False), spec=mrgddb_spec,
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


#TODO old implementation of GeneratePiezoElasticFlowFWAbinitTask based on SRC. Needs to be rewritten.
# @explicit_serialize
# class GeneratePiezoElasticFlowFWAbinitTask(BasicAbinitTaskMixin, FireTaskBase):
#     def __init__(self, piezo_elastic_factory=None, previous_scf_task_type=ScfFWTask.task_type,
#                  previous_ddk_task_type=DdkTask.task_type,
#                  handlers=None, validators=None, mrgddb_task_type='mrgddb-strains', rf_tol=None):
#         if piezo_elastic_factory is None:
#             self.piezo_elastic_factory = PiezoElasticFromGsFactory(rf_tol=rf_tol, rf_split=True)
#         else:
#             self.piezo_elastic_factory = piezo_elastic_factory
#         self.previous_scf_task_type = previous_scf_task_type
#         self.previous_ddk_task_type = previous_ddk_task_type
#         self.handlers = handlers
#         self.validators = validators
#         self.mrgddb_task_type = mrgddb_task_type
#         self.rf_tol = rf_tol
#
#     def run_task(self, fw_spec):
#         # Get the previous SCF input
#         previous_scf_input = fw_spec.get('previous_fws', {}).get(self.previous_scf_task_type,
#                                                                  [{}])[0].get('input', None)
#         if not previous_scf_input:
#             raise InitializationError('No input file available '
#                                       'from task of type {}'.format(self.previous_scf_task_type))
#         #previous_scf_input = AbinitInput.from_dict(previous_scf_input)
#
#         # # Get the previous DDK input
#         # previous_ddk_input = fw_spec.get('previous_fws', {}).get(self.previous_ddk_task_type,
#         #                                                          [{}])[0].get('input', None)
#         # if not previous_ddk_input:
#         #     raise InitializationError('No input file available '
#         #                               'from task of type {}'.format(self.previous_ddk_task_type))
#         # previous_ddk_input = AbinitInput.from_dict(previous_ddk_input)
#
#         ftm = self.get_fw_task_manager(fw_spec)
#         tasks._USER_CONFIG_TASKMANAGER = ftm.task_manager
#         # if self.with_autoparal:
#         #     if not ftm.has_task_manager():
#         #         msg = 'No task manager available: autoparal could not be performed.'
#         #         logger.error(msg)
#         #         raise InitializationError(msg)
#         #
#         #     # inject task manager
#         #     tasks._USER_CONFIG_TASKMANAGER = ftm.task_manager
#
#         # Get the strain RF inputs
#         piezo_elastic_inputs = self.piezo_elastic_factory.build_input(previous_scf_input)
#         rf_strain_inputs = piezo_elastic_inputs.filter_by_tags(STRAIN)
#
#         initialization_info = fw_spec.get('initialization_info', {})
#         initialization_info['input_factory'] = self.piezo_elastic_factory.as_dict()
#         new_spec = dict(previous_fws=fw_spec['previous_fws'], initialization_info=initialization_info)
#
#         # Get the initial queue_adapter_updates
#         queue_adapter_update = initialization_info.get('queue_adapter_update', None)
#
#         # Create the SRC fireworks for each perturbation
#         all_SRC_rf_fws = []
#         total_list_fws = []
#         fws_deps = {}
#         rf_strain_handlers = self.handlers['_all'] if self.handlers is not None else []
#         rf_strain_validators = self.validators['_all'] if self.validators is not None else []
#         for istrain_pert, rf_strain_input in enumerate(rf_strain_inputs):
#             SRC_rf_fws = createSRCFireworksOld(task_class=StrainPertTask, task_input=rf_strain_input, SRC_spec=new_spec,
#                                                initialization_info=initialization_info,
#                                                wf_task_index_prefix='rfstrains-pert-{:d}'.format(istrain_pert+1),
#                                                handlers=rf_strain_handlers, validators=rf_strain_validators,
#                                                deps={self.previous_scf_task_type: 'WFK',
#                                                   self.previous_ddk_task_type: 'DDK'},
#                                                queue_adapter_update=queue_adapter_update)
#             all_SRC_rf_fws.append(SRC_rf_fws)
#             total_list_fws.extend(SRC_rf_fws['fws'])
#             links_dict_update(links_dict=fws_deps, links_update=SRC_rf_fws['links_dict'])
#
#         # Adding the MrgDdb Firework
#         mrgddb_spec = dict(new_spec)
#         mrgddb_spec['wf_task_index_prefix'] = 'mrgddb-rfstrains'
#         mrgddb_spec['wf_task_index'] = mrgddb_spec['wf_task_index_prefix']
#         mrgddb_spec = set_short_single_core_to_spec(mrgddb_spec)
#         mrgddb_spec['_priority'] = 10
#         num_ddbs_to_be_merged = len(all_SRC_rf_fws)
#         mrgddb_fw = Firework(MergeDdbAbinitTask(num_ddbs=num_ddbs_to_be_merged, delete_source_ddbs=True,
#                                                 task_type= self.mrgddb_task_type),
#                              spec=mrgddb_spec,
#                              name=mrgddb_spec['wf_task_index'])
#         total_list_fws.append(mrgddb_fw)
#         #Adding the dependencies
#         for src_fws in all_SRC_rf_fws:
#             links_dict_update(links_dict=fws_deps, links_update={src_fws['check_fw']: mrgddb_fw})
#
#         rf_strains_wf = Workflow(total_list_fws, fws_deps)
#
#         return FWAction(detours=rf_strains_wf)

##############################
# Exceptions
##############################

class ErrorCode(object):
    """
    Error code to classify the errors
    """

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
    """
    Base class for the errors in abiflows
    """

    def __init__(self, msg):
        super(AbiFWError, self).__init__(msg)
        self.msg = msg

    def to_dict(self):
        return dict(error_code=self.ERROR_CODE, msg=self.msg)


class AbinitRuntimeError(AbiFWError):
    """
    Exception raised for errors during Abinit calculation.
    Contains the information about the errors and warning extracted from the output files.
    Initialized with a task, uses it to prepare a suitable error message.
    """

    ERROR_CODE = ErrorCode.ERROR

    def __init__(self, task=None, msg=None, num_errors=None, num_warnings=None, errors=None, warnings=None):
        """
        If the task has a report all the information will be extracted from it, otherwise the arguments will be used.

        Args:
            task: the abiflows Task
            msg: the error message
            num_errors: number of errors in the abinit execution. Only used if task doesn't have a report.
            num_warnings: number of warning in the abinit execution. Only used if task doesn't have a report.
            errors: list of errors in the abinit execution. Only used if task doesn't have a report.
            warnings: list of warnings in the abinit execution. Only used if task doesn't have a report.
        """

        # This can handle both the cases of DECODE_MONTY=True and False (Since it has a from_dict method).
        super(AbinitRuntimeError, self).__init__(msg)
        self.task = task
        if self.task is not None and hasattr(self.task, "report") and self.task.report is not None:
            report = self.task.report
            self.num_errors = report.num_errors
            self.num_warnings = report.num_warnings
            self.errors = report.errors
            self.warnings = report.warnings
        else:
            self.num_errors = num_errors
            self.num_warnings = num_warnings
            self.errors = errors
            self.warnings = warnings
        self.msg = msg

    @pmg_serialize
    def to_dict(self):
        d = {}
        d['num_errors'] = self.num_errors
        d['num_warnings'] = self.num_warnings
        if self.errors:
            errors = []
            for error in self.errors:
                errors.append(error.as_dict())
            d['errors'] = errors
        if self.warnings:
            warnings = []
            for warning in self.warnings:
                warnings.append(warning.as_dict())
            d['warnings'] = warnings
        if self.msg:
            d['error_message'] = self.msg

        d['error_code'] = self.ERROR_CODE

        return d

    def as_dict(self):
        return self.to_dict()

    @classmethod
    def from_dict(cls, d):
        dec = MontyDecoder()
        warnings = [dec.process_decoded(w) for w in d['warnings']] if 'warnings' in d else []
        errors = [dec.process_decoded(w) for w in d['errors']] if 'errors' in d else []
        msg = d['error_message'] if 'error_message' in d else None

        return cls(warnings=warnings, errors=errors, num_errors=d['num_errors'], num_warnings=d['num_warnings'],
                   msg=msg)


class UnconvergedError(AbinitRuntimeError):
    """
    Exception raised when a calculation didn't converge within the selected number of restarts.
    """

    ERROR_CODE = ErrorCode.UNCONVERGED

    def __init__(self, task=None, msg=None, num_errors=None, num_warnings=None, errors=None, warnings=None,
                 abiinput=None, restart_info=None, history=None):
        """
        If the task has a report all the information will be extracted from it, otherwise the arguments will be used.
        It contains information that can be used to further restart the job.

        Args:
            task: the abiflows Task
            msg: the error message
            num_errors: number of errors in the abinit execution. Only used if task doesn't have a report.
            num_warnings: number of warning in the abinit execution. Only used if task doesn't have a report.
            errors: list of errors in the abinit execution. Only used if task doesn't have a report.
            warnings: list of warnings in the abinit execution. Only used if task doesn't have a report.
            abiinput: the last AbinitInput used.
            restart_info: the RestartInfo required to restart the job.
            history: a TaskHistory.
        """
        super(UnconvergedError, self).__init__(task, msg, num_errors, num_warnings, errors, warnings)
        self.abiinput = abiinput
        self.restart_info = restart_info
        self.history = history

    @pmg_serialize
    def to_dict(self):
        d = super(UnconvergedError, self).to_dict()
        d['abiinput'] = self.abiinput.as_dict() if self.abiinput else None
        d['restart_info'] = self.restart_info.as_dict() if self.restart_info else None
        d['history'] = self.history.as_dict() if self.history else None
        return d

    @classmethod
    def from_dict(cls, d):
        dec = MontyDecoder()
        warnings = [dec.process_decoded(w) for w in d['warnings']] if 'warnings' in d else []
        errors = [dec.process_decoded(w) for w in d['errors']] if 'errors' in d else []
        if 'abiinput' in d and d['abiinput'] is not None:
            abiinput = dec.process_decoded(d['abiinput'])
        else:
            abiinput = None
        if 'restart_info' in d and d['restart_info'] is not None:
            restart_info = dec.process_decoded(d['restart_info'])
        else:
            restart_info = None
        if 'history' in d and d['history'] is not None:
            history = dec.process_decoded(d['history'])
        else:
            history = None
        return cls(warnings=warnings, errors=errors, num_errors=d['num_errors'], num_warnings=d['num_warnings'],
                   msg=d['error_message'], abiinput=abiinput, restart_info=restart_info, history=history)


class UnconvergedParametersError(UnconvergedError):
    """
    Exception raised when the iteration to converge some parameter didn't converge within the selected number
    of restarts.
    """

    ERROR_CODE = ErrorCode.UNCONVERGED_PARAMETERS


class WalltimeError(AbiFWError):
    """
    Exception raised when the calculation didn't complete within the specified walltime.
    """
    ERROR_CODE = ErrorCode.WALLTIME


class InitializationError(AbiFWError):
    """
    Exception raised if errors are present during the initialization of the task, before abinit is started.
    """
    ERROR_CODE = ErrorCode.INITIALIZATION


class RestartError(InitializationError):
    """
    Exception raised if errors show up during the set up of the restart.
    """
    ERROR_CODE = ErrorCode.RESTART


class PostProcessError(AbiFWError):
    """
    Exception raised if problems are encountered during the post processing of the abinit calculation.
    """

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
        """
        A Directory object pointing to the outdir of the previous step.
        """

        return Directory(os.path.join(self.previous_dir, OUTDIR_NAME))

    @property
    def prev_indir(self):
        """
        A Directory object pointing to the indir of the previous step.
        """

        return Directory(os.path.join(self.previous_dir, INDIR_NAME))


class ElasticComplianceTensor(Has_Structure):
    """This object is used to store the elastic and compliance tensors."""

    def __init__(self, elastic_tensor, compliance_tensor, structure, additional_info=None):
        """

        Args:
            elastic_tensor: (6, 6) array with the elastic tensor in Cartesian coordinates
            compliance_tensor: (6, 6) array with the compliance tensor in Cartesian coordinates
            structure: |Structure| object.
        """
        self._structure = structure
        self.elastic_tensor = elastic_tensor
        self.compliance_tensor = compliance_tensor
        self.additional_info = additional_info

    @property
    def structure(self):
        """|Structure| object."""
        return self._structure

    def __repr__(self):
        return self.to_string()

    @classmethod
    def from_ec_nc_file(cls, ec_nc_file, tensor_type='relaxed_ion'):
        with NetcdfReader(ec_nc_file) as nc_reader:
            if tensor_type == 'relaxed_ion':
                ec = np.array(nc_reader.read_variable('elastic_constants_relaxed_ion'))
                compl = np.array(nc_reader.read_variable('compliance_constants_relaxed_ion'))
            elif tensor_type == 'clamped_ion':
                ec = np.array(nc_reader.read_variable('elastic_constants_clamped_ion'))
                compl = np.array(nc_reader.read_variable('compliance_constants_clamped_ion'))
            elif tensor_type == 'relaxed_ion_stress_corrected':
                ec = np.array(nc_reader.read_variable('elastic_constants_relaxed_ion_stress_corrected'))
                compl = np.array(nc_reader.read_variable('compliance_constants_relaxed_ion_stress_corrected'))
            else:
                raise ValueError('tensor_type "{0}" not allowed'.format(tensor_type))
        #TODO: add the structure object!
        return cls(elastic_tensor=ec, compliance_tensor=compl, structure=None,
                   additional_info={'tensor_type': tensor_type})

    def as_dict(self):
        return {'elastic_tensor': self.elastic_tensor, 'compliance_tensor': self.compliance_tensor,
                'structure': self.structure.as_dict() if self.structure is not None else None,
                'additional_info': self.additional_info}

    def extended_dict(self):
        dd = self.as_dict()
        K_Voigt = (self.elastic_tensor[0, 0] + self.elastic_tensor[1, 1] + self.elastic_tensor[2, 2] +
                   2.0*self.elastic_tensor[0, 1] + 2.0*self.elastic_tensor[1, 2] + 2.0*self.elastic_tensor[2, 0]) / 9.0
        K_Reuss = 1.0 / (self.compliance_tensor[0, 0] + self.compliance_tensor[1, 1] + self.compliance_tensor[2, 2] +
                         2.0*self.compliance_tensor[0, 1] + 2.0*self.compliance_tensor[1, 2] +
                         2.0*self.compliance_tensor[2, 0])
        G_Voigt = (self.elastic_tensor[0, 0] + self.elastic_tensor[1, 1] + self.elastic_tensor[2, 2] -
                   self.elastic_tensor[0, 1] - self.elastic_tensor[1, 2] - self.elastic_tensor[2, 0] +
                   3.0*self.elastic_tensor[3, 3] + 3.0*self.elastic_tensor[4, 4] + 3.0*self.elastic_tensor[5, 5]) / 15.0
        G_Reuss = 15.0 / (4.0*self.compliance_tensor[0, 0] + 4.0*self.compliance_tensor[1, 1] +
                          4.0*self.compliance_tensor[2, 2] - 4.0*self.compliance_tensor[0, 1] -
                          4.0*self.compliance_tensor[1, 2] - 4.0*self.compliance_tensor[2, 0] +
                          3.0*self.compliance_tensor[3, 3] + 3.0*self.compliance_tensor[4, 4] +
                          3.0*self.compliance_tensor[5, 5])
        K_VRH = (K_Voigt + K_Reuss) / 2.0
        G_VRH = (G_Voigt + G_Reuss) / 2.0
        universal_elastic_anisotropy = 5.0*G_Voigt/G_Reuss + K_Voigt/K_Reuss - 6.0
        isotropic_poisson_ratio = (3.0*K_VRH - 2.0*G_VRH) / (6.0*K_VRH + 2.0*G_VRH)
        dd['K_Voigt'] = K_Voigt
        dd['G_Voigt'] = G_Voigt
        dd['K_Reuss'] = K_Reuss
        dd['G_Reuss'] = G_Reuss
        dd['K_VRH'] = K_VRH
        dd['G_VRH'] = G_VRH
        dd['universal_elastic_anistropy'] = universal_elastic_anisotropy
        dd['isotropic_poisson_ratio'] = isotropic_poisson_ratio
        return dd

    @classmethod
    def from_dict(cls, dd):
        return cls(elastic_tensor=dd['elastic_tensor'], compliance_tensor=dd['compliance_tensor'],
                   structure=dd['structure'] if dd['structure'] is not None else None,
                   additional_info=dd['additional_info'])

    def get_pmg_elastic_tensor(self):
        """
        Converts to a pymatgen :class:`ElasticTensor` object.
        """
        return ElasticTensor.from_voigt(self.elastic_tensor)
