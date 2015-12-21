from __future__ import print_function, division, unicode_literals

import os
import shutil
import logging
import collections
import errno
import threading
import subprocess
from monty.json import MSONable
from abiflows.fireworks.tasks.src_tasks_abc import SetupTask, RunTask, ControlTask, SetupError
from abiflows.fireworks.utils.fw_utils import FWTaskManager
from abiflows.fireworks.tasks.abinit_common import TMPDIR_NAME, OUTDIR_NAME, INDIR_NAME, STDERR_FILE_NAME, \
    LOG_FILE_NAME, FILES_FILE_NAME, OUTPUT_FILE_NAME, INPUT_FILE_NAME, MPIABORTFILE, DUMMY_FILENAME, \
    ELPHON_OUTPUT_FILE_NAME, DDK_FILES_FILE_NAME, HISTORY_JSON
from fireworks import explicit_serialize
from pymatgen.serializers.json_coders import json_pretty_dump, pmg_serialize
from pymatgen.io.abinit.utils import Directory, File
from pymatgen.io.abinit.utils import irdvars_for_ext
from pymatgen.io.abinit import events
from pymatgen.io.abinit.qutils import time2slurm
from abipy.abio.factories import InputFactory
from abipy.abio.inputs import AbinitInput


logger = logging.getLogger(__name__)

class AbinitSRCMixin(object):

    # Prefixes for Abinit (input, output, temporary) files.
    Prefix = collections.namedtuple("Prefix", "idata odata tdata")
    pj = os.path.join

    prefix = Prefix(pj("indata", "in"), pj("outdata", "out"), pj("tmpdata", "tmp"))
    del Prefix, pj

    def get_fw_task_manager(self, fw_spec):
        if 'ftm_file' in fw_spec:
            ftm = FWTaskManager.from_file(fw_spec['ftm_file'])
        else:
            ftm = FWTaskManager.from_user_config()
        ftm.update_fw_policy(fw_spec.get('fw_policy', {}))
        return ftm

    def setup_rundir(self, rundir, create_dirs=False):
        """Set the run directory."""

        # Files required for the execution.
        self.input_file = File(os.path.join(rundir, INPUT_FILE_NAME))
        self.output_file = File(os.path.join(rundir, OUTPUT_FILE_NAME))
        self.files_file = File(os.path.join(rundir, FILES_FILE_NAME))
        self.log_file = File(os.path.join(rundir, LOG_FILE_NAME))
        self.stderr_file = File(os.path.join(rundir, STDERR_FILE_NAME))

        # This file is produce by Abinit if nprocs > 1 and MPI_ABORT.
        self.mpiabort_file = File(os.path.join(rundir, MPIABORTFILE))

        # Directories with input|output|temporary data.
        self.indir = Directory(os.path.join(rundir, INDIR_NAME))
        self.outdir = Directory(os.path.join(rundir, OUTDIR_NAME))
        self.tmpdir = Directory(os.path.join(rundir, TMPDIR_NAME))

        if create_dirs:
            # Create dirs for input, output and tmp data.
            self.indir.makedirs()
            self.outdir.makedirs()
            self.tmpdir.makedirs()


@explicit_serialize
class AbinitSetupTask(AbinitSRCMixin, SetupTask):

    RUN_PARAMETERS = ['_queueadapter', 'qtk_queueadapter']

    def __init__(self, abiinput, deps=None, task_helper=None, restart_info=None):
        SetupTask.__init__(self, restart_info=restart_info)
        self.abiinput = abiinput

        # deps are transformed to be a list or a dict of lists
        if isinstance(deps, dict):
            deps = dict(deps)
            for k, v in deps.items():
                if not isinstance(v, (list, tuple)):
                    deps[k] = [v]
        elif deps and not isinstance(deps, (list, tuple)):
            deps = [deps]
        self.deps = deps

        self.task_helper = task_helper
        self.task_helper.set_task = self

    def set_restart_info(self, restart_info=None):
        self.restart_info = restart_info

    def run_task(self, fw_spec):
        #TODO create a initialize_setup abstract function in SetupTask and put it there? or move somewhere else?
        #setup the FWTaskManager
        self.ftm = self.get_fw_task_manager(fw_spec)
        return super(AbinitSetupTask, self).run_task(fw_spec)

    def setup_run_parameters(self, fw_spec, parameters=RUN_PARAMETERS):
        optconf, qtk_qadapter = self.run_autoparal(self.abiinput, fw_spec)

        # if 'queue_adapter_update' in fw_spec:
        #     for qa_key, qa_val in fw_spec['queue_adapter_update'].items():
        #         if qa_key == 'timelimit':
        #             qtk_qadapter.set_timelimit(qa_val)
        #         elif qa_key == 'mem_per_proc':
        #             qtk_qadapter.set_mem_per_proc(qa_val)
        #         elif qa_key == 'master_mem_overhead':
        #             qtk_qadapter.set_master_mem_overhead(qa_val)
        #         else:
        #             raise ValueError('queue_adapter update "{}" is not valid'.format(qa_key))

        return {'_queueadapter': qtk_qadapter.get_subs_dict(), 'qtk_queueadapter': qtk_qadapter}

    def file_transfers(self, fw_spec):
        pass

    def prepare_run(self, fw_spec):
        # if the input is a factory, dynamically create the abinit input. From now on the code will expect an
        # AbinitInput and not a factory. In this case there should be either a single input coming from the previous
        # fws or a deps specifying which input use
        if isinstance(self.abiinput, InputFactory):
            #FIXME save initialization_info somewhere. What to do with the TaskHistory?
            # initialization_info['input_factory'] = self.abiinput
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
                        raise SetupError(msg)
                    task_type_source = previous_fws.keys()[0]
                # the task_type_source should contain just one task and contain the 'input' key
                if len(previous_fws[task_type_source]) != 1 or not previous_fws[task_type_source][0].get('input', None):
                    msg = 'The factory {} requires the input from previous run in the spec'.format(self.abiinput.__class__)
                    logger.error(msg)
                    raise SetupError(msg)
                # a single input exists
                previous_input = previous_fws[task_type_source][0]['input']
                if not isinstance(previous_input, AbinitInput):
                    previous_input = AbinitInput.from_dict(previous_input)
                # initialization_info['previous_input'] = previous_input

            self.abiinput = self.abiinput.build_input(previous_input)

        # initialization_info['initial_input'] = self.abiinput

        # if it's the first run log the initialization of the task
        # if len(self.history) == 0:
        #     self.history.log_initialization(self, initialization_info)

        #TODO check if keeping this in the helpers or removing
        # update data from previous run if it is not a restart
        # if 'previous_fws' in fw_spec and not self.restart_info:
        #     self.load_previous_fws_data(fw_spec)

        self.setup_rundir(rundir=self.run_dir, create_dirs=True)

        # Copy the appropriate dependencies in the in dir
        self.resolve_deps(fw_spec)

        # if it's the restart of a previous task, perform specific task updates.
        # perform these updates before writing the input, but after creating the dirs.
        if self.restart_info:
            #TODO check if this is the correct way of doing the restart
            # self.history.log_restart(self.restart_info)
            self.task_helper.restart(self.restart_info)

        # Write files file and input file.
        if not self.files_file.exists:
            self.files_file.write(self.filesfile_string)

        self.input_file.write(str(self.abiinput))

    def run_autoparal(self, abiinput, fw_spec, clean_up='partial'):
        """
        Runs the autoparal using AbinitInput abiget_autoparal_pconfs method.
        The information are retrieved from the FWTaskManager that should be present and contain the standard
        abipy TaskManager, that provides information about the queue adapters.
        """
        #FIXME autoparal may need the deps in some cases. here they are not resolved
        manager = self.ftm.task_manager
        if not manager:
            msg = 'No task manager available: autoparal could not be performed.'
            logger.error(msg)
            raise SetupError(msg)

        autoparal_dir = 'run_autoparal'
        pconfs = abiinput.abiget_autoparal_pconfs(max_ncpus=manager.max_cores, workdir=autoparal_dir,
                                                       manager=manager)
        optconf = manager.select_qadapter(pconfs)

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

        # clean up useless files.
        if clean_up == 'partial':
            to_be_removed = [TMPDIR_NAME, OUTDIR_NAME, INDIR_NAME]
            for r in to_be_removed:
                safe_rm(os.path.join(autoparal_dir, r))
        elif clean_up == 'full':
            shutil.rmtree(autoparal_dir)

        return optconf, manager.qadapter

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
                raise SetupError(msg)
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
            raise SetupError(msg)
        exts = [os.path.basename(ddk).split('_')[-2] for ddk in ddks]
        ddk_files = []
        for ext in exts:
            ddk_files.append(self.link_ext(ext, source_dir))

        return ddk_files


    def resolve_deps_per_task_type(self, previous_tasks, deps_list):
        for previous_task in previous_tasks:
            for d in deps_list:
                if d.startswith('@structure'):
                    if 'structure' not in previous_task:
                        msg = "previous_fws does not contain the structure."
                        logger.error(msg)
                        raise SetupError(msg)
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
                raise SetupError(msg)

            if isinstance(self.deps, (list, tuple)):
                # check that there is only one previous_fws
                if len(previous_fws) != 1 or len(previous_fws.values()[0]) != 1:
                    msg = "previous_fws does not contain a single reference. " \
                          "Specify the dependency for {}.".format(str(self.deps))
                    logger.error(msg)
                    raise SetupError(msg)

                self.resolve_deps_per_task_type(previous_fws.values()[0], self.deps)

            else:
                # deps should be a dict
                for task_type, deps_list in self.deps.items():
                    if task_type not in previous_fws:
                        msg = "No previous_fws data for task type {}.".format(task_type)
                        logger.error(msg)
                        raise SetupError(msg)
                    if len(previous_fws[task_type]) < 1:
                        msg = "Previous_fws does not contain any reference for task type {}, " \
                              "needed in reference {}. ".format(task_type, str(self.deps))
                        logger.error(msg)
                        raise SetupError(msg)
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

    @property
    def filesfile_string(self):
        """String with the list of files and prefixes needed to execute ABINIT."""
        lines = []
        app = lines.append
        pj = os.path.join

        app(self.input_file.path)                 # Path to the input file
        app(self.output_file.path)                # Path to the output file
        app(pj(self.run_dir, self.prefix.idata))  # Prefix for input data
        app(pj(self.run_dir, self.prefix.odata))  # Prefix for output data
        app(pj(self.run_dir, self.prefix.tdata))  # Prefix for temporary data

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

    #TODO move this functions to AbinitTaskHelper? they will probably be called just by them
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
        if self.ftm.fw_policy.copy_deps:
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


@explicit_serialize
class AbinitRunTask(AbinitSRCMixin, RunTask):


    def __init__(self, control_procedure, task_helper):
        RunTask.__init__(self, control_procedure=control_procedure)
        self.task_helper = task_helper

    def config(self, fw_spec):
        self.ftm = self.get_fw_task_manager(fw_spec)
        self.setup_rundir(self.run_dir, create_dirs=False)

    def run(self, fw_spec):
        #TODO switch back to a simple process instead of a separate thread?
        def abinit_process():
            command = []
            #consider the case of serial execution
            if self.ftm.fw_policy.mpirun_cmd:
                command.extend(self.ftm.fw_policy.mpirun_cmd.split())
                if 'mpi_ncpus' in fw_spec:
                    command.extend(['-np', str(fw_spec['mpi_ncpus'])])
            command.append(self.ftm.fw_policy.abinit_cmd)
            mytimelimit = fw_spec['qtk_queueadapter'].timelimit-self.ftm.fw_policy.timelimit_buffer
            if mytimelimit < 120:
                raise ValueError('Abinit timelimit less than 2 min. Probably wrong queue/job configuration')
            command.extend(['--timelimit', time2slurm(mytimelimit)])
            with open(self.files_file.path, 'r') as stdin, open(self.log_file.path, 'w') as stdout, \
                    open(self.stderr_file.path, 'w') as stderr:
                self.process = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=stderr)

            (stdoutdata, stderrdata) = self.process.communicate()
            self.returncode = self.process.returncode

        # initialize returncode to avoid missing references in case of exception in the other thread
        self.returncode = None

        thread = threading.Thread(target=abinit_process)
        thread.start()
        thread.join()

    def postrun(self, fw_spec):
        #TODO should this be a general feature of the SRC?
        return {'qtk_queueadapter' :fw_spec['qtk_queueadapter']}


@explicit_serialize
class AbinitControlTask(AbinitSRCMixin, ControlTask):

    def __init__(self, control_procedure, manager=None, max_restarts=10, task_helper=None):
        ControlTask.__init__(self, control_procedure=control_procedure, manager=manager, max_restarts=max_restarts)
        self.task_helper = task_helper

    def get_initial_objects_info(self):
        return {'abinit_input': {'object': self.setup_fw.tasks[-1].abiinput,
                                 'updates': [{'target': 'setup_task',
                                              'attribute': 'abiinput'}]},
                'abinit_output_filepath': {'object': os.path.join(self.run_dir, OUTPUT_FILE_NAME)},
                'abinit_log_filepath': {'object': os.path.join(self.run_dir, LOG_FILE_NAME)},
                'abinit_mpi_abort_filepath': {'object': os.path.join(self.run_dir, MPIABORTFILE)},
                'abinit_outdir_path': {'object': os.path.join(self.run_dir, OUTDIR_NAME)}}

            # initial_objects_info.update({'queue_adapter': {'object': self.run_fw.spec['qtk_queueadapter'],
            #                                            'updates': [{'target': 'fw_spec',
            #                                                         'key': 'qtk_queueadapter'},
            #                                                        {'target': 'fw_spec',
            #                                                         'key': '_queueadapter',
            #                                                         'mod': 'get_subs_dict'}]},


####################
# Helpers
####################

class AbinitTaskHelper(MSONable):
    #TODO add the task_type as an init parameter?
    task_type = 'abinit'

    CRITICAL_EVENTS = []

    def __init__(self):
        self.task = None

    def set_task(self, task):
        self.task = task

    def restart(self, restart_info):
        """
        Restart method. Each subclass should implement its own restart.
        """
        pass

    def prepare_restart(self):
        pass

    def conclude_task(self):
        pass

    def additional_update_spec(self):
        pass

    @pmg_serialize
    def as_dict(self):
        return {}

    @classmethod
    def from_dict(cls, d):
        return cls()


class GsTaskHelper(AbinitTaskHelper):
    @property
    def gsr_path(self):
        """Absolute path of the GSR file. Empty string if file is not present."""
        # Lazy property to avoid multiple calls to has_abiext.
        try:
            return self._gsr_path
        except AttributeError:
            path = self.task.outdir.has_abiext("GSR")
            if path: self._gsr_path = path
            return path

    def open_gsr(self):
        """
        Open the GSR file located in the in the task outdir.
        Returns :class:`GsrFile` object, raise a HelperError exception if file could not be found or file is not readable.
        """
        gsr_path = self.gsr_path
        if not gsr_path:
            msg = "No GSR file available for task {} in {}".format(self, self.task.outdir)
            logger.critical(msg)
            raise HelperError(msg)

        # Open the GSR file.
        from abipy.electrons.gsr import GsrFile
        try:
            return GsrFile(gsr_path)
        except Exception as exc:
            msg = "Exception while reading GSR file at %s:\n%s" % (gsr_path, str(exc))
            logger.critical(msg)
            raise HelperError(msg)


class ScfTaskHelper(GsTaskHelper):
    task_type = "scf"

    CRITICAL_EVENTS = [
        events.ScfConvergenceWarning,
    ]

    def restart(self, restart_info):
        """SCF calculations can be restarted if we have either the WFK file or the DEN file."""
        # Prefer WFK over DEN files since we can reuse the wavefunctions.
        if restart_info.reset:
            # remove non reset keys that may have been added in a previous restart
            self.task.remove_restart_vars(["WFK", "DEN"])
        else:
            for ext in ("WFK", "DEN"):
                restart_file = self.task.prev_outdir.has_abiext(ext)
                irdvars = irdvars_for_ext(ext)
                if restart_file: break
            else:
                msg = "Cannot find WFK or DEN file to restart from."
                logger.error(msg)
                raise RestartError(msg)

            # Move out --> in.
            self.task.out_to_in(restart_file)

            # Add the appropriate variable for restarting.
            self.task.abiinput.set_vars(irdvars)



class NscfTaskHelper(GsTaskHelper):
    pass


class RelaxTaskHelper(GsTaskHelper):
    pass


class DfptTaskHelper(AbinitTaskHelper):
    pass


class DdkTaskHelper(DfptTaskHelper):
    pass


class DdeTaskHelper(DfptTaskHelper):
    pass


class PhononTaskHelper(DfptTaskHelper):
    pass


class BecTaskHelper(DfptTaskHelper):
    pass


class StrainPertTaskHelper(DfptTaskHelper):
    pass


####################
# Exceptions
####################

class HelperError(Exception):
    pass


class RestartError(Exception):
    pass

##############################
# Other objects
##############################

class RestartInfo(MSONable):
    """
    Object that contains the information about the restart of a task.
    """
    def __init__(self, previous_dir, reset=False):
        self.previous_dir = previous_dir
        self.reset = reset

    @pmg_serialize
    def as_dict(self):
        return dict(previous_dir=self.previous_dir, reset=self.reset)

    @classmethod
    def from_dict(cls, d):
        return cls(previous_dir=d['previous_dir'], reset=d['reset'])

    @property
    def prev_outdir(self):
        return Directory(os.path.join(self.previous_dir, OUTDIR_NAME))

    @property
    def prev_indir(self):
        return Directory(os.path.join(self.previous_dir, INDIR_NAME))