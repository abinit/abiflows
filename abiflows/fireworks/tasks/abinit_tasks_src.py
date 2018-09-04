from __future__ import print_function, division, unicode_literals, absolute_import

import os
import shutil
import inspect
import logging
import collections
import errno
import threading
import subprocess
import time

from monty.json import MSONable
from monty.json import MontyDecoder
from monty.os.path import which
from pymatgen.util.serialization import json_pretty_dump, pmg_serialize
from pymatgen.io.abinit.utils import Directory, File
from pymatgen.io.abinit.utils import irdvars_for_ext
from pymatgen.io.abinit import events
from pymatgen.io.abinit.qutils import time2slurm
from abipy.abio.factories import InputFactory
from abipy.abio.factories import PiezoElasticFromGsFactory
from abipy.abio.inputs import AbinitInput, Cut3DInput
from abipy.abio.input_tags import STRAIN, GROUND_STATE, NSCF, BANDS, PHONON
from abipy.abio.outputs import OutNcFile
from abipy.electrons.gsr import GsrFile
from abipy.electrons.charges import HirshfeldCharges
from abipy.flowtk.netcdf import NetcdfReader, NO_DEFAULT
from fireworks import explicit_serialize
from fireworks.utilities.fw_serializers import serialize_fw
from fireworks.core.firework import Firework, FireTaskBase, FWAction, Workflow
from abiflows.fireworks.tasks.src_tasks_abc import SetupTask, RunTask, ControlTask, SetupError, createSRCFireworks
from abiflows.core.mastermind_abc import ControllerNote, ControlProcedure
from abiflows.core.controllers import AbinitController, WalltimeController, MemoryController
from abiflows.fireworks.utils.fw_utils import FWTaskManager, links_dict_update, set_short_single_core_to_spec
from abiflows.fireworks.utils.math_utils import divisors
from abiflows.fireworks.tasks.abinit_tasks import MergeDdbAbinitTask
from abiflows.fireworks.tasks.abinit_common import TMPDIR_NAME, OUTDIR_NAME, INDIR_NAME, STDERR_FILE_NAME, \
    LOG_FILE_NAME, FILES_FILE_NAME, OUTPUT_FILE_NAME, INPUT_FILE_NAME, MPIABORTFILE, DUMMY_FILENAME, \
    ELPHON_OUTPUT_FILE_NAME, DDK_FILES_FILE_NAME, HISTORY_JSON


RESET_RESTART = ControllerNote.RESET_RESTART
SIMPLE_RESTART = ControllerNote.SIMPLE_RESTART
RESTART_FROM_SCRATCH = ControllerNote.RESTART_FROM_SCRATCH

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

    def setup_rundir(self, rundir, create_dirs=False, directories_only=False):
        """Set the run directory."""

        # Files required for the execution.
        if not directories_only:
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

    def __init__(self, abiinput, deps=None, task_helper=None, task_type=None, restart_info=None, pass_input=False):
        if task_type is None:
            task_type = task_helper.task_type
        SetupTask.__init__(self, deps=deps, restart_info=restart_info, task_type=task_type)
        self.abiinput = abiinput
        self.pass_input = pass_input

        # # deps are transformed to be a list or a dict of lists
        # if isinstance(deps, dict):
        #     deps = dict(deps)
        #     for k, v in deps.items():
        #         if not isinstance(v, (list, tuple)):
        #             deps[k] = [v]
        # elif deps and not isinstance(deps, (list, tuple)):
        #     deps = [deps]
        # self.deps = deps

        self.task_helper = task_helper
        self.task_helper.set_task(self)

    def setup_directories(self, fw_spec, create_dirs=False):
        SetupTask.setup_directories(self, fw_spec=fw_spec, create_dirs=create_dirs)
        self.setup_rundir(rundir=self.run_dir, create_dirs=create_dirs)

    def run_task(self, fw_spec):
        #TODO create a initialize_setup abstract function in SetupTask and put it there? or move somewhere else?
        #setup the FWTaskManager
        self.ftm = self.get_fw_task_manager(fw_spec)
        if 'previous_src' in fw_spec:
            self.prev_outdir = Directory(os.path.join(fw_spec['previous_src']['src_directories']['run_dir'],
                                                      OUTDIR_NAME))
        return super(AbinitSetupTask, self).run_task(fw_spec)

    def setup_run_parameters(self, fw_spec, parameters=RUN_PARAMETERS):
        self.abiinput.remove_vars(['npkpt', 'npspinor', 'npband', 'npfft', 'bandpp'], strict=False)
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
        self.abiinput.set_vars(optconf.vars)

        return {'_queueadapter': qtk_qadapter.get_subs_dict(), 'qtk_queueadapter': qtk_qadapter}

    def file_transfers(self, fw_spec):
        pass

    def fetch_previous_info(self, fw_spec):
        # Copy the appropriate dependencies in the in dir
        self.resolve_deps(fw_spec)

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

        # THE FOLLOWING HAS BEEN MOVED TO setup_directories
        # self.setup_rundir(rundir=self.run_dir, create_dirs=True)

        # THE FOLLOWING HAS BEEN MOVED TO fetch_previous_info
        # Copy the appropriate dependencies in the in dir
        # self.resolve_deps(fw_spec)

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
        # Set npfft by hand in autoparal if ngfft is found ...
        if 'ngfft' in abiinput:
            if GROUND_STATE in abiinput.runlevel or NSCF in abiinput.runlevel:
                npfft_set = {1, 2, 3, 4, 5, 6}
                ngfft = abiinput['ngfft']
                divsy = divisors(ngfft[1])
                divsz = divisors(ngfft[2])
                npfft_set = npfft_set.intersection(set(divsy))
                npfft_set = npfft_set.intersection(set(divsz))
                if abiinput.ispaw:
                    ngfftdg = abiinput['ngfftdg']
                    divsy = divisors(ngfftdg[1])
                    divsz = divisors(ngfftdg[2])
                    npfft_set = npfft_set.intersection(set(divsy))
                    npfft_set = npfft_set.intersection(set(divsz))
                abiinput['npfft'] = max(npfft_set)
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
        dest = os.path.join(self.run_dir, self.prefix.idata + "_" + ext)
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
                    source_dir = previous_task['dir']
                    gsr_file = GsrFile(os.path.join(source_dir, 'outdata', 'out_GSR.nc'))
                    self.abiinput.set_structure(gsr_file.structure)
                    # if 'structure' not in previous_task:
                    #     msg = "previous_fws does not contain the structure."
                    #     logger.error(msg)
                    #     raise SetupError(msg)
                    # self.abiinput.set_structure(previous_task['structure'])
                elif d.startswith('@outnc') or d.startswith('#outnc'):
                    varname = d.split('.')[1]
                    outnc_path = os.path.join(previous_task['dir'], self.prefix.odata + "_OUT.nc")
                    outnc_file = _AbinitOutNcFile(outnc_path)
                    vars = outnc_file.get_vars(vars=[varname], strict=True)
                    self.abiinput.set_vars(vars)
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

        # If this is the first run of the task, the informations are taken from the 'previous_fws',
        # that should be present.
        previous_fws = fw_spec.get('previous_fws', None)
        if previous_fws is None:
            msg = "No previous_fws data. Needed for dependecies {}.".format(str(self.deps))
            logger.error(msg)
            raise SetupError(msg)

        if not self.restart_info:

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
            # if self.restart_info.previous_dir == self.run_dir:
            previous_run_dir = fw_spec['previous_src']['src_directories']['run_dir']
            #TODO: remove this ?
            if previous_run_dir == self.run_dir:
                logger.info('rerunning in the same dir, no action on the deps')
                return

            #just link everything from the indata folder of the previous run. files needed for restart will be overwritten

            prev_indata = os.path.join(previous_run_dir, INDIR_NAME)
            for f in os.listdir(prev_indata):
                # if the target is already a link, link to the source to avoid many nested levels of linking
                source = os.path.join(prev_indata, f)
                if os.path.islink(source):
                    source = os.readlink(source)
                os.symlink(source, os.path.join(self.run_dir, INDIR_NAME, f))

            # Resolve the dependencies that start with '#'
            if isinstance(self.deps, (list, tuple)):
                # check that there is only one previous_fws
                if len(previous_fws) != 1 or len(previous_fws.values()[0]) != 1:
                    msg = "previous_fws does not contain a single reference. " \
                          "Specify the dependency for {}.".format(str(self.deps))
                    logger.error(msg)
                    raise SetupError(msg)
                deps = [dep for dep in self.deps if dep[0] == '#']
                self.resolve_deps_per_task_type(previous_fws.values()[0], deps)

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

                    deps_list = [dep for dep in deps_list if dep[0] == '#']
                    self.resolve_deps_per_task_type(previous_fws[task_type], deps_list)

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
            except OSError as e:
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

    def additional_task_info(self):
        if self.pass_input:
            return {'input': self.abiinput}
        else:
            return {}


@explicit_serialize
class AbinitRunTask(AbinitSRCMixin, RunTask):


    def __init__(self, control_procedure, task_helper, task_type=None):
        if task_type is None:
            task_type = task_helper.task_type
        RunTask.__init__(self, control_procedure=control_procedure, task_type=task_type)
        self.task_helper = task_helper
        self.task_helper.set_task(self)

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

        if os.path.isfile('run.log'):
            from pymatgen.io.abinit.events import EventsParser
            parser = EventsParser()
            report = parser.parse('run.log')
            if report.run_completed:
                self.returncode = 0
                return

        # initialize returncode to avoid missing references in case of exception in the other thread
        self.returncode = None

        thread = threading.Thread(target=abinit_process)
        thread.start()
        thread.join()

    def postrun(self, fw_spec):
        #TODO should this be a general feature of the SRC?
        self.task_helper.conclude_task()
        return {'qtk_queueadapter' :fw_spec['qtk_queueadapter']}


@explicit_serialize
class AbinitControlTask(AbinitSRCMixin, ControlTask):

    def __init__(self, control_procedure, manager=None, max_restarts=10, src_cleaning=None, task_helper=None):
        ControlTask.__init__(self, control_procedure=control_procedure, manager=manager, max_restarts=max_restarts,
                             src_cleaning=src_cleaning)
        self.task_helper = task_helper

    def get_initial_objects_info(self, setup_fw, run_fw, src_directories):
        run_dir = src_directories['run_dir']
        run_task = run_fw.tasks[-1]
        run_task.setup_rundir(rundir=run_dir, create_dirs=False, directories_only=False)
        task_helper = run_task.task_helper
        task_helper.set_task(run_task)
        init_obj_info = {'abinit_input': {'object': setup_fw.tasks[-1].abiinput,
                                          'updates': [{'target': 'setup_task',
                                                       'attribute': 'abiinput'}]},
                         'abinit_output_filepath': {'object': os.path.join(run_dir, OUTPUT_FILE_NAME)},
                         'abinit_log_filepath': {'object': os.path.join(run_dir, LOG_FILE_NAME)},
                         'abinit_mpi_abort_filepath': {'object': os.path.join(run_dir, MPIABORTFILE)},
                         'abinit_outdir_path': {'object': os.path.join(run_dir, OUTDIR_NAME)},
                         'abinit_err_filepath': {'object': os.path.join(run_dir, STDERR_FILE_NAME)}}
        # 'structure': {'object': task_helper.get_final_structure(),
        #               'updates': [{'target': 'setup_task.abiinput',
        #                            'setter': 'set_structure'}]}}
        if hasattr(task_helper, 'get_final_structure'):
            try:
                final_structure = task_helper.get_final_structure()
            except HelperError:
                final_structure = None
            except PostProcessError:
                final_structure = None
            init_obj_info['structure'] = {'object': final_structure,
                                          'updates': [{'target': 'setup_task.abiinput',
                                                       'setter': 'set_structure'}]}
        return init_obj_info

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
        if restart_info == RESET_RESTART:
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
    task_type = "nscf"

    CRITICAL_EVENTS = [
        events.NscfConvergenceWarning,
    ]

    def restart(self, restart_info):
        """NSCF calculations can be restarted only if we have the WFK file."""
        if restart_info == RESET_RESTART:
            # remove non reset keys that may have been added in a previous restart
            self.task.remove_restart_vars(["WFK"])
        else:
            ext = "WFK"
            restart_file = self.task.prev_outdir.has_abiext(ext)
            if not restart_file:
                msg = "Cannot find the WFK file to restart from."
                logger.error(msg)
                raise RestartError(msg)

            # Move out --> in.
            self.task.out_to_in(restart_file)

            # Add the appropriate variable for restarting.
            irdvars = irdvars_for_ext(ext)
            self.task.abiinput.set_vars(irdvars)


class RelaxTaskHelper(GsTaskHelper):
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
            msg = "Cannot find the GSR file with the final structure."
            logger.error(msg)
            raise PostProcessError(msg)

    def get_computed_entry(self):
        """Get the computed entry from the GSR file."""
        try:
            with self.open_gsr() as gsr:
                return gsr.get_computed_entry(inc_structure=True)
        except AttributeError:
            msg = "Cannot find the GSR file with the information to get the computed entry."
            logger.error(msg)
            raise PostProcessError(msg)

    # def prepare_restart(self):
    #     self.task.abiinput.set_structure(self.get_final_structure())
    #
    #     return super(RelaxTaskHelper, self).prepare_restart()

    def restart(self, restart_info):
        """
        Restart the structural relaxation.

        See original RelaxTask for more details
        """

        if restart_info == RESET_RESTART:
            # remove non reset keys that may have been added in a previous restart
            self.task.remove_restart_vars(["WFK", "DEN"])
        else:
            # for optcell > 0 it may fail to restart if paral_kgb == 0. Do not use DEN or WFK in this case
            #FIXME fix when Matteo makes the restart possible for paral_kgb == 0
            paral_kgb = self.task.abiinput.get('paral_kgb', 0)
            optcell = self.task.abiinput.get('optcell', 0)

            if optcell == 0 or paral_kgb == 1:
                restart_file = None

                # Try to restart from the WFK file if possible.
                # FIXME: This part has been disabled because WFK=IO is a mess if paral_kgb == 1
                # This is also the reason why I wrote my own MPI-IO code for the GW part!
                wfk_file = self.task.prev_outdir.has_abiext("WFK")
                if False and wfk_file:
                    irdvars = irdvars_for_ext("WFK")
                    restart_file = self.task.out_to_in(wfk_file)

                # Fallback to DEN file. Note that here we look for out_DEN instead of out_TIM?_DEN
                # This happens when the previous run completed and task.on_done has been performed.
                # ********************************************************************************
                # Note that it's possible to have an undected error if we have multiple restarts
                # and the last relax died badly. In this case indeed out_DEN is the file produced
                # by the last run that has executed on_done.
                # ********************************************************************************
                if restart_file is None:
                    out_den = self.task.prev_outdir.path_in("out_DEN")
                    if os.path.exists(out_den):
                        irdvars = irdvars_for_ext("DEN")
                        restart_file = self.task.out_to_in(out_den)

                if restart_file is None:
                    # Try to restart from the last TIM?_DEN file.
                    # This should happen if the previous run didn't complete in clean way.
                    # Find the last TIM?_DEN file.
                    last_timden = self.task.prev_outdir.find_last_timden_file()
                    if last_timden is not None:
                        ofile = self.task.prev_outdir.path_in("out_DEN")
                        os.rename(last_timden.path, ofile)
                        restart_file = self.task.out_to_in(ofile)
                        irdvars = irdvars_for_ext("DEN")

                if restart_file is None:
                    # Don't raise RestartError as the structure has been updated
                    logger.warning("Cannot find the WFK|DEN|TIM?_DEN file to restart from.")
                else:
                    # Add the appropriate variable for restarting.
                    self.task.abiinput.set_vars(irdvars)
                    logger.info("Will restart from %s", restart_file)

    # def current_task_info(self, fw_spec):
    #     d = super(RelaxTaskHelper, self).current_task_info(fw_spec)
    #     d['structure'] = self.get_final_structure()
    #     return d

    # def conclude_task(self, fw_spec):
    #     update_spec, mod_spec, stored_data = super(RelaxFWTask, self).conclude_task(fw_spec)
    #     update_spec['previous_run']['structure'] = self.get_final_structure()
    #     return update_spec, mod_spec, stored_data


class DfptTaskHelper(AbinitTaskHelper):
    task_type = "dfpt"

    CRITICAL_EVENTS = [
        events.ScfConvergenceWarning,
    ]

    def restart(self, restart_info):
        """
        Phonon calculations can be restarted only if we have the 1WF file or the 1DEN file.
        from which we can read the first-order wavefunctions or the first order density.
        Prefer 1WF over 1DEN since we can reuse the wavefunctions.
        Try to handle an input with many perturbation calculated at the same time. link/copy all the 1WF or 1DEN files
        """
        # Abinit adds the idir-ipert index at the end of the file and this breaks the extension
        # e.g. out_1WF4, out_DEN4. find_1wf_files and find_1den_files returns the list of files found
        #TODO check for reset
        if restart_info == RESET_RESTART:
            # remove non reset keys that may have been added in a previous restart
            self.task.remove_restart_vars(["1WF", "1DEN"])
        else:
            restart_files, irdvars = None, None

            # Highest priority to the 1WF file because restart is more efficient.
            wf_files = self.task.prev_outdir.find_1wf_files()
            if wf_files is not None:
                restart_files = [f.path for f in wf_files]
                irdvars = irdvars_for_ext("1WF")
                # if len(wf_files) != 1:
                #     restart_files = None
                #     logger.critical("Found more than one 1WF file. Restart is ambiguous!")

            if restart_files is None:
                den_files = self.task.prev_outdir.find_1den_files()
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
            for restart_file in restart_files:
                self.task.out_to_in(restart_file)

            # Add the appropriate variable for restarting.
            self.task.abiinput.set_vars(irdvars)


class DdkTaskHelper(DfptTaskHelper):

    task_type = "ddk"

    def conclude_task(self):
        # make a link to _DDK of the 1WF file to ease the link in the dependencies
        wf_files = self.task.outdir.find_1wf_files()
        if not wf_files:
            raise HelperError("Couldn't link 1WF files.")
        for f in wf_files:
            os.symlink(f.path, f.path+'_DDK')

        super(DdkTaskHelper, self).conclude_task()


class DdeTaskHelper(DfptTaskHelper):
    pass


class PhononTaskHelper(DfptTaskHelper):
    pass


class BecTaskHelper(DfptTaskHelper):
    pass


class StrainPertTaskHelper(DfptTaskHelper):
    task_type = "strain-pert"


##############################
# Post-Processing tasks
##############################


@explicit_serialize
class Cut3DAbinitTaskOld(AbinitSRCMixin, FireTaskBase):
    task_type = "cut3d"

    CUT3D_OPTIONS = ['den_to_cube']


    def __init__(self, cut3d_option=None, deps=None, cut3d_input_file='cut3d.in',
                 cut3d_log_file='cut3d.log', cut3d_err_file='cut3d.err', task_type=None):
        """
        General constructor for Cut3D task.
        """
        self.deps = deps
        if cut3d_option not in self.CUT3D_OPTIONS:
            raise ValueError('Option "{}" for cut3d is not allowed'.format(cut3d_option))
        self.cut3d_option = cut3d_option
        self.cut3d_input_file = cut3d_input_file
        self.cut3d_log_file = cut3d_log_file
        self.cut3d_err_file = cut3d_err_file
        self.files = {}
        if task_type is not None:
            self.task_type = task_type

    def run_cut3d(self):
        """
        executes cut3d and waits for the end of the process.
        """

        def cut3d_process():
            command = []
            #consider the case of serial execution
            command.append(self.ftm.fw_policy.cut3d_cmd)
            with open(self.cut3d_input_file, 'r') as stdin, open(self.cut3d_log_file, 'w') as stdout, \
                    open(self.cut3d_err_file, 'w') as stderr:
                self.process = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=stderr)

            (stdoutdata, stderrdata) = self.process.communicate()
            self.returncode = self.process.returncode

        # initialize returncode to avoid missing references in case of exception in the other thread
        self.returncode = None

        thread = threading.Thread(target=cut3d_process)
        # the amount of time left plus a buffer of 2 minutes
        timeout = (self.walltime - (time.time() - self.start_time) - 120) if self.walltime else None
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            self.process.terminate()
            thread.join()
            raise WalltimeError("The cut3d task couldn't be terminated within the time limit. Killed.")

    def setup_task(self, fw_spec):
        self.start_time = time.time()

        # self.set_logger()

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

    def resolve_deps(self, fw_spec):
        if not self.deps:
            return
        previous_fws = fw_spec.get('previous_fws', None)
        if previous_fws is None:
            msg = "No previous_fws data. Needed for dependecies {}.".format(str(self.deps))
            logger.error(msg)
            raise InitializationError(msg)
        #TODO: right now, only one file is allowed, in some uses of cut3d, this might not be enough
        if len(self.deps) != 1:
            raise NotImplementedError('Only one dependency for cut3d is allowed right now')
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

    def resolve_deps_per_task_type(self, previous_tasks, deps_list):
        for previous_task in previous_tasks:
            for d in deps_list:
                source_dir = previous_task['dir']
                self.link_ext(d, source_dir)

    def link_ext(self, ext, source_dir):
        source = os.path.join(source_dir, self.prefix.odata + "_" + ext)
        logger.info("Need path {} with ext {}".format(source, ext))
        dest = os.path.join(self.workdir, self.prefix.idata + "_" + ext)
        if not os.path.exists(source):
            # Try netcdf file. TODO: this case should be treated in a cleaner way.
            source += "-etsf.nc"
            if os.path.exists(source): dest += "-etsf.nc"
        if not os.path.exists(source):
            msg = "{} is needed by this task but it does not exist".format(source)
            logger.error(msg)
            raise InitializationError(msg)
        # Link path to dest if dest link does not exist.
        # else check that it points to the expected file.
        logger.info("Linking path {} --> {}".format(source, dest))
        if ext == 'DEN':
            self.files['density'] = dest
        else:
            raise InitializationError('Only density files are allowed right now')
        if not os.path.exists(dest):
            if self.ftm.fw_policy.copy_deps:
                shutil.copyfile(source, dest)
            else:
                os.symlink(source, dest)
            return dest

    def run_task(self, fw_spec):
        self.setup_task(fw_spec=fw_spec)
        self.setup_rundir(rundir=os.getcwd(), create_dirs=True, directories_only=True)
        self.workdir = os.getcwd()

        self.resolve_deps(fw_spec=fw_spec)

        cube_filename = 'density.cube'
        # TODO: make this more general ?
        from abiflows.fireworks.tasks.abinit_common import Cut3DInput
        cut3d_input = Cut3DInput.den_to_cube(self.files['density'], cube_filename=cube_filename)
        cut3d_input.write_input(self.cut3d_input_file)
        self.run_cut3d()
        return FWAction(update_spec={'cut3d_directory': self.workdir, 'cube_filename': cube_filename})

    @classmethod
    def den_to_cube(cls, deps, task_type=None):
        if task_type is None:
            task_type = 'cut3d-den-to-cube'
        return cls(cut3d_option='den_to_cube', deps=deps, task_type=task_type)

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

    # Prefixes for Abinit (input, output, temporary) files.
    Prefix = collections.namedtuple("Prefix", "idata odata tdata")
    pj = os.path.join

    prefix = Prefix(pj("indata", "in"), pj("outdata", "out"), pj("tmpdata", "tmp"))
    del Prefix, pj


@explicit_serialize
class Cut3DAbinitTask(AbinitSRCMixin, FireTaskBase):
    task_type = "cut3d"

    def __init__(self, cut3d_input, structure=None, deps=None, task_type=None):
        """
        General constructor for Cut3D task.
        Structure needed for Hirshfeld
        """
        self.deps = deps
        self.cut3d_input = cut3d_input
        self.structure = structure
        if task_type is not None:
            self.task_type = task_type
        self.files = {}

    def run_cut3d(self):
        """
        executes cut3d and waits for the end of the process.
        """

        def cut3d_process():
            command = []
            #consider the case of serial execution
            command.append(self.ftm.fw_policy.cut3d_cmd)
            with open(self.input_file.path, 'r') as stdin, open(self.log_file.path, 'w') as stdout, \
                    open(self.stderr_file.path, 'w') as stderr:
                self.process = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=stderr)

            (stdoutdata, stderrdata) = self.process.communicate()
            self.returncode = self.process.returncode

        # initialize returncode to avoid missing references in case of exception in the other thread
        self.returncode = None

        thread = threading.Thread(target=cut3d_process)
        # the amount of time left plus a buffer of 2 minutes
        timeout = (self.walltime - (time.time() - self.start_time) - 120) if self.walltime else None
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            self.process.terminate()
            thread.join()
            raise WalltimeError("The cut3d task couldn't be terminated within the time limit. Killed.")

    def set_workdir(self, workdir):
        """Set the working directory."""

        self.workdir = os.path.abspath(workdir)

        # Files required for the execution.
        self.input_file = File(os.path.join(self.workdir, 'cut3d.in'))
        self.log_file = File(os.path.join(self.workdir, 'cut3d.log'))
        self.stderr_file = File(os.path.join(self.workdir, 'cut3d.err'))

    def setup_task(self, fw_spec):
        self.start_time = time.time()

        # self.set_logger()

        # load the FWTaskManager to get configuration parameters
        self.ftm = self.get_fw_task_manager(fw_spec)

        self.set_workdir(workdir=os.getcwd())

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

    def resolve_deps(self, fw_spec):
        if not self.deps:
            return
        previous_fws = fw_spec.get('previous_fws', None)
        if previous_fws is None:
            msg = "No previous_fws data. Needed for dependecies {}.".format(str(self.deps))
            logger.error(msg)
            raise InitializationError(msg)
        #TODO: right now, only one file is allowed, in some uses of cut3d, this might not be enough
        if len(self.deps) != 1:
            raise NotImplementedError('Only one dependency for cut3d is allowed right now')
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

    def resolve_deps_per_task_type(self, previous_tasks, deps_list):
        for previous_task in previous_tasks:
            for d in deps_list:
                source_dir = previous_task['dir']
                self.link_ext(d, source_dir)

    def link_ext(self, ext, source_dir):
        source = os.path.join(source_dir, self.prefix.odata + "_" + ext)
        logger.info("Need path {} with ext {}".format(source, ext))
        dest = os.path.join(self.workdir, self.prefix.idata + "_" + ext)
        if not os.path.exists(source):
            # Try netcdf file. TODO: this case should be treated in a cleaner way.
            source += "-etsf.nc"
            if os.path.exists(source): dest += "-etsf.nc"
        if not os.path.exists(source):
            msg = "{} is needed by this task but it does not exist".format(source)
            logger.error(msg)
            raise InitializationError(msg)
        # Link path to dest if dest link does not exist.
        # else check that it points to the expected file.
        logger.info("Linking path {} --> {}".format(source, dest))
        if ext == 'DEN':
            self.files['density'] = dest
        else:
            raise InitializationError('Only density files are allowed right now')
        if not os.path.exists(dest):
            if self.ftm.fw_policy.copy_deps:
                shutil.copyfile(source, dest)
            else:
                os.symlink(source, dest)
            return dest

    def run_task(self, fw_spec):
        self.setup_task(fw_spec=fw_spec)
        self.setup_rundir(rundir=os.getcwd(), create_dirs=True, directories_only=True)
        self.workdir = os.getcwd()

        self.resolve_deps(fw_spec=fw_spec)

        #TODO now it's just the filename as there are problems with full paths in cut3d, it should became the abspath
        # when it's fixed in abinit
        converted_filename = self.cut3d_input.output_filepath
        self.cut3d_input.infile_path = self.files['density']
        self.input_file.write(str(self.cut3d_input))
        self.run_cut3d()
        return FWAction(update_spec={'cut3d_directory': self.workdir, 'converted_filename': converted_filename})

    @classmethod
    def den_to_cube(cls, deps, task_type=None):
        if task_type is None:
            task_type = 'cut3d-den-to-cube'

        cut3d_input = Cut3DInput.den_to_cube(density_filepath=None, output_filepath='density.cube')

        return cls(cut3d_input=cut3d_input, deps=deps, task_type=task_type)

    @classmethod
    def hirshfeld(cls, deps, structure, task_type=None, all_el_dens_paths = None, fhi_all_el_path = None):
        if task_type is None:
            task_type = 'cut3d-hirshfeld'

        if all_el_dens_paths is None and fhi_all_el_path is None:
            raise ValueError("At least one source of all electron densities should be provided")

        if all_el_dens_paths is not None:
            cut3d_input = Cut3DInput.hirshfeld(None, all_el_dens_paths)
        else:
            cut3d_input = Cut3DInput.hirshfeld_from_fhi_path(None, structure, fhi_all_el_path)

        return cls(cut3d_input=cut3d_input, structure=structure, deps=deps, task_type=task_type)

    def get_hirshfeld_charges(self):
        hc = HirshfeldCharges.from_cut3d_outfile(self.log_file.path, self.structure)
        return hc

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

    # Prefixes for Abinit (input, output, temporary) files.
    Prefix = collections.namedtuple("Prefix", "idata odata tdata")
    pj = os.path.join

    prefix = Prefix(pj("indata", "in"), pj("outdata", "out"), pj("tmpdata", "tmp"))
    del Prefix, pj


@explicit_serialize
class BaderTask(AbinitSRCMixin, FireTaskBase):
    task_type = "bader"


    def __init__(self, bader_log_file='bader.log', bader_err_file='bader.err', electrons=None):
        """
        General constructor for Cut3D task.
        """

        self.bader_log_file = bader_log_file
        self.bader_err_file = bader_err_file
        if electrons is not None:
            if electrons not in ['valence', 'all-electron']:
                raise ValueError('Argument "electrons" should be "valence" or "all-electron"')
        self.electrons = electrons

    def set_workdir(self, workdir):
        self.workdir = workdir

    def setup_task(self, fw_spec):
        self.start_time = time.time()

        # self.set_logger()

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

    def run_bader(self):
        """
        executes bader and waits for the end of the process.
        """

        bader_exe = which('bader')
        if bader_exe is None:
            raise ValueError('The bader executable is not in the PATH')
        def bader_process():
            command = []
            #consider the case of serial execution
            command.append(bader_exe)
            (cube_dirpath, cube_filename) = os.path.split(self.cube_filepath)
            os.symlink(self.cube_filepath, cube_filename)
            command.append(cube_filename)
            with open(self.bader_log_file, 'w') as stdout, open(self.bader_err_file, 'w') as stderr:
                self.process = subprocess.Popen(command, stdout=stdout, stderr=stderr)

            (stdoutdata, stderrdata) = self.process.communicate()
            self.returncode = self.process.returncode

        # initialize returncode to avoid missing references in case of exception in the other thread
        self.returncode = None

        thread = threading.Thread(target=bader_process)
        # the amount of time left plus a buffer of 2 minutes
        timeout = (self.walltime - (time.time() - self.start_time) - 120) if self.walltime else None
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            self.process.terminate()
            thread.join()
            raise WalltimeError("The cut3d task couldn't be terminated within the time limit. Killed.")

    def setup_rundir(self, rundir, create_dirs=True, directories_only=False):
        # Directories with input|output|temporary data.
        if self.electrons is None:
            self.bader_dir = Directory(os.path.join(rundir, 'bader'))
        else:
            self.bader_dir = Directory(os.path.join(rundir, 'bader', self.electrons))
        self.rundir = self.bader_dir.path

        # Create dir for bader
        if create_dirs:
            self.bader_dir.makedirs()

    def run_task(self, fw_spec):
        self.setup_task(fw_spec=fw_spec)
        self.cube_filepath = os.path.join(fw_spec['cut3d_directory'], fw_spec['converted_filename'])
        self.setup_rundir(os.getcwd())
        os.chdir(self.rundir)
        self.run_bader()
        return FWAction(update_spec={'bader_directory': self.rundir})

    def get_bader_data(self):
        os.chdir(self.rundir)
        data = []
        with open("ACF.dat") as f:
            raw = f.readlines()
            headers = [s.lower() for s in raw.pop(0).split()]
            raw.pop(0)
            while True:
                l = raw.pop(0).strip()
                if l.startswith("-"):
                    break
                vals = map(float, l.split()[1:])
                data.append(dict(zip(headers[1:], vals)))
            for l in raw:
                toks = l.strip().split(":")
                if toks[0] == "VACUUM CHARGE":
                    self.vacuum_charge = float(toks[1])
                elif toks[0] == "VACUUM VOLUME":
                    self.vacuum_volume = float(toks[1])
                elif toks[0] == "NUMBER OF ELECTRONS":
                    self.nelectrons = float(toks[1])
        return data

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



##############################
# Generation tasks
##############################


@explicit_serialize
class GeneratePiezoElasticFlowFWSRCAbinitTask(FireTaskBase):
    def __init__(self, piezo_elastic_factory=None, helper=None, previous_scf_task_type=ScfTaskHelper.task_type,
                 previous_ddk_task_type=DdkTaskHelper.task_type, control_procedure=None,
                 additional_controllers=None,
                 mrgddb_task_type='mrgddb-strains',
                 rf_tol=None, additional_input_vars=None, rf_deps=None,
                 allow_parallel_perturbations=True, do_phonons=True):
        if piezo_elastic_factory is None:
            self.piezo_elastic_factory = PiezoElasticFromGsFactory(rf_tol=rf_tol, rf_split=True)
        else:
            self.piezo_elastic_factory = piezo_elastic_factory
        if helper is None:
            self.helper = StrainPertTaskHelper()
        else:
            self.helper = helper
        self.previous_scf_task_type = previous_scf_task_type
        self.previous_ddk_task_type = previous_ddk_task_type
        if control_procedure is None:
            if additional_controllers is None:
                controllers = [AbinitController.from_helper(self.helper),
                               WalltimeController(), MemoryController()]
            else:
                controllers = [AbinitController.from_helper(self.helper)]
                controllers.extend(additional_controllers)
            self.control_procedure = ControlProcedure(controllers=controllers)
        else:
            self.control_procedure = control_procedure
        self.additional_controllers = additional_controllers
        self.mrgddb_task_type = mrgddb_task_type
        self.rf_tol = rf_tol
        self.additional_input_vars = additional_input_vars
        self.rf_deps = rf_deps
        self.allow_parallel_perturbations = allow_parallel_perturbations
        self.do_phonons = do_phonons

    def run_task(self, fw_spec):

        # Get the previous SCF input
        previous_scf_input = fw_spec.get('previous_fws', {}).get(self.previous_scf_task_type,
                                                                 [{}])[0].get('input', None)
        if not previous_scf_input:
            raise InitializationError('No input file available '
                                      'from task of type {}'.format(self.previous_scf_task_type))
        from pymatgen.io.abinit import tasks
        ftm = self.get_fw_task_manager(fw_spec)
        tasks._USER_CONFIG_TASKMANAGER = ftm.task_manager

        # Get the strain RF inputs
        piezo_elastic_inputs = self.piezo_elastic_factory.build_input(previous_scf_input)
        if self.do_phonons:
            rf_strain_inputs = piezo_elastic_inputs.filter_by_tags(STRAIN)
        else:
            rf_strain_inputs = piezo_elastic_inputs.filter_by_tags(STRAIN, exclude_tags=PHONON)

        initialization_info = fw_spec.get('initialization_info', {})
        initialization_info['input_factory'] = self.piezo_elastic_factory.as_dict()
        new_spec = dict(initialization_info=initialization_info, previous_fws=fw_spec.get('previous_fws', {}))
        initial_parameters = fw_spec.get('initial_parameters', None)
        if initial_parameters:
            new_spec['initial_parameters'] = initial_parameters

        if '_preserve_fworker' in fw_spec:
            new_spec['_preserve_fworker']=True
        if '_fworker' in fw_spec:
            new_spec['_fworker'] = fw_spec['_fworker']

        # Create the SRC fireworks for each perturbation
        all_SRC_rf_fws = []
        total_list_fws = []
        strain_task_types = []
        fws_deps = {}

        if self.rf_deps is not None:
            rf_deps = self.rf_deps
        else:
            rf_deps = {self.previous_scf_task_type: 'WFK'}
            if self.previous_ddk_task_type is not None:
                rf_deps[self.previous_ddk_task_type] = 'DDK'

        prev_src_pert = None

        for istrain_pert, rf_strain_input in enumerate(rf_strain_inputs):
            strain_task_type = 'strain-pert-{:d}'.format(istrain_pert+1)
            if self.additional_input_vars is not None:
                rf_strain_input.set_vars(self.additional_input_vars)
            rf_strain_input.set_vars(mem_test=0)
            setup_rf_task = AbinitSetupTask(abiinput=rf_strain_input, task_helper=self.helper,
                                            deps=rf_deps)
            run_rf_task = AbinitRunTask(control_procedure=self.control_procedure, task_helper=self.helper,
                                        task_type=strain_task_type)
            control_rf_task = AbinitControlTask(control_procedure=self.control_procedure, task_helper=self.helper)

            rf_fws = createSRCFireworks(setup_task=setup_rf_task, run_task=run_rf_task,
                                        control_task=control_rf_task,
                                        spec=new_spec, initialization_info=initialization_info)
            all_SRC_rf_fws.append(rf_fws)
            total_list_fws.extend(rf_fws['fws'])
            strain_task_types.append(strain_task_type)
            links_dict_update(links_dict=fws_deps, links_update=rf_fws['links_dict'])
            # Additional links if we want to avoid multiple perturbations to be run at the same time (e.g. to avoid
            # I/O bottlenecks because of reading the same file
            if not self.allow_parallel_perturbations:
                if prev_src_pert is not None:
                    link_dict_update = {prev_src_pert['control_fw'].fw_id: [rf_fws['setup_fw'].fw_id]}
                    links_dict_update(links_dict=fws_deps, links_update=link_dict_update)
                prev_src_pert = rf_fws



        # Adding the MrgDdb Firework
        mrgddb_spec = dict(new_spec)
        mrgddb_spec = set_short_single_core_to_spec(mrgddb_spec)
        mrgddb_spec['_priority'] = 10
        num_ddbs_to_be_merged = len(all_SRC_rf_fws)
        mrgddb_fw = Firework(MergeDdbAbinitTask(ddb_source_task_types=strain_task_types,
                                                num_ddbs=num_ddbs_to_be_merged,
                                                delete_source_ddbs=True,
                                                task_type= self.mrgddb_task_type),
                             spec=mrgddb_spec, name='mrgddb-strains')
        total_list_fws.append(mrgddb_fw)
        #Adding the dependencies
        for src_fws in all_SRC_rf_fws:
            links_dict_update(links_dict=fws_deps, links_update={src_fws['control_fw'].fw_id: mrgddb_fw.fw_id})

        rf_strains_wf = Workflow(total_list_fws, fws_deps)

        return FWAction(detours=rf_strains_wf)

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




####################
# Exceptions
####################

class HelperError(Exception):
    pass


class InitializationError(Exception):
    pass


class RestartError(Exception):
    pass


class WalltimeError(Exception):
    pass


class PostProcessError(Exception):
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



#@deprecated(message="AbinitOutNcFile is deprecated, use abipy.abio.outputs.OutNcFile")
class _AbinitOutNcFile(NetcdfReader):
    """
    Class representing the _OUT.nc file.
    """

    def get_vars(self, vars, strict=False):
        # TODO: add a check on the variable names ?
        default = NO_DEFAULT if strict else None
        var_values = {}
        for var in vars:
            var_values[var] = self.read_value(varname=var, default=default)
        return var_values