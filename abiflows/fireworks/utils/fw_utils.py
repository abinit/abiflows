# coding: utf-8
"""
Utilities for fireworks
"""
from __future__ import print_function, division, unicode_literals, absolute_import

import copy
import os
import traceback
import logging
import warnings
import json
import yaml
import io

from collections import namedtuple
from monty.json import MontyDecoder
from abipy.flowtk.tasks import TaskManager, ParalHints
from fireworks.core.firework import Firework, Workflow
from fireworks.core.launchpad import LaunchPad
from abiflows.fireworks.utils.time_utils import TimeReport

logger = logging.getLogger(__name__)

SHORT_SINGLE_CORE_SPEC = {'_queueadapter': {'ntasks': 1, 'time': '00:10:00'}, 'mpi_ncpus': 1}


def append_fw_to_wf(new_fw, wf):

    if new_fw.fw_id in wf.id_fw:
        raise ValueError('FW ids must be unique!')

    for leaf_id in wf.leaf_fw_ids:
        wf.links[leaf_id].append(new_fw.fw_id)

    wf.links[new_fw.fw_id] = []
    wf.id_fw[new_fw.fw_id] = new_fw

    # add depends on
    for pfw in new_fw.parents:
        if pfw.fw_id not in wf.links:
            raise ValueError(
                "FW_id: {} defines a dependent link to FW_id: {}, but the latter was not added to the workflow!".format(
                    new_fw.fw_id, pfw.fw_id))
        wf.links[pfw.fw_id].append(new_fw.fw_id)

    # sanity: make sure the set of nodes from the links_dict is equal to
    # the set of nodes from id_fw
    if set(wf.links.nodes) != set(map(int, wf.id_fw.keys())):
        raise ValueError("Specified links don't match given FW")

    wf.fw_states[new_fw.fw_id] = new_fw.state


def get_short_single_core_spec(fw_manager=None, master_mem_overhead=0, return_qtk=False, timelimit=None):
    if isinstance(fw_manager, FWTaskManager):
        ftm = fw_manager
    elif fw_manager:
        ftm = FWTaskManager.from_file(fw_manager)
    else:
        ftm = FWTaskManager.from_user_config()

    if ftm.has_task_manager():
        #TODO add mem_per_cpu?
        pconf = ParalHints({}, [{'tot_ncpus': 1, 'mpi_ncpus': 1, 'efficiency': 1}])
        try:
            tm = ftm.task_manager
            tm.select_qadapter(pconf)
            #TODO make a FW_task_manager parameter
            if timelimit is None:
                tm.qadapter.set_timelimit(timelimit=ftm.fw_policy.short_job_timelimit)
            else:
                tm.qadapter.set_timelimit(timelimit=timelimit)
            tm.qadapter.set_master_mem_overhead(master_mem_overhead)
            qadapter_spec = tm.qadapter.get_subs_dict()
            if return_qtk:
                return qadapter_spec, tm.qadapter
            else:
                return qadapter_spec
        except RuntimeError as e:
            traceback.print_exc()

    # No taskmanger or no queue available
    #FIXME return something else? exception?
    return {}


def set_short_single_core_to_spec(spec={}, master_mem_overhead=0, fw_manager=None):
        if spec is None:
            spec = {}
        else:
            spec = dict(spec)

        qadapter_spec = get_short_single_core_spec(master_mem_overhead=master_mem_overhead, fw_manager=fw_manager)
        spec['mpi_ncpus'] = 1
        spec['_queueadapter'] = qadapter_spec
        return spec


class FWTaskManager(object):
    """
    Object containing the configuration parameters and policies to run abipy.
    The policies needed for the abinit FW will always be available through default values. These can be overridden
    also setting the parameters in the spec.
    The standard abipy task manager is contained as an object on its own that can be used to run the autoparal or
    factories if needed.
    The rationale behind this choice, instead of subclassing, is to not force the user to fill the qadapter part
    of the task manager, which is needed only for the autoparal, but is required in the TaskManager initialization.
    Wherever the TaskManager is needed just pass the ftm.task_manager.
    The TaskManager part can be loaded from an external manager.yml file using the "abipy_manager" key in fw_policy.
    This is now the preferred choice. If this value is not defined, it will be loaded with TaskManager.from_user_config
    """

    YAML_FILE = "fw_manager.yaml"
    USER_CONFIG_DIR = TaskManager.USER_CONFIG_DIR # keep the same as the standard TaskManager

    fw_policy_defaults = dict(rerun_same_dir=False,
                              max_restarts=10,
                              autoparal=False,
                              abinit_cmd='abinit',
                              mrgddb_cmd='mrgddb',
                              anaddb_cmd='anaddb',
                              cut3d_cmd='cut3d',
                              mpirun_cmd='mpirun',
                              copy_deps=False,
                              walltime_command=None,
                              continue_unconverged_on_rerun=True,
                              allow_local_restart=False,
                              timelimit_buffer=120,
                              short_job_timelimit=600,
                              recover_previous_job=True,
                              abipy_manager=None)
    FWPolicy = namedtuple("FWPolicy", fw_policy_defaults.keys())

    def __init__(self, **kwargs):
        self._kwargs = copy.deepcopy(kwargs)

        fw_policy = kwargs.pop('fw_policy', {})
        unknown_keys = set(fw_policy.keys()) - set(self.fw_policy_defaults.keys())
        if unknown_keys:
            msg = "Unknown key(s) present in fw_policy: {}".format(", ".join(unknown_keys))
            logger.error(msg)
            raise RuntimeError(msg)
        fw_policy = dict(self.fw_policy_defaults, **fw_policy)

        # make a namedtuple for easier access to the attributes
        self.fw_policy = self.FWPolicy(**fw_policy)

        #TODO consider if raising an exception if it's requested when not available
        # create the task manager only if possibile
        if 'qadapters' in kwargs:
            self.task_manager = TaskManager.from_dict(kwargs)
            msg = "Loading the abipy TaskManager from inside the fw_manager.yaml file is deprecated. " \
                  "Use a separate file"
            logger.warning(msg)
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
        else:
            if self.fw_policy.abipy_manager:
                self.task_manager = TaskManager.from_file(self.fw_policy.abipy_manager)
            else:
                try:
                    self.task_manager = TaskManager.from_user_config()
                except:
                    logger.warning("Couldn't load the abipy task manager.")
                    self.task_manager = None

    @classmethod
    def from_user_config(cls, fw_policy=None):
        """
        Initialize the manager using the dict in the following order of preference:
        - the "fw_manager.yaml" file in the folder where the command is executed
        - a yaml file pointed by the "FW_TASK_MANAGER"
        - the "fw_manager.yaml" in the ~/.abinit/abipy folder
        - if no file available, fall back to default values
        """

        if fw_policy is None:
            fw_policy = {}

        # Try in the current directory then in user configuration directory.
        paths = [os.path.join(os.getcwd(), cls.YAML_FILE), os.getenv("FW_TASK_MANAGER"),
                 os.path.join(cls.USER_CONFIG_DIR, cls.YAML_FILE)]

        config = {}
        for path in paths:
            if path and os.path.exists(path):
                with io.open(path, "rt", encoding="utf-8") as fh:
                    config = yaml.load(fh)
                logger.info("Reading manager from {}.".format(path))
                break

        return cls(**config)

    @classmethod
    def from_file(cls, path):
        """Read the configuration parameters from the Yaml file filename."""
        with open(path, "rt") as fh:
            d = yaml.load(fh)
        return cls(**d)

    def has_task_manager(self):
        return self.task_manager is not None

    def update_fw_policy(self, d):
        self.fw_policy = self.fw_policy._replace(**d)


def get_time_report_for_wf(wf):

        total_run_time = 0
        total_cpu_time = 0
        contributed_cpu_time = 0
        total_run_time_per_tag = {}
        total_cpu_time_per_tag = {}
        contributed_cpu_time_per_tag = {}
        worker = None

        for fw in wf.fws:
            # skip not completed fws
            if fw.state != 'COMPLETED':
                continue

            launches = fw.archived_launches + fw.launches
            completed_launch = None

            #take the last completed launch
            for l in launches[-1::-1]:
                if l.state == 'COMPLETED':
                    completed_launch = l
                    break

            # assume all fw have runned on the same worker
            if worker is None:
                worker = completed_launch.fworker.name

            run_time = completed_launch.runtime_secs
            total_run_time += run_time

            if 'qtk_queueadapter' in fw.spec:
                ncpus = fw.spec['qtk_queueadapter'].num_cores
            elif 'mpi_ncpus' in fw.spec:
                ncpus = fw.spec['mpi_ncpus']
            elif 'ntasks' in fw.spec:
                ncpus = fw.spec['ntasks']
            else:
                ncpus = None

            if ncpus:
                cpu_time = run_time*ncpus
                total_cpu_time += cpu_time
                contributed_cpu_time += 1
            else:
                cpu_time = None

            wf_task_index = fw.spec.get('wf_task_index', 'Unclassified')

            task_tag = wf_task_index.rsplit('_', 1)[0]
            total_run_time_per_tag[task_tag] = total_run_time_per_tag.get(task_tag, 0) + run_time
            if cpu_time is not None:
                total_cpu_time_per_tag[task_tag] = total_cpu_time_per_tag.get(task_tag, 0) + cpu_time
                contributed_cpu_time_per_tag[task_tag] = contributed_cpu_time_per_tag.get(task_tag, 0) + 1

        tr = TimeReport(total_run_time, len(wf.fws), total_cpu_time=total_cpu_time, contributed_cpu_time=contributed_cpu_time,
                       total_run_time_per_tag=total_run_time_per_tag, total_cpu_time_per_tag=total_cpu_time_per_tag,
                       contributed_cpu_time_per_tag=contributed_cpu_time_per_tag, worker=worker)

        return tr


def links_dict_update(links_dict, links_update):
    for parent_id, child_ids in links_update.items():
        if isinstance(parent_id, Firework):
            parent_id = parent_id.fw_id
        if isinstance(child_ids, int):
            child_ids = [child_ids]
        elif isinstance(child_ids, Firework):
            child_ids = [child_ids.fw_id]
        if parent_id in links_dict:
            for child_id in child_ids:
                if child_id in links_dict[parent_id]:
                    raise ValueError('Child fireworks already defined for parent ...')
                links_dict[parent_id].append(child_id)
        else:
            links_dict[parent_id] = child_ids


def get_last_completed_launch(fw):
    """
    Given a Firework object returns the last completed launch
    """
    return next((l for l in reversed(fw.archived_launches + fw.launches) if
                 l.state == 'COMPLETED'), None)


def load_abitask(fw):
    """
    Given a Firework object returns the abinit related task contained. Sets the list of directories set from the
    last completed launch. If no abinit related firetasks are found or the task has no completed launch returns None.
    """

    from abiflows.fireworks.tasks.abinit_tasks import AbiFireTask, AnaDdbAbinitTask, MergeDdbAbinitTask

    for t in fw.tasks:
        if isinstance(t, (AbiFireTask, AnaDdbAbinitTask, MergeDdbAbinitTask)):
            launch = get_last_completed_launch(fw)
            if launch:
                t.set_workdir(workdir=launch.launch_dir)
                return t

    return None

def get_fw_by_task_index(wf, task_tag, index=1):
    """
    Given a workflow object (with connection to the db) returns the wf corresponding to the task_type.

    Args:
        wf: a fireworks Workflow object.
        task_tag: the task tag associated with the task as defined in abinit_workflows. Should not include the index.
        index: the numerical or text index of the task. If negative the the last fw corresponding to task_tag will
            be selected. If None, no index will be considered and the first match will be returned.

    Returns:
        a fireworks Firework object. None if no match is found.
    """

    task_index = None
    if index is not None and index >=0:
        task_index = "{}_{}".format(task_tag, index)

    selected_fw = None
    max_ind = -1
    for fw in wf.fws:
        fw_task_index = fw.spec.get('wf_task_index', '')
        if task_index:
            if fw_task_index == task_index:
                return fw
        else:
            if task_tag in fw_task_index:
                if index is None:
                    return fw
                # the last part of the task_index can be text (i.e. "autoparal") so the conversion to int may fail
                # if no other indices has been found select that one
                try:
                    fw_ind = int(fw_task_index.split('_')[-1])
                    if  fw_ind > max_ind:
                        selected_fw = fw
                        max_ind = fw_ind
                except:
                    if selected_fw is None:
                        selected_fw = fw

    return selected_fw


def get_lp_and_fw_id_from_task(task, fw_spec):
    """
    Given an instance of a running task and its spec, tries to load the LaunchPad and the current fw_id.
    It will first check for "_add_launchpad_and_fw_id", then try to load from FW.json/FW.yaml file.

    Should be used inside tasks that require to access to the LaunchPad and to the whole workflow.
    Args:
        task: An instance of a running task
        fw_spec: The spec of the task

    Returns:
        an instance of LaunchPah and the fw_id of the current task
    """
    if '_add_launchpad_and_fw_id' in fw_spec:
        lp = task.launchpad
        fw_id = task.fw_id

        # lp may be None in offline mode
        if lp is None:
            raise RuntimeError("The LaunchPad in spec is None.")
    else:
        try:
            with open('FW.json', "rt") as fh:
                fw_dict = json.load(fh, cls=MontyDecoder)
        except IOError:
            try:
                with open('FW.yaml', "rt") as fh:
                    fw_dict = yaml.load(fh)
            except IOError:
                raise RuntimeError("Launchpad/fw_id not present in spec and No FW.json nor FW.yaml file present: "
                                   "impossible to determine fw_id")

        logger.warning("LaunchPad not available from spec. Generated with auto_load.")
        lp = LaunchPad.auto_load()
        fw_id = fw_dict['fw_id']

        # since it is not given that the LaunchPad is the correct one, try to verify if the workflow
        # and the fw_id are being accessed correctly
        try:
            fw = lp.get_fw_by_id(fw_id)
        except ValueError as e:
            traceback.print_exc()
            raise RuntimeError("The firework with id {} is not present in the LaunchPad {}. The LaunchPad is "
                               "probably incorrect.". format(fw_id, lp))

        if fw.state != "RUNNING":
            raise RuntimeError("The firework with id {} from LaunchPad {} is {}. There might be an error in the "
                               "selection of the LaunchPad". format(fw_id, lp, fw.state))

        if len(fw.tasks) != len(fw_dict['spec']['_tasks']):
            raise RuntimeError("The firework with id {} from LaunchPad {} is has different number of tasks "
                               "from the current.".format(fw_id, lp))

        for db_t, dict_t in zip(fw.tasks, fw_dict['spec']['_tasks']):
            if db_t.fw_name != dict_t['_fw_name']:
                raise RuntimeError("The firework with id {} from LaunchPad {} has task that don't  match: "
                                   "{} and {}.".format(fw_id, lp, db_t.fw_name, dict_t['fw_name']))

    return lp, fw_id
