# coding: utf-8
"""
Utilities for fireworks
"""

from __future__ import print_function, division, unicode_literals
from collections import namedtuple
import copy
from monty.serialization import loadfn
import os
from pymatgen.io.abinit import TaskManager

from pymatgen.io.abinit.tasks import ParalHints
from fireworks import Workflow
import traceback
import logging
from abiflows.fireworks.utils.time_utils import TimeReport

logger = logging.getLogger(__name__)

SHORT_SINGLE_CORE_SPEC = {'_queueadapter': {'ntasks': 1, 'time': '00:10:00'}, 'mpi_ncpus': 1}


def parse_workflow(fws, links_dict):
    new_list = []
    for fw in fws:
        if isinstance(fw, Workflow):
            new_list.extend(fw.fws)
        else:
            new_list.append(fw)

    new_links_dict = {}
    for parent, children in links_dict.items():
        if isinstance(parent, Workflow):
            new_links_dict.update(parent.links)
            for leaf_fw_id in parent.leaf_fw_ids:
                new_links_dict[leaf_fw_id] = children
        else:
            new_links_dict[parent] = children

    # dict since the collection will be updated
    for parent, children in dict(new_links_dict).items():
        final_childrens = []
        for child in children:
            if isinstance(child, Workflow):
                new_links_dict.update(child.links)
                final_childrens.extend(child.root_fw_ids)
            else:
                final_childrens.append(child)
        new_links_dict[parent] = final_childrens

    return new_list, new_links_dict


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


def get_short_single_core_spec(fw_manager=None, master_mem_overhead=0):
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
            tm.qadapter.set_timelimit(600)
            tm.qadapter.set_master_mem_overhead(master_mem_overhead)
            qadapter_spec = tm.qadapter.get_subs_dict()
            return qadapter_spec
        except RuntimeError as e:
            traceback.print_exc()

    # No taskmanger or no queue available
    #FIXME return something else? exception?
    return {}


def set_short_single_core_to_spec(spec={}, master_mem_overhead=0):
        spec = dict(spec)

        qadapter_spec = get_short_single_core_spec(master_mem_overhead=master_mem_overhead)
        spec['mpi_ncpus'] = 1
        spec['_queueadapter'] = qadapter_spec
        return spec


class FWTaskManager(object):
    """
    Object containing the configuration parameters and policies to run abipy.
    The policies needed for the abinit FW will always be available through default values. These can be overridden
    also setting the parameters in the spec.
    The standard abipy task manager is contained as an object on its own that can be used to run the autoparal.
    The rationale behind this choice, instead of subclassing, is to not force the user to fill the qadapter part
    of the task manager, which is needed only for the autoparal, but is required in the TaskManager initialization.
    Wherever the TaskManager is needed just pass the ftm.task_manager
    """

    YAML_FILE = "fw_manager.yaml"
    USER_CONFIG_DIR = TaskManager.USER_CONFIG_DIR # keep the same as the standard TaskManager

    fw_policy_defaults = dict(rerun_same_dir=False,
                              max_restarts=10,
                              autoparal=False,
                              abinit_cmd='abinit',
                              mrgddb_cmd='mrgddb',
                              anaddb_cmd='anaddb',
                              mpirun_cmd='mpirun',
                              copy_deps=False,
                              walltime_command=None,
                              continue_unconverged_on_rerun=True,
                              allow_local_restart=False,
                              timelimit_buffer=120)
    FWPolicy = namedtuple("FWPolicy", fw_policy_defaults.keys())

    def __init__(self, **kwargs):
        self._kwargs = copy.deepcopy(kwargs)

        fw_policy = kwargs.pop('fw_policy', {})
        unknown_keys = set(fw_policy.keys()) - set(self.fw_policy_defaults.keys())
        if unknown_keys:
            msg = "Unknown key(s) present in fw_policy: ".format(", ".join(unknown_keys))
            logger.error(msg)
            raise RuntimeError(msg)
        fw_policy = dict(self.fw_policy_defaults, **fw_policy)

        # make a namedtuple for easier access to the attributes
        self.fw_policy = self.FWPolicy(**fw_policy)

        #TODO consider if raising an exception if it's requested when not available
        # create the task manager only if possibile
        if 'qadapters' in kwargs:
            self.task_manager = TaskManager.from_dict(kwargs)
        else:
            self.task_manager = None

    @classmethod
    def from_user_config(cls, fw_policy={}):
        """
        Initialize the manager using the dict in the following order of preference:
        - the "fw_manager.yaml" file in the folder where the command is executed
        - a yaml file pointed by the "FW_TASK_MANAGER"
        - the "fw_manager.yaml" in the ~/.abinit/abipy folder
        - if no file available, fall back to default values
        """

        # Try in the current directory then in user configuration directory.
        paths = [os.path.join(os.getcwd(), cls.YAML_FILE), os.getenv("FW_TASK_MANAGER"),
                 os.path.join(cls.USER_CONFIG_DIR, cls.YAML_FILE)]

        config = {}
        for path in paths:
            if path and os.path.exists(path):
                config = loadfn(path)
                logger.info("Reading manager from {}.".format(path))
                break

        return cls(**config)

    @classmethod
    def from_file(cls, path):
        """Read the configuration parameters from the Yaml file filename."""
        return cls(**(loadfn(path)))

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

            if 'mpi_ncpus' in fw.spec:
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
        if isinstance(child_ids, int):
                child_ids = [child_ids]
        if parent_id in links_dict:
            for child_id in child_ids:
                if child_id in links_dict[parent_id]:
                    raise ValueError('Child fireworks already defined for parent ...')
                links_dict[parent_id].append(child_id)
        else:
            links_dict[parent_id] = child_ids