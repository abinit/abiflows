# coding: utf-8
"""
Utility tasks for Fireworks.
"""

from __future__ import print_function, division, unicode_literals

from fireworks.core.firework import Firework, FireTaskBase, FWAction
from fireworks.core.launchpad import LaunchPad
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.utilities.fw_serializers import serialize_fw

import os
import shutil
import logging
import traceback
import importlib
from abiflows.fireworks.tasks.abinit_tasks import INDIR_NAME, OUTDIR_NAME, TMPDIR_NAME, FWTaskManager
from abiflows.fireworks.utils.databases import MongoDatabase
from monty.serialization import loadfn
from monty.json import jsanitize


logger = logging.getLogger(__name__)


@explicit_serialize
class FinalCleanUpTask(FireTaskBase):
    task_type = 'finalclnup'

    def __init__(self, out_exts=["WFK", "1WF"]):
        if isinstance(out_exts, str):
            out_exts = [s.strip() for s in out_exts.split(',')]

        self.out_exts = out_exts

    @serialize_fw
    def to_dict(self):
        return dict(out_exts=self.out_exts)

    @classmethod
    def from_dict(cls, m_dict):
        return cls(out_exts=m_dict['out_exts'])

    @staticmethod
    def delete_files(d, exts=None):
        deleted_files = []
        if os.path.isdir(d):
            for f in os.listdir(d):
                if exts is None or "*" in exts or any(ext in f for ext in exts):
                    fp = os.path.join(d, f)
                    try:
                        if os.path.isfile(fp):
                            os.unlink(fp)
                        elif os.path.isdir(fp):
                            shutil.rmtree(fp)
                        deleted_files.append(fp)
                    except:
                        logger.warning("Couldn't delete {}: {}".format(fp, traceback.format_exc()))

        return deleted_files

    def run_task(self, fw_spec):
        # the FW.json/yaml file is mandatory to get the fw_id
        # no need to deserialize the whole FW

        if '_add_launchpad_and_fw_id' in fw_spec:
            lp = self.launchpad
            fw_id = self.fw_id
        else:
            try:
                fw_dict = loadfn('FW.json')
            except IOError:
                try:
                    fw_dict = loadfn('FW.yaml')
                except IOError:
                    raise RuntimeError("Launchpad/fw_id not present in spec and No FW.json nor FW.yaml file present: "
                                       "impossible to determine fw_id")
            lp = LaunchPad.auto_load()
            fw_id = fw_dict['fw_id']

        wf = lp.get_wf_by_fw_id_lzyfw(fw_id)

        deleted_files = []
        # iterate over all the fws and launches
        for fw_id, fw in wf.id_fw.items():
            for l in fw.launches+fw.archived_launches:
                l_dir = l.launch_dir

                deleted_files.extend(self.delete_files(os.path.join(l_dir, TMPDIR_NAME)))
                deleted_files.extend(self.delete_files(os.path.join(l_dir, INDIR_NAME)))
                deleted_files.extend(self.delete_files(os.path.join(l_dir, OUTDIR_NAME), self.out_exts))

        logging.info("Deleted files:\n {}".format("\n".join(deleted_files)))

        return FWAction(stored_data={'deleted_files': deleted_files})


@explicit_serialize
class DatabaseInsertTask(FireTaskBase):
    task_type = 'dbinsert'

    def __init__(self, insertion_data={'structure': 'get_final_structure_and_history'}, criteria=None):
        self.insertion_data = insertion_data
        self.criteria = criteria

    @serialize_fw
    def to_dict(self):
        return dict(insertion_data=self.insertion_data, criteria=self.criteria)

    @classmethod
    def from_dict(cls, m_dict):
        return cls(insertion_data=m_dict['insertion_data'],
                   criteria=m_dict['criteria'] if 'criteria' in m_dict else None)

    @staticmethod
    def insert_objects():
        return None

    def run_task(self, fw_spec):
        # the FW.json/yaml file is mandatory to get the fw_id
        # no need to deserialize the whole FW

        if '_add_launchpad_and_fw_id' in fw_spec:
            lp = self.launchpad
            fw_id = self.fw_id
        else:
            try:
                fw_dict = loadfn('FW.json')
            except IOError:
                try:
                    fw_dict = loadfn('FW.yaml')
                except IOError:
                    raise RuntimeError("Launchpad/fw_id not present in spec and No FW.json nor FW.yaml file present: "
                                       "impossible to determine fw_id")
            lp = LaunchPad.auto_load()
            fw_id = fw_dict['fw_id']

        wf = lp.get_wf_by_fw_id(fw_id)
        wf_module = importlib.import_module(wf.metadata['workflow_module'])
        wf_class = getattr(wf_module, wf.metadata['workflow_class'])

        database = MongoDatabase.from_dict(fw_spec['mongo_database'])
        if self.criteria is not None:
            entry = database.get_entry(criteria=self.criteria)
        else:
            entry = {}

        inserted = []
        for root_key, method_name in self.insertion_data.items():
            get_results_method = getattr(wf_class, method_name)
            results = get_results_method(wf)
            for key, val in results.items():
                entry[key] = jsanitize(val)
                inserted.append(key)

        if self.criteria is not None:
            database.save_entry(entry=entry)
        else:
            database.insert_entry(entry=entry)

        logging.info("Inserted data:\n{}".format('- {}\n'.join(inserted)))
        return FWAction()


@explicit_serialize
class CheckMemoryTask(FireTaskBase):
    task_type = 'checkmem'

    def __init__(self, memory_increase_megabytes=1000, max_memory_megabytes=7600):
        self.memory_increase_megabytes = memory_increase_megabytes
        self.max_memory_megabytes = max_memory_megabytes

    @serialize_fw
    def to_dict(self):
        return dict(memory_increase_megabytes=self.memory_increase_megabytes,
                    max_memory_megabytes=self.max_memory_megabytes)

    @classmethod
    def from_dict(cls, m_dict):
        return cls(memory_increase_megabytes=m_dict['memory_increase_megabytes'],
                   max_memory_megabytes=m_dict['max_memory_megabytes'])

    def run_task(self, fw_spec):

        # Get the fw_id and launchpad
        if '_add_launchpad_and_fw_id' in fw_spec:
            lp = self.launchpad
            fw_id = self.fw_id
        else:
            try:
                fw_dict = loadfn('FW.json')
            except IOError:
                try:
                    fw_dict = loadfn('FW.yaml')
                except IOError:
                    raise RuntimeError("Launchpad/fw_id not present in spec and No FW.json nor FW.yaml file present: "
                                       "impossible to determine fw_id")
            lp = LaunchPad.auto_load()
            fw_id = fw_dict['fw_id']

        # Treat the case where there was no memory error => forward "needed" outputs of the previous firework to the
        # next one.
        if not '_fizzled_parents' in fw_spec:
            stored_data = {}
            update_spec = {}
            mod_spec = []
            for task_type, task_info in fw_spec['previous_fws'].items():
                mod_spec.append({'_push': {'previous_fws->'+task_type: task_info}})
            return FWAction(stored_data=stored_data, update_spec=update_spec, mod_spec=mod_spec)

        if len(fw_spec['_fizzled_parents']) > 1:
            raise ValueError('Multiple parents fizzled ... Only one is allowed.')

        fizzled_fw_id = Firework.from_dict(fw_spec['_fizzled_parents'][0]['fw_id'])
        fizzled_fw = lp.get_fw_by_id(fizzled_fw_id)
        fizzled_fw_dir = fizzled_fw.launches[-1].launch_dir

        manager = self.get_fw_task_manager(fw_spec=fw_spec)

        # Analyze the stderr and stdout files of the resource manager system.
        qerr_info = None
        qout_info = None
        qerr_file = os.path.join(fizzled_fw_dir, 'queue.qerr')
        qout_file = os.path.join(fizzled_fw_dir, 'queue.qout')
        runerr_file = fw_spec['runerr_file'] if 'runerr_file' in fw_spec else None
        if os.path.exists(qerr_file):
            with open(qerr_file, "r") as f:
                qerr_info = f.read()
        if os.path.exists(qout_file):
            with open(qout_file, "r") as f:
                qout_info = f.read()

        if qerr_info or qout_info:
            from pymatgen.io.abinitio.scheduler_error_parsers import get_parser
            scheduler_parser = get_parser(manager.qadapter.QTYPE, err_file=qerr_file,
                                          out_file=qout_file, run_err_file=runerr_file)

            if scheduler_parser is None:
                raise ValueError('Cannot find scheduler_parser for qtype {}'.format(manager.qadapter.QTYPE))

            scheduler_parser.parse()

            if scheduler_parser.errors:
                # the queue errors in the task
                logger.debug('scheduler errors found:')
                logger.debug(str(scheduler_parser.errors))
                queue_errors = scheduler_parser.errors
            else:
                if len(qerr_info) > 0:
                    logger.debug('found unknown queue error: {}'.format(str(qerr_info)))
                    raise ValueError(qerr_info)
                    # The job is killed or crashed but we don't know what happened
        return FWAction()

    def get_fw_task_manager(self, fw_spec):
        if 'ftm_file' in fw_spec:
            ftm = FWTaskManager.from_file(fw_spec['ftm_file'])
        else:
            ftm = FWTaskManager.from_user_config()
        ftm.update_fw_policy(fw_spec.get('fw_policy', {}))
        return ftm