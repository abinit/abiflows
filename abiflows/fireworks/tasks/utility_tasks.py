# coding: utf-8
"""
Utility tasks for Fireworks.
"""
from __future__ import print_function, division, unicode_literals, absolute_import

import copy
import os
import shutil
import logging
import traceback
import importlib
import json

from monty.json import jsanitize
from monty.json import MontyDecoder
from custodian.ansible.interpreter import Modder
from fireworks.core.firework import Firework, FireTaskBase, FWAction, Workflow
from fireworks.core.launchpad import LaunchPad
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.utilities.fw_serializers import serialize_fw
from abiflows.fireworks.tasks.abinit_common import TMPDIR_NAME, OUTDIR_NAME, INDIR_NAME
from abiflows.fireworks.utils.custodian_utils import SRCErrorHandler
from abiflows.fireworks.utils.fw_utils import set_short_single_core_to_spec, FWTaskManager, get_lp_and_fw_id_from_task
from abiflows.database.mongoengine.utils import DatabaseData


logger = logging.getLogger(__name__)


SRC_TIMELIMIT_BUFFER = 120


#TODO: make it possible to use "any" task and in particular, MergeDdbTask, AnaDdbTask, ...
# within the SRC scheme (to be rationalized), as well as Task not related to abinit ...
def createSRCFireworksOld(task_class, task_input, SRC_spec, initialization_info, wf_task_index_prefix, current_task_index=1,
                          handlers=None, validators=None,
                          deps=None, task_type=None, queue_adapter_update=None):
    SRC_spec = copy.deepcopy(SRC_spec)
    SRC_spec['initialization_info'] = initialization_info
    SRC_spec['_add_launchpad_and_fw_id'] = True
    SRC_spec['SRCScheme'] = True
    prefix_allowed_chars = ['-']
    prefix_test_string = str(wf_task_index_prefix)
    for allowed_char in prefix_allowed_chars:
        prefix_test_string = prefix_test_string.replace(allowed_char, "")
    if not prefix_test_string.isalnum():
        raise ValueError('wf_task_index_prefix should only contain letters '
                         'and the following characters : {}'.format(prefix_test_string))
    SRC_spec['wf_task_index_prefix'] = wf_task_index_prefix

    # Remove any initial queue_adapter_update from the spec
    SRC_spec.pop('queue_adapter_update', None)
    if queue_adapter_update is not None:
        SRC_spec['queue_adapter_update'] = queue_adapter_update

    # Setup (Autoparal) run
    SRC_spec_setup = copy.deepcopy(SRC_spec)
    SRC_spec_setup = set_short_single_core_to_spec(SRC_spec_setup)
    SRC_spec_setup['wf_task_index'] = '_'.join(['setup', wf_task_index_prefix, str(current_task_index)])
    setup_task = task_class(task_input, is_autoparal=True, use_SRC_scheme=True, deps=deps, task_type=task_type)
    setup_fw = Firework(setup_task, spec=SRC_spec_setup, name=SRC_spec_setup['wf_task_index'])
    # Actual run of simulation
    SRC_spec_run = copy.deepcopy(SRC_spec)
    SRC_spec_run['wf_task_index'] = '_'.join(['run', wf_task_index_prefix, str(current_task_index)])
    run_task = task_class(task_input, is_autoparal=False, use_SRC_scheme=True, deps=deps, task_type=task_type)
    run_fw = Firework(run_task, spec=SRC_spec_run, name=SRC_spec_run['wf_task_index'])
    # Check memory firework
    SRC_spec_check = copy.deepcopy(SRC_spec)
    SRC_spec_check = set_short_single_core_to_spec(SRC_spec_check)
    SRC_spec_check['wf_task_index'] = '_'.join(['check', wf_task_index_prefix, str(current_task_index)])
    check_task = CheckTask(handlers=handlers, validators=validators)
    SRC_spec_check['_allow_fizzled_parents'] = True
    check_fw = Firework(check_task, spec=SRC_spec_check, name=SRC_spec_check['wf_task_index'])
    links_dict = {setup_fw.fw_id: [run_fw.fw_id],
                  run_fw.fw_id: [check_fw.fw_id]}
    return {'setup_fw': setup_fw, 'run_fw': run_fw, 'check_fw': check_fw, 'links_dict': links_dict,
            'fws': [setup_fw, run_fw, check_fw]}


@explicit_serialize
class FinalCleanUpTask(FireTaskBase):
    """
    Eliminates all the files in outdata based on the extension of the files and deletes of the files in
    indata and tmpdata.
    """
    task_type = 'finalclnup'

    def __init__(self, out_exts=None):
        """
        Args:
            out_exts: list of extensions. can be a list or a single string with extensions separated by a ",".
                If None "WFK" and "1WF" will be deleted
        """
        if out_exts is None:
            out_exts = ["WFK", "1WF"]
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
        if exts and not isinstance(exts, (list, tuple)):
            exts = [exts]
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
                        elif os.path.islink(fp):
                            os.unlink(fp)
                        deleted_files.append(fp)
                    except:
                        logger.warning("Couldn't delete {}: {}".format(fp, traceback.format_exc()))

        return deleted_files

    def run_task(self, fw_spec):
        lp, fw_id = get_lp_and_fw_id_from_task(self, fw_spec)

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

    def __init__(self, insertion_data=None, criteria=None):
        if insertion_data is None:
            insertion_data = {'structure': 'get_final_structure_and_history'}
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
        lp, fw_id = get_lp_and_fw_id_from_task(self, fw_spec)

        wf = lp.get_wf_by_fw_id(fw_id)
        wf_module = importlib.import_module(wf.metadata['workflow_module'])
        wf_class = getattr(wf_module, wf.metadata['workflow_class'])

        database = fw_spec['mongo_database']
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
class MongoEngineDBInsertionTask(FireTaskBase):
    """
    Task that insert the results of a workflow in a database. The method calls the "get_mongoengine_results"
    of the workflow generator.
    The database is defined according to a DatabaseData.
    """

    def __init__(self, db_data, object_id=None):
        self.db_data = db_data
        #TODO allow for updates based on object_id. A bit tricky as one need to know the class of the Document
        # before calling get_results. (add object_id to dict methods)
        self.object_id = object_id

    @serialize_fw
    def to_dict(self):
        return dict(db_data=self.db_data.as_dict())

    @classmethod
    def from_dict(cls, m_dict):
        return cls(db_data=DatabaseData.from_dict(m_dict['db_data']))

    def run_task(self, fw_spec):
        self.db_data.connect_mongoengine()

        lp, fw_id = get_lp_and_fw_id_from_task(self, fw_spec)

        wf = lp.get_wf_by_fw_id_lzyfw(fw_id)
        wf_module = importlib.import_module(wf.metadata['workflow_module'])
        wf_class = getattr(wf_module, wf.metadata['workflow_class'])

        get_results_method = getattr(wf_class, 'get_mongoengine_results')

        #TODO extend for multiple documents?
        document = get_results_method(wf)

        with self.db_data.switch_collection(document.__class__) as document.__class__:
            #TODO it would be better to try to remove automatically the FileFields already saved if the save of
            # the document fails.
            document.save()


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

    # def run_task(self, fw_spec):
    #
    #     # Get the fw_id and launchpad
    #     if '_add_launchpad_and_fw_id' in fw_spec:
    #         lp = self.launchpad
    #         fw_id = self.fw_id
    #     else:
    #         try:
    #             fw_dict = loadfn('FW.json')
    #         except IOError:
    #             try:
    #                 fw_dict = loadfn('FW.yaml')
    #             except IOError:
    #                 raise RuntimeError("Launchpad/fw_id not present in spec and No FW.json nor FW.yaml file present: "
    #                                    "impossible to determine fw_id")
    #         lp = LaunchPad.auto_load()
    #         fw_id = fw_dict['fw_id']
    #
    #     # Treat the case where there was no memory error => forward "needed" outputs of the previous firework to the
    #     # next one.
    #     if not '_fizzled_parents' in fw_spec:
    #         stored_data = {}
    #         update_spec = {}
    #         mod_spec = []
    #         for task_type, task_info in fw_spec['previous_fws'].items():
    #             mod_spec.append({'_push_all': {'previous_fws->'+task_type: task_info}})
    #         return FWAction(stored_data=stored_data, update_spec=update_spec, mod_spec=mod_spec)
    #
    #     if len(fw_spec['_fizzled_parents']) > 1:
    #         raise ValueError('Multiple parents fizzled ... Only one is allowed.')
    #
    #     fizzled_fw_id = fw_spec['_fizzled_parents'][0]['fw_id']
    #     fizzled_fw = lp.get_fw_by_id(fizzled_fw_id)
    #     fizzled_fw_dir = fizzled_fw.launches[-1].launch_dir
    #
    #     # Analyze the stderr and stdout files of the resource manager system.
    #     qerr_info = None
    #     qout_info = None
    #     qerr_file = os.path.join(fizzled_fw_dir, 'queue.qerr')
    #     qout_file = os.path.join(fizzled_fw_dir, 'queue.qout')
    #     runerr_file = fw_spec['runerr_file'] if 'runerr_file' in fw_spec else None
    #     if os.path.exists(qerr_file):
    #         with open(qerr_file, "r") as f:
    #             qerr_info = f.read()
    #     if os.path.exists(qout_file):
    #         with open(qout_file, "r") as f:
    #             qout_info = f.read()
    #
    #     if qerr_info or qout_info:
    #         from pymatgen.io.abinit.scheduler_error_parsers import get_parser
    #         qtk_qadapter = fizzled_fw.spec['qtk_queueadapter']
    #         qtype = qtk_qadapter.QTYPE
    #         scheduler_parser = get_parser(qtype, err_file=qerr_file,
    #                                       out_file=qout_file, run_err_file=runerr_file)
    #
    #         if scheduler_parser is None:
    #             raise ValueError('Cannot find scheduler_parser for qtype {}'.format(qtype))
    #
    #         scheduler_parser.parse()
    #         queue_errors = scheduler_parser.errors
    #
    #         if queue_errors:
    #             # the queue errors in the task
    #             logger.debug('scheduler errors found:')
    #             logger.debug(str(queue_errors))
    #         else:
    #             if len(qerr_info) > 0:
    #                 logger.debug('found unknown queue error: {}'.format(str(qerr_info)))
    #                 raise ValueError(qerr_info)
    #                 # The job is killed or crashed but we don't know what happened
    #
    #         to_be_corrected = False
    #         for error in queue_errors:
    #             if isinstance(error, MemoryCancelError):
    #                 logger.debug('found memory error.')
    #                 to_be_corrected = True
    #         if to_be_corrected:
    #             if len(fizzled_fw.tasks) > 1:
    #                 raise ValueError('More than 1 task found in "memory-fizzled" firework, not yet supported')
    #             logger.debug('adding SRC detour')
    #             mytask = fizzled_fw.tasks[0]
    #             task_class = mytask.__class__
    #             # TODO: make this more general ... right now, it is based on AbinitInput and thus is strongly tight
    #             #       to abinit
    #             task_input = AbinitInput.from_dict(fizzled_fw.spec['_tasks'][0]['abiinput'])
    #             spec = fizzled_fw.spec
    #             initialization_info = fizzled_fw.spec['initialization_info']
    #             # Update the task index
    #             fizzled_fw_task_index = int(fizzled_fw.spec['wf_task_index'].split('_')[-1])
    #             new_index = fizzled_fw_task_index + 1
    #             # Update the memory in the queue adapter
    #             old_mem = qtk_qadapter.mem_per_proc
    #             new_mem = old_mem + self.memory_increase_megabytes
    #             if new_mem > self.max_memory_megabytes:
    #                 raise ValueError('New memory {:d} is larger than '
    #                                  'max memory per proc {:d}'.format(new_mem, self.max_memory_megabytes))
    #             qtk_qadapter.set_mem_per_proc(new_mem)
    #             spec['qtk_queueadapter'] = qtk_qadapter
    #             qadapter_spec = qtk_qadapter.get_subs_dict()
    #             spec['_queueadapter'] = qadapter_spec
    #             if 'run_timelimit' in spec:
    #                 run_timelimit = spec['run_timelimit']
    #             else:
    #                 run_timelimit = None
    #
    #             SRC_fws = SRCFireworks(task_class=task_class, task_input=task_input, spec=spec,
    #                                    initialization_info=initialization_info,
    #                                    wf_task_index_prefix=spec['wf_task_index_prefix'],
    #                                    current_task_index=new_index,
    #                                    current_memory_per_proc_mb=new_mem,
    #                                    memory_increase_megabytes=self.memory_increase_megabytes,
    #                                    max_memory_megabytes=self.max_memory_megabytes,
    #                                    task_type=mytask.task_type, run_timelimit=run_timelimit)
    #             wf = Workflow(fireworks=SRC_fws['fws'], links_dict=SRC_fws['links_dict'])
    #             return FWAction(detours=[wf])
    #     raise ValueError('Could not check for memory problem ...')


@explicit_serialize
class CheckTask(FireTaskBase):
    task_type = 'check'

    optional_params = ['handlers', 'validators']

    def __init__(self, handlers=None, validators=None, max_restarts=10):
        #super(CheckTask).__init__(self)
        self.handlers = handlers
        # Check that there is only one FIRST and one LAST handler (PRIORITY_FIRST and PRIORITY_LAST)
        if self.handlers is not None:
            h_priorities = [h.handler_priority for h in self.handlers]
            nhfirst = h_priorities.count(SRCErrorHandler.PRIORITY_FIRST)
            nhlast = h_priorities.count(SRCErrorHandler.PRIORITY_LAST)
            if nhfirst > 1 or nhlast > 1:
                raise ValueError('More than one first or last handler :\n'
                                 ' - nfirst : {:d}\n - nlast : {:d}'.format(nhfirst,
                                                                            nhlast))
        self.validators = validators
        self.max_restarts = max_restarts

    @serialize_fw
    def to_dict(self):
        return dict(handlers=[h.as_dict() for h in self.handlers] if self.handlers is not None else None,
                    validators=[v.as_dict() for v in self.validators] if self.validators is not None else None,
                    max_restarts=self.max_restarts)

    @classmethod
    def from_dict(cls, m_dict):
        m = MontyDecoder()
        return cls(handlers=m.process_decoded(m_dict['handlers']), validators=m.process_decoded(m_dict['validators'])
                   , max_restarts=m_dict['max_restarts'])

    def run_task(self, fw_spec):
        # Get the fw_id and launchpad
        if '_add_launchpad_and_fw_id' in fw_spec:
            lp = self.launchpad
            fw_id = self.fw_id
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
            lp = LaunchPad.auto_load()
            fw_id = fw_dict['fw_id']

        # Treat the case where there was some error that led to a fizzled state
        if '_fizzled_parents' in fw_spec:
            if len(fw_spec['_fizzled_parents']) != 1:
                raise ValueError('CheckTask\'s Firework should have exactly one parent firework')
            # Get the fizzled fw
            fizzled_fw_id = fw_spec['_fizzled_parents'][0]['fw_id']
            fizzled_fw = lp.get_fw_by_id(fizzled_fw_id)
            # Sort handlers by their priority
            sorted_handlers = sorted([h for h in self.handlers if h.allow_fizzled], key=lambda x: x.handler_priority)
            # Get the corrections for all the handlers
            corrections = []
            for handler in sorted_handlers:
                # Set needed data for the handlers (the spec of this check task/fw and the fw that has to be checked)
                handler.src_setup(fw_spec=fw_spec, fw_to_check=fizzled_fw)
                if handler.check():
                    corrections.append(handler.correct())
                    if handler.skip_remaining_handlers:
                        break

            # In case of a fizzled parent, at least one correction is needed !
            if len(corrections) == 0:
                raise RuntimeError('No corrections found for fizzled firework ...')

            # Apply the corrections
            fw_action = self.apply_corrections(fw_to_correct=fizzled_fw, corrections=corrections)
            return fw_action
        # Treat the case where there was no fizzled parents => forward "needed" outputs of the previous firework to the
        # next one.
        else:
            # Get the previous fw
            this_lzy_wf = lp.get_wf_by_fw_id_lzyfw(fw_id)
            parents_fw_ids = this_lzy_wf.links.parent_links[fw_id]
            if len(parents_fw_ids) != 1:
                raise ValueError('CheckTask\'s Firework should have exactly one parent firework')
            run_fw = lp.get_fw_by_id(parents_fw_ids[0])
            # Get the corrections for all the handlers
            # Sort handlers by their priority
            if self.handlers is not None:
                sorted_handlers = sorted([h for h in self.handlers if h.allow_completed],
                                         key=lambda x: x.handler_priority)
            else:
                sorted_handlers = []
            # Get the corrections for all the handlers
            corrections = []
            for handler in sorted_handlers:
                # Set needed data for the handlers (the spec of this check task/fw and the fw that has to be checked)
                handler.src_setup(fw_spec=fw_spec, fw_to_check=run_fw)
                if handler.check():
                    corrections.append(handler.correct())
                if handler.skip_remaining_handlers:
                    break

            # If some corrections are found, apply and return the FWAction
            if len(corrections) > 0:
                fw_action = self.apply_corrections(fw_to_correct=run_fw, corrections=corrections)
                return fw_action

            # Validate the results if no error was found
            validators = self.validators if self.validators is not None else []
            for validator in validators:
                if not validator.check():
                    raise RuntimeError('Validator invalidate results ...')
            stored_data = {}
            update_spec = {}
            mod_spec = []
            for task_type, task_info in fw_spec['previous_fws'].items():
                mod_spec.append({'_push_all': {'previous_fws->'+task_type: task_info}})
            return FWAction(stored_data=stored_data, update_spec=update_spec, mod_spec=mod_spec)

    def apply_corrections(self, fw_to_correct, corrections):
        # Apply the corrections
        spec = fw_to_correct.spec
        modder = Modder()
        for correction in corrections:
            actions = correction['actions']
            for action in actions:
                if action['action_type'] == 'modify_object':
                    if action['object']['source'] == 'fw_spec':
                        myobject = spec[action['object']['key']]
                    else:
                        raise NotImplementedError('Object source "{}" not implemented in '
                                                  'CheckTask'.format(action['object']['source']))
                    newobj = modder.modify_object(action['action'], myobject)
                    spec[action['object']['key']] = newobj
                elif action['action_type'] == 'modify_dict':
                    if action['dict']['source'] == 'fw_spec':
                        mydict = spec[action['dict']['key']]
                    else:
                        raise NotImplementedError('Dict source "{}" not implemented in '
                                                  'CheckTask'.format(action['dict']['source']))
                    modder.modify(action['action'], mydict)
                else:
                    raise NotImplementedError('Action type "{}" not implemented in '
                                              'CheckTask'.format(action['action_type']))
        # Keep track of the corrections that have been applied
        spec['SRC_check_corrections'] = corrections

        # Update the task index
        fws_task_index = int(fw_to_correct.spec['wf_task_index'].split('_')[-1])
        new_index = fws_task_index + 1
        # Update the Fireworks _queueadapter key
        #TODO: in the future, see whether the FW queueadapter might be replaced by the qtk_queueadapter ?
        #      ... to be discussed with Anubhav, when the qtk queueadapter is in a qtk toolkit and not anymore
        #          in pymatgen/io/abinit
        spec['_queueadapter'] = spec['qtk_queueadapter'].get_subs_dict()
        queue_adapter_update = get_queue_adapter_update(qtk_queueadapter=spec['qtk_queueadapter'],
                                                        corrections=corrections)

        # Get and update the task_input if needed
        # TODO: make this more general ... right now, it is based on AbinitInput and thus is strongly tight
        #       to abinit due to abiinput, deps, ...
        mytask = fw_to_correct.tasks[0]
        task_class = mytask.__class__
        decoder = MontyDecoder()
        task_input = decoder.process_decoded(fw_to_correct.spec['_tasks'][0]['abiinput'])
        initialization_info = fw_to_correct.spec['initialization_info']
        deps = mytask.deps

        # Create the new Setup/Run/Check fireworks
        SRC_fws = createSRCFireworksOld(task_class=task_class, task_input=task_input, SRC_spec=spec,
                                        initialization_info=initialization_info,
                                        wf_task_index_prefix=spec['wf_task_index_prefix'],
                                        current_task_index=new_index,
                                        handlers=self.handlers, validators=self.validators,
                                        deps=deps,
                                        task_type=mytask.task_type, queue_adapter_update=queue_adapter_update)
        wf = Workflow(fireworks=SRC_fws['fws'], links_dict=SRC_fws['links_dict'])
        return FWAction(detours=[wf])


def print_myself():
    print('myself')

def get_fw_task_manager(fw_spec):
    if 'ftm_file' in fw_spec:
        ftm = FWTaskManager.from_file(fw_spec['ftm_file'])
    else:
        ftm = FWTaskManager.from_user_config()
    ftm.update_fw_policy(fw_spec.get('fw_policy', {}))
    return ftm


def apply_corrections_to_spec(corrections, spec):
    modder = Modder()
    for correction in corrections:
        actions = correction['actions']
        for action in actions:
            if action['action_type'] == 'modify_object':
                if action['object']['source'] == 'fw_spec':
                    myobject = spec[action['object']['key']]
                else:
                    raise NotImplementedError('Object source "{}" not implemented in '
                                              'CheckTask'.format(action['object']['source']))
                newobj = modder.modify_object(action['action'], myobject)
                spec[action['object']['key']] = newobj
            elif action['action_type'] == 'modify_dict':
                if action['dict']['source'] == 'fw_spec':
                    mydict = spec[action['dict']['key']]
                else:
                    raise NotImplementedError('Dict source "{}" not implemented in '
                                              'CheckTask'.format(action['dict']['source']))
                modder.modify(action['action'], mydict)
            else:
                raise NotImplementedError('Action type "{}" not implemented in '
                                          'CheckTask'.format(action['action_type']))


def get_queue_adapter_update(qtk_queueadapter, corrections, qa_params=None):
    if qa_params is None:
        qa_params = ['timelimit', 'mem_per_proc', 'master_mem_overhead']
    queue_adapter_update = {}
    for qa_param in qa_params:
        if qa_param == 'timelimit':
            queue_adapter_update[qa_param] = qtk_queueadapter.timelimit
        elif qa_param == 'mem_per_proc':
            queue_adapter_update[qa_param] = qtk_queueadapter.mem_per_proc
        elif qa_param == 'master_mem_overhead':
            queue_adapter_update[qa_param] = qtk_queueadapter.master_mem_overhead
        else:
            raise ValueError('Wrong queue adapter parameter for update')
    for correction in corrections:
        for action in correction['actions']:
            if action['object']['key'] == 'qtk_queueadapter':
                qa_update = action['action']['_set']
                queue_adapter_update.update(qa_update)
    return queue_adapter_update
