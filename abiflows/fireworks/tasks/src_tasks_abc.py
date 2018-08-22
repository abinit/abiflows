from __future__ import print_function, division, unicode_literals, absolute_import

import abc
import copy
import inspect
import json
import os
import six

from fireworks.core.firework import FireTaskBase
from fireworks.core.firework import FWAction
from fireworks.core.firework import Workflow
from fireworks.core.firework import Firework
from fireworks.core.launchpad import LaunchPad
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.utilities.fw_serializers import serialize_fw

from monty.json import MontyDecoder, MontyEncoder
from monty.json import MSONable
from monty.serialization import loadfn
from monty.subprocess import Command

from abiflows.core.mastermind_abc import ControlProcedure, ControlledItemType
from abiflows.core.mastermind_abc import ControllerNote
from abiflows.core.mastermind_abc import Cleaner
from abiflows.fireworks.utils.fw_utils import set_short_single_core_to_spec, get_short_single_core_spec

RESTART_FROM_SCRATCH = ControllerNote.RESTART_FROM_SCRATCH
RESET_RESTART = ControllerNote.RESET_RESTART
SIMPLE_RESTART = ControllerNote.SIMPLE_RESTART

class SRCTaskMixin(object):

    src_type = ''

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

    def setup_directories(self, fw_spec, create_dirs=False):
        if self.src_type == 'setup':
            self.src_root_dir = fw_spec.get('_launch_dir', os.getcwd())
        elif 'src_directories' in fw_spec:
            self.src_root_dir = fw_spec['src_directories']['src_root_dir']
        # elif self.src_type in ['run', 'control']:
        #     self.src_root_dir = os.path.split(os.path.abspath(fw_spec['_launch_dir']))[0]
        else:
            raise ValueError('Cannot setup directories for "src_type" = "{}"'.format(self.src_type))
        self.setup_dir = os.path.join(self.src_root_dir, 'setup')
        self.run_dir = os.path.join(self.src_root_dir, 'run')
        self.control_dir = os.path.join(self.src_root_dir, 'control')
        # if 'src_directories' in fw_spec:
        #     if (self.src_root_dir != fw_spec['src_directories']['src_root_dir'] or
        #         self.setup_dir != fw_spec['src_directories']['setup_dir'] or
        #         self.run_dir != fw_spec['src_directories']['run_dir'] or
        #         self.control_dir != fw_spec['src_directories']['control_dir']):
        #         raise ValueError('src_directories in fw_spec do not match actual SRC directories ...')
        if create_dirs:
            os.makedirs(self.setup_dir)
            os.makedirs(self.run_dir)
            os.makedirs(self.control_dir)

    @property
    def src_directories(self):
        return {'src_root_dir': self.src_root_dir,
                'setup_dir': self.setup_dir,
                'run_dir': self.run_dir,
                'control_dir': self.control_dir
                }


@explicit_serialize
class SetupTask(SRCTaskMixin, FireTaskBase):

    src_type = 'setup'
    task_type = 'unknown'

    RUN_PARAMETERS = ['_queueadapter', 'mpi_ncpus', 'qtk_queueadapter']

    def __init__(self, deps=None, restart_info=None, task_type=None):

        # TODO: if at some point, this is not enough (as for the ddk, or for the structure, or for anything else,
        #       we could thing of an object ?
        # deps are transformed to be a list or a dict of lists
        if isinstance(deps, dict):
            deps = dict(deps)
            for k, v in deps.items():
                if not isinstance(v, (list, tuple)):
                    deps[k] = [v]
        elif deps and not isinstance(deps, (list, tuple)):
            deps = [deps]
        self.deps = deps

        self.restart_info = restart_info

        if task_type is not None:
            self.task_type = task_type

    def set_restart_info(self, restart_info=None):
        self.restart_info = restart_info

    def run_task(self, fw_spec):
        # Set up and create the directory tree of the Setup/Run/Control trio,
        self.setup_directories(fw_spec=fw_spec, create_dirs=True)
        #  Forward directory information to run and control fireworks #HACK in _setup_run_and_control_dirs
        self._setup_run_and_control_dirs_and_fworker(fw_spec=fw_spec)
        # Move to the setup directory
        os.chdir(self.setup_dir)
        # Make the file transfers from another worker if needed
        self.file_transfers(fw_spec=fw_spec)
        # Get back information from the previous runs
        self.fetch_previous_info(fw_spec=fw_spec)
        # Setup the parameters for the run (number of cpus, time, memory, openmp, ...)
        params = list(self.RUN_PARAMETERS)
        if 'src_modified_objects' in fw_spec:
            for target, modified_object in fw_spec['src_modified_objects'].items():
                params.remove(target)
        run_parameters = self._setup_run_parameters(fw_spec=fw_spec, parameters=params)
        # Prepare run (make links to output files from previous tasks, write input files, create the directory
        # tree of the program, ...)
        self.prepare_run(fw_spec=fw_spec)

        # Update the spec of the Run Firework with the directory tree, the run_parameters obtained from
        #  setup_run_parameters and the modified_objects transferred directly from the Control Firework
        update_spec = {'src_directories': self.src_directories}
        update_spec.update(run_parameters)
        if 'src_modified_objects' in fw_spec:
            update_spec.update(fw_spec['src_modified_objects'])
        if 'previous_fws' in fw_spec:
            update_spec['previous_fws'] = fw_spec['previous_fws']
        return FWAction(update_spec=update_spec)

    def _setup_run_parameters(self, fw_spec, parameters):
        qadapter_spec, qtk_queueadapter = get_short_single_core_spec(return_qtk=True)
        params = {'_queueadapter': qadapter_spec, 'mpi_ncpus': 1, 'qtk_queueadapter': qtk_queueadapter}
        setup_params = self.setup_run_parameters(fw_spec=fw_spec)
        params.update(setup_params)
        if 'initial_parameters' in fw_spec and fw_spec['SRC_task_index'].index == 1:
            qtk_queueadapter = params['qtk_queueadapter']
            initial_parameters = fw_spec['initial_parameters']
            if 'run_timelimit' in initial_parameters:
                qtk_queueadapter.set_timelimit(timelimit=initial_parameters['run_timelimit'])
            if 'run_mem_per_proc' in initial_parameters:
                qtk_queueadapter.set_mem_per_proc(mem_mb=initial_parameters['run_mem_per_proc'])
            if 'run_mpi_ncpus' in initial_parameters:
                qtk_queueadapter.set_mpi_procs(mpi_procs=initial_parameters['run_mpi_ncpus'])
            qadapter_spec = qtk_queueadapter.get_subs_dict()
            params.update({'qtk_queueadapter': qtk_queueadapter, '_queueadapter': qadapter_spec})
        return {param: params[param] for param in parameters}

    def setup_run_parameters(self, fw_spec):
        return {}

    def file_transfers(self, fw_spec):
        pass

    def fetch_previous_info(self, fw_spec):
        pass

    def prepare_run(self, fw_spec):
        pass

    def _setup_run_and_control_dirs_and_fworker(self, fw_spec):
        """
        This method is used to update the spec of the run and control fireworks with the src_directories as well as
        set the _launch_dir of the run and control fireworks to be the run_dir and control_dir respectively.
        WARNING: This is a bit hackish! Do not change this unless you know exactly what you are doing!
        :param fw_spec: Firework's spec
        """
        # Get the launchpad
        if '_add_launchpad_and_fw_id' in fw_spec:
            lp = self.launchpad
            setup_fw_id = self.fw_id
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
            setup_fw_id = fw_dict['fw_id']
        if '_add_fworker' in fw_spec:
            fworker = self.fworker
        else:
            raise ValueError('Should have access to the fworker in SetupTask ...')
        this_lzy_wf = lp.get_wf_by_fw_id_lzyfw(setup_fw_id)
        # Check that SetupTask and RunTask have only one child firework
        child_fw_ids = this_lzy_wf.links[setup_fw_id]
        if len(child_fw_ids) != 1:
            raise ValueError('SetupTask\'s Firework should have exactly one child firework')
        run_fw_id = child_fw_ids[0]
        child_run_fw_ids = this_lzy_wf.links[run_fw_id]
        if len(child_run_fw_ids) != 1:
            raise ValueError('RunTask\'s Firework should have exactly one child firework')
        control_fw_id = child_run_fw_ids[0]
        spec_update = {'_launch_dir': self.run_dir,
                       'src_directories': self.src_directories,
                       '_fworker': fworker.name}
        lp.update_spec(fw_ids=[run_fw_id],
                       spec_document=spec_update)
        spec_update['_launch_dir'] = self.control_dir
        lp.update_spec(fw_ids=[control_fw_id],
                       spec_document=spec_update)

    def additional_task_info(self):
        return {}


class RunTask(SRCTaskMixin, FireTaskBase):

    src_type = 'run'
    task_type = 'unknown'

    def __init__(self, control_procedure, task_type=None):
        self.set_control_procedure(control_procedure=control_procedure)
        if task_type is not None:
            self.task_type = task_type

    def set_control_procedure(self, control_procedure):
        self.control_procedure = control_procedure
        #TODO: check something here with the monitors ?

    def run_task(self, fw_spec):
        self.setup_directories(fw_spec=fw_spec, create_dirs=False)
        launch_dir = os.getcwd()
        # Move to the run directory
        os.chdir(self.run_dir)
        f = open(os.path.join(self.run_dir, 'fw_info.txt'), 'a')
        f.write('FW launch_directory :\n{}'.format(launch_dir))
        f.close()
        # The Run and Control tasks have to run on the same worker

        #TODO: do something here with the monitoring controllers ...
        #      should stop the RunTask but the correction should be applied in control !
        self.config(fw_spec=fw_spec)
        self.run(fw_spec=fw_spec)
        update_spec = self.postrun(fw_spec=fw_spec)

        if update_spec is None:
            update_spec = {}

        if 'previous_fws' in fw_spec:
            update_spec['previous_fws'] = fw_spec['previous_fws']

        #TODO: the directory is passed thanks to _pass_job_info. Should we pass anything else ?
        return FWAction(stored_data=None, exit=False, update_spec=None, mod_spec=None,
                        additions=None, detours=None,
                        defuse_children=False)

    def config(self, fw_spec):
        pass

    @abc.abstractmethod
    def run(self, fw_spec):
        pass

    def postrun(self, fw_spec):
        pass

    def additional_task_info(self):
        return {}


@explicit_serialize
class ScriptRunTask(RunTask):

    task_type = 'script'

    def __init__(self, script_str, control_procedure):
        RunTask.__init__(self, control_procedure=control_procedure)
        self.script_str = script_str

    def run(self, fw_spec):
        f = open('script_run.log', 'w')
        cmds_strs = self.script_str.split(';')
        for cmd_str in cmds_strs:
            cmd = Command(cmd_str)
            cmd = cmd.run()
            if cmd.retcode != 0:
                raise ValueError('Command "{}" returned exit code {:d}'.format(cmd_str, cmd.retcode))
            if cmd.output is not None:
                print(cmd.output)
            f.write('{}\n'.format(str(cmd)))
        f.close()

    @serialize_fw
    def to_dict(self):
        return {'script_str': self.script_str,
                'control_procedure': self.control_procedure.as_dict()}

    @classmethod
    def from_dict(cls, d):
        control_procedure = ControlProcedure.from_dict(d['control_procedure'])
        return cls(script_str=d['script_str'], control_procedure=control_procedure)


@explicit_serialize
class ControlTask(SRCTaskMixin, FireTaskBase):
    src_type = 'control'

    def __init__(self, control_procedure, manager=None, max_restarts=10, src_cleaning=None):
        self.control_procedure = control_procedure
        self.manager = manager
        self.max_restarts = max_restarts
        self.src_cleaning = src_cleaning

    def run_task(self, fw_spec):
        self.setup_directories(fw_spec=fw_spec, create_dirs=False)
        launch_dir = os.getcwd()
        # Move to the control directory
        os.chdir(self.control_dir)
        f = open(os.path.join(self.control_dir, 'fw_info.txt'), 'a')
        f.write('FW launch_directory :\n{}'.format(launch_dir))
        f.close()
        # Get the task index
        task_index = SRCTaskIndex.from_any(fw_spec['SRC_task_index'])
        # Get the setup and run fireworks
        setup_and_run_fws = self.get_setup_and_run_fw(fw_spec=fw_spec)
        self.setup_fw = setup_and_run_fws['setup_fw']
        self.run_fw = setup_and_run_fws['run_fw']

        # Specify the type of the task that is controlled:
        #  - aborted : the task has been aborted due to a monitoring controller during the Run Task, the FW state
        #              is COMPLETED
        #  - completed : the task has completed, the FW state is COMPLETE
        #  - failed : the task has failed, the FW state is FIZZLED
        if self.run_fw.state == 'COMPLETED':
            if 'src_run_task_aborted' in fw_spec:
                self.control_procedure.set_controlled_item_type(ControlledItemType.task_aborted())
            else:
                self.control_procedure.set_controlled_item_type(ControlledItemType.task_completed())
        elif self.run_fw.state == 'FIZZLED':
            self.control_procedure.set_controlled_item_type(ControlledItemType.task_failed())
        else:
            raise RuntimeError('The state of the Run Firework is "{}" '
                               'while it should be COMPLETED or FIZZLED'.format(self.run_fw.state))

        # Get the keyword_arguments to be passed to the process method of the control_procedure
        #TODO: how to do that kind of automatically ??
        # each key should have : how to get it from the run_fw/(setup_fw)
        #                        how to force/apply it to the next SRC (e.g. how do we say to setup that)
        # Actually, the object, can come from : the setup_fw or from the run_fw (from the setup_spec, from the run_spec,
        # from the setup_task or the run_task (or in general one of the tasks ...
        # even multiple tasks is not yet supported ... should it be ? or should we stay with only one task allways ?)
        # If it is modified, it should update the corresponding bit (setup_spec and/or run_spec and/or
        # setup_task and/or run_task)
        initial_objects_info = self.get_initial_objects_info(setup_fw=self.setup_fw, run_fw= self.run_fw,
                                                             src_directories=self.src_directories)
        qerr_filepath = os.path.join(self.run_fw.launches[-1].launch_dir, 'queue.qerr')
        qout_filepath = os.path.join(self.run_fw.launches[-1].launch_dir, 'queue.qout')
        initial_objects_info.update({'queue_adapter': {'object': self.run_fw.spec['qtk_queueadapter'],
                                                       'updates': [{'target': 'fw_spec',
                                                                    'key': 'qtk_queueadapter'},
                                                                   {'target': 'fw_spec',
                                                                    'key': '_queueadapter',
                                                                    'mod': 'get_subs_dict'}]},
                                     'qerr_filepath': {'object': qerr_filepath},
                                     'qout_filepath': {'object': qout_filepath}})
        initial_objects = {name: obj_info['object'] for name, obj_info in initial_objects_info.items()}
        control_report = self.control_procedure.process(**initial_objects)

        if control_report.unrecoverable:
            f = open(os.path.join(self.control_dir, 'control_report.json'), 'w')
            json.dump(control_report.as_dict(), f)
            f.close()
            #TODO: apply the cleaning here
            if self.src_cleaning is not None:
                pass
            raise ValueError('Errors are unrecoverable. Control report written in "control_report.json"')

        # If everything is ok, update the spec of the children
        if control_report.finalized:
            stored_data = {'control_report': control_report, 'finalized': True}
            update_spec = {}
            mod_spec = []
            run_task = self.run_fw.tasks[-1]
            setup_task = self.setup_fw.tasks[-1]
            task_type = run_task.task_type
            #TODO: should we also add here the cluster in which the calculation was performed so that if the next
            #      SRC trio starts on another cluster, it should fetch the needed files from the run_dir of this cluster
            task_info = {'dir': self.run_dir}
            task_info.update(run_task.additional_task_info())
            task_info.update(setup_task.additional_task_info())
            mod_spec.append({'_push': {'previous_fws->'+task_type: task_info}})
            if self.src_cleaning is not None:
                pass
            return FWAction(stored_data=stored_data, exit=False, update_spec=update_spec, mod_spec=mod_spec,
                            additions=None, detours=None, defuse_children=False)

        # Check the maximum number of restarts
        if task_index.index == self.max_restarts:
            # TODO: decide when to apply cleaning here ?
            if self.src_cleaning is not None:
                pass
            raise ValueError('Maximum number of restarts ({:d}) reached'.format(self.max_restarts))

        # Increase the task_index
        task_index.increase_index()

        # Apply the actions on the objects to get the modified objects (to be passed to SetupTask)
        # modified_objects = {}
        # for target, action in control_report.actions.items():
        #     # Special case right now for the queue adapter ...
        #     if target == 'queue_adapter':
        #         qtk_qadapter = initial_objects[target]
        #         action.apply(qtk_qadapter)
        #         modified_objects['qtk_queueadapter'] = qtk_qadapter
        #         modified_objects['_queueadapter'] = qtk_qadapter.get_subs_dict()
        #     else:
        #         modified_objects[target] = action.apply(initial_objects[target])

        # New spec
        # remove "_tasks" which is present in spec for recent fireworks versions. Remove it here to avoid
        # problems with deepcopy.
        new_spec = dict(self.run_fw.spec)
        new_spec.pop("_tasks", None)
        new_spec = copy.deepcopy(new_spec)

        # New tasks
        setup_task = self.setup_fw.tasks[-1]
        run_task = self.run_fw.tasks[-1]
        control_task = self

        if 'src_modified_objects' in fw_spec:
            modified_objects = fw_spec['src_modified_objects']
        else:
            modified_objects = {}
        setup_spec_update = {}
        run_spec_update = {}
        for target, action in control_report.actions.items():
            target_object = initial_objects[target]
            action.apply(target_object)
            if target not in initial_objects_info:
                raise ValueError('Object "{}" to be modified was not in the initial_objects'.format(target))
            if 'updates' not in initial_objects_info[target]:
                raise ValueError('Update information not present for object "{}"'.format(target))
            for update in initial_objects_info[target]['updates']:
                if update['target'] == 'fw_spec':
                    if 'mod' in update:
                        mod = getattr(target_object, update['mod'])()
                        new_spec[update['key']] = mod
                        modified_objects[update['key']] = mod
                    else:
                        new_spec[update['key']] = target_object
                        modified_objects[update['key']] = target_object
                elif update['target'] == 'setup_fw_spec':
                    if 'mod' in update:
                        mod = getattr(target_object, update['mod'])()
                        setup_spec_update[update['key']] = mod
                    else:
                        setup_spec_update[update['key']] = target_object
                elif update['target'] == 'run_fw_spec':
                    if 'mod' in update:
                        mod = getattr(target_object, update['mod'])()
                        run_spec_update[update['key']] = mod
                    else:
                        run_spec_update[update['key']] = target_object
                elif update['target'] in ['setup_task', 'run_task']:
                    task = setup_task if update['target'] == 'setup_task' else run_task
                    attr = getattr(task, update['attribute'])
                    if 'mod' in update:
                        mod = getattr(target_object, update['mod'])()
                        attr = mod
                    else:
                        attr = target_object
                elif 'setup_task' in update['target']:
                    sp = update['target'].split('.')
                    if len(sp) != 2:
                        raise ValueError('target is "{}" and contains more than 1 "."'.format(update['target']))
                    if sp[0] != 'setup_task':
                        raise ValueError('target does not start with "setup_task" ...')
                    task = setup_task
                    task_attribute = getattr(task, sp[1])
                    if 'attribute' in update:
                        attr = getattr(task_attribute, update['attribute'])
                        if 'mod' in update:
                            mod = getattr(target_object, update['mod'])()
                            attr = mod
                        else:
                            attr = target_object
                    elif 'setter' in update:
                        setter = getattr(task_attribute, update['setter'])
                        if 'mod' in update:
                            mod = getattr(target_object, update['mod'])()
                            setter(mod)
                        else:
                            setter(target_object)
                else:
                    raise ValueError('Only changes to fw_spec, setup_task and run_task are allowed right now ...')

        # Set the restart_info
        setup_task.set_restart_info(control_report.restart_info)

        # Pass the modified objects to the next SetupTask
        new_spec['src_modified_objects'] = modified_objects
        new_spec.pop('_launch_dir')
        new_spec.pop('src_directories')
        new_spec['previous_src'] = {'src_directories': self.src_directories}
        if 'all_src_directories' in new_spec:
            new_spec['all_src_directories'].append({'src_directories': self.src_directories})
        else:
            new_spec['all_src_directories'] = [{'src_directories': self.src_directories}]
        # if '_queueadapter' in modified_objects:
        #     new_spec['_queueadapter'] = modified_objects['_queueadapter']
        #TODO: what to do here ? Right now this should work, just transfer information from the run_fw to the
        # next SRC group
        if 'previous_fws' in fw_spec:
            new_spec['previous_fws'] = fw_spec['previous_fws']
        # Create the new SRC trio
        # TODO: check initialization info, deps, ... previous_fws, ... src_previous_fws ? ...
        new_SRC_fws = createSRCFireworks(setup_task=setup_task, run_task=run_task, control_task=control_task,
                                         spec=new_spec, initialization_info=None, task_index=task_index,
                                         run_spec_update=run_spec_update, setup_spec_update=setup_spec_update)
        wf = Workflow(fireworks=new_SRC_fws['fws'], links_dict=new_SRC_fws['links_dict'])
        return FWAction(stored_data={'control_report': control_report}, detours=[wf])

    def get_setup_and_run_fw(self, fw_spec):
        # Get the launchpad
        if '_add_launchpad_and_fw_id' in fw_spec:
            lp = self.launchpad
            control_fw_id = self.fw_id
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
            control_fw_id = fw_dict['fw_id']
        # Check that this ControlTask has only one parent firework
        this_lzy_wf = lp.get_wf_by_fw_id_lzyfw(control_fw_id)
        parents_fw_ids = this_lzy_wf.links.parent_links[control_fw_id]
        if len(parents_fw_ids) != 1:
            raise ValueError('ControlTask\'s Firework should have exactly one parent firework')
        run_fw_id = parents_fw_ids[0]
        # Get the Run Firework and its state
        run_fw = lp.get_fw_by_id(fw_id=run_fw_id)
        run_is_fizzled = '_fizzled_parents' in fw_spec
        if run_is_fizzled and not run_fw.state == 'FIZZLED':
            raise ValueError('ControlTask has "_fizzled_parents" key but parent Run firework is not fizzled ...')
        run_is_completed = run_fw.state == 'COMPLETED'
        if run_is_completed and run_is_fizzled:
            raise ValueError('Run firework is FIZZLED and COMPLETED ...')
        if (not run_is_completed) and (not run_is_fizzled):
            raise ValueError('Run firework is neither FIZZLED nor COMPLETED ...')
        # Get the Setup Firework
        setup_job_info = run_fw.spec['_job_info'][-1]
        setup_fw_id = setup_job_info['fw_id']
        setup_fw = lp.get_fw_by_id(fw_id=setup_fw_id)
        return {'setup_fw': setup_fw, 'run_fw': run_fw}

    def get_initial_objects_info(self, setup_fw, run_fw, src_directories):
        return {}

    @classmethod
    def from_controllers(cls, controllers, max_restarts=10):
        cp = ControlProcedure(controllers=controllers)
        return cls(control_procedure=cp, max_restarts=max_restarts)

    @serialize_fw
    def to_dict(self):
        return {'control_procedure': self.control_procedure.as_dict(),
                'manager': self.manager.as_dict() if self.manager is not None else None,
                'max_restarts': self.max_restarts,
                'src_cleaning': self.src_cleaning.as_dict() if self.src_cleaning is not None else None}

    @classmethod
    def from_dict(cls, d):
        control_procedure = ControlProcedure.from_dict(d['control_procedure'])
        dec = MontyDecoder()
        if d['manager'] is None:
            manager = None
        else:
            manager = dec.process_decoded(d['manager']),
        if 'src_cleaning' in d:
            src_cleaning = SRCCleaning.from_dict(d['src_cleaning']) if d['src_cleaning'] is not None else None
        else:
            src_cleaning = None
        return cls(control_procedure=control_procedure, manager=manager, max_restarts=d['max_restarts'],
                   src_cleaning=src_cleaning)


class SRCCleanerOptions(MSONable):

    WHEN_TO_CLEAN = ['EACH_STEP', 'LAST_STEP', 'EACH_STEP_EXCEPT_LAST']
    CURRENT_SRC_STATES_ALLOWED = ['RECOVERABLE', 'UNRECOVERABLE', 'MAXRESTARTS', 'FINALIZED']

    def __init__(self, when_to_clean, current_src_states_allowed, which_src_steps_to_clean):
        self.when_to_clean = when_to_clean
        self.current_src_states_allowed = current_src_states_allowed
        self.which_src_steps_to_clean = which_src_steps_to_clean

    @classmethod
    def clean_all(cls):
        return cls(when_to_clean='EACH_STEP', current_src_states_allowed=cls.CURRENT_SRC_STATES_ALLOWED,
                   which_src_steps_to_clean='all')

    @classmethod
    def clean_all_except_last(cls):
        return cls(when_to_clean='EACH_STEP', current_src_states_allowed=cls.CURRENT_SRC_STATES_ALLOWED,
                   which_src_steps_to_clean='all_before_this_one')

    @property
    def when_to_clean(self):
        return self._when_to_clean

    @when_to_clean.setter
    def when_to_clean(self, when_to_clean):
        if when_to_clean not in self.WHEN_TO_CLEAN:
            raise ValueError('Argument "when_to_clean" is "{}" while it should be one of the following : '
                             '{}'.format(when_to_clean,
                                         ', '.join(self.WHEN_TO_CLEAN)))
        self._when_to_clean = when_to_clean

    @property
    def current_src_states_allowed(self):
        return self._current_src_states_allowed

    @current_src_states_allowed.setter
    def current_src_states_allowed(self, current_src_states_allowed):
        for current_src_state_allowed in current_src_states_allowed:
            if current_src_state_allowed not in self.CURRENT_SRC_STATES_ALLOWED:
                raise ValueError('One of the items in  "current_src_states_allowed" is "{}" while it should be one of '
                                 'the following : {}'.format(current_src_state_allowed,
                                                             ', '.join(self.CURRENT_SRC_STATES_ALLOWED)))
        self._current_src_states_allowed = current_src_states_allowed

    @property
    def which_src_steps_to_clean(self):
        return self._which_src_steps_to_clean

    @which_src_steps_to_clean.setter
    def which_src_steps_to_clean(self, which_src_steps_to_clean):
        if which_src_steps_to_clean in ['all', 'this_one', 'all_before_this_one', 'all_before_the_previous_one',
                                        'the_one_before_this_one', 'the_one_before_the_previous_one']:
            self._which_src_steps_to_clean = which_src_steps_to_clean
            self._which_src_steps_to_clean_pattern = which_src_steps_to_clean
        elif which_src_steps_to_clean[:7] == 'single_':
            sp = which_src_steps_to_clean.split('_')
            if len(sp) != 2:
                raise ValueError('Argument "which_src_steps_to_clean" is "{}". It starts with "single_" but has more '
                                 'than one underscore ... '
                                 'Impossible to identify the step to clean.'.format(which_src_steps_to_clean))
            try:
                istep = int(sp[1])
            except ValueError:
                raise ValueError('Argument "which_src_steps_to_clean" is "{}". It starts with "single_" but the '
                                 'remaining part is not an integer ... '
                                 'Impossible to identify the step to clean.'.format(which_src_steps_to_clean))
            if istep < 1:
                raise ValueError('Argument "which_src_steps_to_clean" is "{}". It starts with "single_" but the '
                                 'remaining part is an integer < 1 ... '
                                 'Impossible to identify the step to clean.'.format(which_src_steps_to_clean))
            self._which_src_steps_to_clean = which_src_steps_to_clean
            self._which_src_steps_to_clean_pattern = 'single_N'
        elif (len(which_src_steps_to_clean) > 29 and which_src_steps_to_clean[:15] == 'all_before_the_' and
                  which_src_steps_to_clean[-14:] == '_previous_ones'):
            sp = which_src_steps_to_clean.split('_')
            if len(sp) != 6:
                raise ValueError('Argument "which_src_steps_to_clean" is "{}". It starts with "all_before_the_", '
                                 'ends with "_previous_ones" but has more than 5 underscores ... Impossible to '
                                 'identify the steps to clean.'.format(which_src_steps_to_clean))
            try:
                istep = int(sp[3])
            except ValueError:
                raise ValueError('Argument "which_src_steps_to_clean" is "{}". It starts with "all_before_the_", '
                                 'ends with "_previous_ones" but the remaining part is not an integer ... '
                                 'Impossible to identify the steps to clean.'.format(which_src_steps_to_clean))
            if istep < 2:
                raise ValueError('Argument "which_src_steps_to_clean" is "{}". It starts with "all_before_the_", '
                                 'ends with "_previous_ones" but the remaining part an integer less than 2 ... '
                                 'Impossible to identify the steps to clean.'.format(which_src_steps_to_clean))
            self._which_src_steps_to_clean = which_src_steps_to_clean
            self._which_src_steps_to_clean_pattern = 'all_before_the_N_previous_ones'
        elif (len(which_src_steps_to_clean) > 33 and which_src_steps_to_clean[:19] == 'the_one_before_the_' and
                  which_src_steps_to_clean[-14:] == '_previous_ones'):
            sp = which_src_steps_to_clean.split('_')
            if len(sp) != 7:
                raise ValueError('Argument "which_src_steps_to_clean" is "{}". It starts with "the_one_before_the_", '
                                 'ends with "_previous_ones" but has more than 6 underscores ... Impossible to '
                                 'identify the steps to clean.'.format(which_src_steps_to_clean))
            try:
                istep = int(sp[4])
            except ValueError:
                raise ValueError('Argument "which_src_steps_to_clean" is "{}". It starts with "the_one_before_the_", '
                                 'ends with "_previous_ones" but the remaining part is not an integer ... '
                                 'Impossible to identify the steps to clean.'.format(which_src_steps_to_clean))
            if istep < 2:
                raise ValueError('Argument "which_src_steps_to_clean" is "{}". It starts with "the_one_before_the_", '
                                 'ends with "_previous_ones" but the remaining part an integer less than 2 ... '
                                 'Impossible to identify the steps to clean.'.format(which_src_steps_to_clean))
            self._which_src_steps_to_clean = which_src_steps_to_clean
            self._which_src_steps_to_clean_pattern = 'the_one_before_the_N_previous_ones'
        #TODO: implement "the_M_before_the_N_previous_ones" if needed ...
        else:
            raise ValueError('Argument "which_src_steps_to_clean" is "{}". This is not allowed. See documentation for '
                             'the allowed options.'.format(which_src_steps_to_clean))

    def steps_to_clean(self, this_step_index, this_step_state):
        if this_step_state not in self.current_src_states_allowed:
            return []
        if self._which_src_steps_to_clean_pattern == 'all':
            return list(range(1, this_step_index+1))
        elif self._which_src_steps_to_clean_pattern == 'this_one':
            return [this_step_index]
        elif self._which_src_steps_to_clean_pattern == 'the_one_before_this_one':
            if this_step_index == 1:
                return []
            return [this_step_index-1]
        elif self._which_src_steps_to_clean_pattern == 'the_one_before_the_previous_one':
            if this_step_index <= 2:
                return []
            return [this_step_index-2]
        elif self._which_src_steps_to_clean_pattern == 'the_one_before_the_N_previous_ones':
            iprev = int(self.which_src_steps_to_clean.split('_')[4])
            istep = this_step_index-iprev-1
            if istep < 1:
                return []
            return [istep]
        elif self._which_src_steps_to_clean_pattern == 'all_before_this_one':
            return list(range(1, this_step_index))
        elif self._which_src_steps_to_clean_pattern == 'all_before_the_previous_one':
            return list(range(1, this_step_index-1))
        elif self._which_src_steps_to_clean_pattern == 'all_before_the_N_previous_ones':
            iprev = int(self.which_src_steps_to_clean.split('_')[3])
            return list(range(1, this_step_index-iprev))
        elif self._which_src_steps_to_clean_pattern == 'single_N':
            istep = int(self.which_src_steps_to_clean.split('_')[1])
            if istep > this_step_index:
                return []
            return [istep]
        raise ValueError('Should not reach this point in "steps_to_clean" of "SRCCleanerOptions"')

    def as_dict(self):
        return {'@class': self.__class__.__name__,
                '@module': self.__class__.__module__,
                'when_to_clean': self.when_to_clean,
                'current_src_states_allowed': self.current_src_states_allowed,
                'which_src_steps_to_clean': self.which_src_steps_to_clean}

    @classmethod
    def from_dict(cls, d):
        return cls(when_to_clean=d['when_to_clean'],
                   current_src_states_allowed=d['current_src_states_allowed'],
                   which_src_steps_to_clean=d['which_src_steps_to_clean'])


class SRCCleaner(MSONable):

    # RECURRENCE_TYPES = ['END', 'STEP', 'STEP_EXCEPT_END']
    # STATE_TYPES = {'END': ['FINALIZED', 'UNRECOVERABLE', 'MAXRESTARTS'],
    #                'STEP': ['RECOVERABLE', 'UNRECOVERABLE', 'MAXRESTARTS', 'FINALIZED'],
    #                'STEP_EXCEPT_END': ['RECOVERABLE']}
    # CLEANING_TYPES = ['ALL', 'ALL_EXCEPT_LAST', 'LAST']
    SRC_TYPES = ['setup', 'run', 'control', 'src_root']

    def __init__(self, cleaners=None, src_type='run', cleaner_options=SRCCleanerOptions.clean_all()):
        if cleaners is None:
            self.cleaners = []
        else:
            self.cleaners = cleaners
        self.src_type = src_type
        self.cleaner_options = cleaner_options

    @property
    def cleaners(self):
        return self._cleaners

    @cleaners.setter
    def cleaners(self, cleaners):
        if cleaners is None:
            self._cleaners = []
        elif isinstance(cleaners, list):
            for cl in cleaners:
                if not isinstance(cl, Cleaner):
                    raise ValueError('One of the items in cleaners is not a Cleaner instance but is an instance '
                                     'of {}'.format(cl.__class__.__name__))
            self._cleaners = cleaners
        else:
            raise ValueError('The variable "cleaners" should be either None or a list of Cleaner objects')

    @property
    def src_type(self):
        return self._src_type

    @src_type.setter
    def src_type(self, src_type):
        if src_type not in self.SRC_TYPES:
            raise ValueError('Argument "src_type" should be one of the following : '
                             '{}'.format(', '.format(self.SRC_TYPES)))
        self._src_type = src_type

    def src_dir_to_clean(self, src_directories):
        return src_directories['{}_dir'.format(self.src_type)]

    def check_recurrence(self, src_task_index, state):
        if state == self.recurrence:
            return True
        return False

    def clean(self, last_src_directories, previous_src_dirs, src_task_index, state):
        dirs_to_clean = []
        if self.cleaning in ['ALL', 'LAST']:
            dirs_to_clean.append(self.src_dir_to_clean(last_src_directories))
            pass
        if self.cleaning in ['ALL', 'ALL_EXCEPT_LAST']:
            for src_directories in previous_src_dirs:
                dirs_to_clean.append(self.src_dir_to_clean(src_directories))
        for dir_to_clean in dirs_to_clean:
            for cleaner in self.cleaners:
                cleaner.clean(root_directory=dir_to_clean)

    def as_dict(self):
        return {'@class': self.__class__.__name__,
                '@module': self.__class__.__module__,
                'cleaners': [c.as_dict() for c in self.cleaners],
                'src_types': self.src_type,
                'cleaner_options': self.cleaner_options}

    @classmethod
    def from_dict(cls, d):
        return cls(recurrence=d['recurrence'],
                   cleaners=[Cleaner.from_dict(d_c) for d_c in d['cleaners']],
                   src_type=d['src_type'])


class SRCCleaning(MSONable):

    def __init__(self, src_cleaners=None):
        if src_cleaners is None:
            self.src_cleaners = []
        else:
            self.src_cleaners = src_cleaners

    def clean(self, src_directories, previous_src_dirs, state):
        pass

    def as_dict(self):
        return {'@class': self.__class__.__name__,
                '@module': self.__class__.__module__,
                'src_cleaners': [src_c.as_dict() for src_c in self.src_cleaners]}

    @classmethod
    def from_dict(cls, d):
        return cls(src_cleaners=[SRCCleaner.from_dict(d_src_c) for d_src_c in d['src_cleaners']])


def createSRCFireworks(setup_task, run_task, control_task, spec=None, initialization_info=None,
                       task_index=None, setup_spec_update=None, run_spec_update=None):
    # Make a full copy of the spec
    if spec is None:
        spec = {}
    if initialization_info is None:
        initialization_info = {}
    spec = copy.deepcopy(spec)
    spec['_add_launchpad_and_fw_id'] = True
    spec['_add_fworker'] = True
    # Initialize the SRC task_index
    if task_index is not None:
        src_task_index = SRCTaskIndex.from_any(task_index)
    else:
        # src_task_index = SRCTaskIndex.from_any('unknown-task')
        src_task_index = SRCTaskIndex.from_task(run_task)
    spec['SRC_task_index'] = src_task_index

    # SetupTask
    setup_spec = copy.deepcopy(spec)
    # Remove any initial queue_adapter_update from the spec
    setup_spec.pop('queue_adapter_update', None)

    setup_spec = set_short_single_core_to_spec(setup_spec)
    setup_spec['_preserve_fworker'] = True
    setup_spec['_pass_job_info'] = True
    setup_spec['initialization_info'] = initialization_info
    setup_spec.update({} if setup_spec_update is None else setup_spec_update)
    setup_fw = Firework(setup_task, spec=setup_spec, name=src_task_index.setup_str)

    # RunTask
    run_spec = copy.deepcopy(spec)
    run_spec['SRC_task_index'] = src_task_index
    run_spec['_preserve_fworker'] = True
    run_spec['_pass_job_info'] = True
    run_spec.update({} if run_spec_update is None else run_spec_update)
    run_fw = Firework(run_task, spec=run_spec, name=src_task_index.run_str)

    # ControlTask
    control_spec = copy.deepcopy(spec)
    control_spec = set_short_single_core_to_spec(control_spec)
    control_spec['SRC_task_index'] = src_task_index
    control_spec['_allow_fizzled_parents'] = True
    control_fw = Firework(control_task, spec=control_spec, name=src_task_index.control_str)

    links_dict = {setup_fw.fw_id: [run_fw.fw_id],
                  run_fw.fw_id: [control_fw.fw_id]}
    return {'setup_fw': setup_fw, 'run_fw': run_fw, 'control_fw': control_fw, 'links_dict': links_dict,
            'fws': [setup_fw, run_fw, control_fw]}


class SRCTaskIndex(MSONable):

    ALLOWED_CHARS = ['-']

    def __init__(self, task_type, index=1):
        self.set_task_type(task_type=task_type)
        self.index = index

    def set_task_type(self, task_type):
        prefix_test_string = str(task_type)
        for allowed_char in self.ALLOWED_CHARS:
            prefix_test_string = prefix_test_string.replace(allowed_char, "")
        if not prefix_test_string.isalnum():
            ac_str = ', '.join(['"{}"'.format(ac) for ac in self.ALLOWED_CHARS])
            raise ValueError('task_type should only contain letters '
                             'and the following characters : {}'.format(ac_str))
        self.task_type = task_type

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        if isinstance(index, int):
            self._index = index
        elif isinstance(index, str):
            try:
                myindex = int(index)
                self._index = myindex
            except:
                raise ValueError('Index in SRCTaskIndex should be an integer or a string '
                                 'that can be cast into an integer')
        else:
            raise ValueError('Index in SRCTaskIndex should be an integer or a string '
                             'that can be cast into an integer')

    def increase_index(self):
        self.index += 1

    def __add__(self, other):
        if not isinstance(other, int):
            raise ValueError('The __add__ method in SRCTaskIndex should be an integer')
        self.index += other

    def __str__(self):
        return '_'.join([self.task_type, str(self.index)])

    @property
    def setup_str(self):
        return '_'.join(['setup', self.__str__()])

    @property
    def run_str(self):
        return '_'.join(['run', self.__str__()])

    @property
    def control_str(self):
        return '_'.join(['control', self.__str__()])

    @classmethod
    def from_string(cls, SRC_task_index_string):
        sp = SRC_task_index_string.split('_')
        # if len(sp) not in [2, 3]:
        #     raise ValueError('SRC_task_index_string should contain 1 or 2 underscores ("_") '
        #                      'while it contains {:d}'.format(len(sp)-1))
        if len(sp) == 1:
            return cls(task_type=sp[0])
        if any([len(part) == 0 for part in sp]):
            raise ValueError('SRC_task_index_string has an empty part when separated by underscores ...')
        if len(sp) == 2:
            return cls(task_type=sp[0], index=sp[1])
        elif len(sp) == 3:
            if sp[0] not in ['setup', 'run', 'control']:
                raise ValueError('SRC_task_index_string should start with "setup", "run" or "control" when 3 parts are '
                                 'identified')
            return cls(task_type=sp[1], index=sp[2])

    @classmethod
    def from_any(cls, SRC_task_index):
        if isinstance(SRC_task_index, six.string_types):
            return cls.from_string(SRC_task_index)
        elif isinstance(SRC_task_index, SRCTaskIndex):
            return cls(task_type=SRC_task_index.task_type, index=SRC_task_index.index)
        else:
            raise ValueError('SRC_task_index should be an instance of "str" or "SRCTaskIndex" '
                             'in "from_any" class method')

    @classmethod
    def from_task(cls, task):
        return cls(task_type=task.task_type)

    @classmethod
    def from_dict(cls, d):
        return cls(task_type=d['task_type'], index=d['index'])

    def as_dict(self):
        return {'@class': self.__class__.__name__,
                '@module': self.__class__.__module__,
                'task_type': self.task_type,
                'index': self.index}



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


################
# Exceptions
################

class SRCError(Exception):
    pass


class SetupError(SRCError):
    pass


class RunError(SRCError):
    pass


class ControlError(SRCError):
    pass


################
# Timings
################


class FWTime(MSONable):
    def __init__(self, fw_name, fw_id, ncpus, fwtime_secs, clustertime_secs=None):
        self.fw_name = fw_name
        self.fw_id = fw_id
        self.ncpus = ncpus
        self.fwtime_secs = fwtime_secs
        self.clustertime_secs = clustertime_secs

    @property
    def time_per_cpu(self):
        if self.clustertime_secs is not None:
            return self.clustertime_secs
        return self.fwtime_secs

    @property
    def total_time(self):
        return self.ncpus*self.time_per_cpu

    def as_dict(self):
        dd = dict(fw_name=self.fw_name, fw_id=self.fw_id,
                  ncpus=self.ncpus, fwtime_secs=self.fwtime_secs,
                  clustertime_secs=self.clustertime_secs)
        return dd

    @classmethod
    def from_dict(cls, d):
        return cls(fw_name=d['fw_name'], fw_id=d['fw_id'],
                   ncpus=d['ncpus'], fwtime_secs=d['fwtime_secs'], clustertime_secs=d['clustertime_secs'])

    @classmethod
    def from_fw_id(cls, fw_id, lpad=None):
        if lpad is None:
            lpad = LaunchPad.auto_load()
        fw_dict = lpad.get_fw_dict_by_id(fw_id=fw_id)
        name = fw_dict['name']
        # TODO: find a way to know the number of cpus here ? Or should we always assume it is 1 ?
        ncpus = 1
        fwtime_secs = 0.0
        # TODO: get the runtime from the cluster (taking the reservation_id and things like that ?)
        clustertime_secs = None
        return cls(fw_name=name, fw_id=fw_id,
                   ncpus=ncpus, fwtime_secs=fwtime_secs, clustertime_secs=clustertime_secs)

class SRCFWTime(FWTime):
    def __init__(self, fw_name, fw_id, ncpus, fwtime_secs, clustertime_secs=None,
                 src_type=None, task_type=None, task_index=None):
        super(SRCFWTime, self).__init__(fw_name=fw_name, fw_id=fw_id, ncpus=ncpus,
                                        fwtime_secs=fwtime_secs, clustertime_secs=clustertime_secs)
        self.src_type = src_type
        self.task_type = task_type
        self.task_index = task_index

    def as_dict(self):
        dd = dict(src_type=self.src_type, task_type=self.task_type, task_index=self.task_index.as_dict(),
                  ncpus=self.ncpus, fwtime_secs=self.fwtime_secs,
                  clustertime_secs=self.clustertime_secs)
        return dd

    @classmethod
    def from_dict(cls, d):
        return cls(src_type=d['src_type'], task_type=d['task_type'], task_index=SRCTaskIndex.from_any(d['task_index']),
                   ncpus=d['ncpus'], fwtime_secs=d['fwtime_secs'], clustertime_secs=d['clustertime_secs'])

    @classmethod
    def from_fw_id(cls, fw_id, lpad=None):
        if lpad is None:
            lpad = LaunchPad.auto_load()
        fw_dict = lpad.get_fw_dict_by_id(fw_id=fw_id)
        name = fw_dict['name']
        # TODO: find a way to know the number of cpus here ? Or should we always assume it is 1 ?
        ncpus = 1
        fwtime_secs = 0.0