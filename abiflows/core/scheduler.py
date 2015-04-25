# coding: utf-8
"""The scheduler is the object responsible for the submission of the flows in the database."""
from __future__ import print_function, division, unicode_literals

import os
import six
import time
import shutil
import json

from datetime import datetime
from tempfile import mkstemp
from monty.collections import AttrDict
from monty.io import FileLock
from mongoengine import *
from mongoengine import connect
from abipy import abilab
from .models import MongoFlow


class FlowUploader(object):

    db_name = "abiflows"

    def __init__(self, **kwargs):
        connect(db=self.db_name)
        self.node_ids = set()

    def upload(self, flow, priority="normal", flow_info=None):
        if flow.node_id in self.node_ids:
            raise ValueError("Cannot upload the same flow twice")
        self.node_ids.update([flow.node_id])

        # allocate the Flow here if workdir has not been defined.
        if not hasattr(flow, "workdir"):
            workdir = "flow_id" + str(flow.node_id)
            assert not os.path.exists(workdir)
            flow.allocate(workdir=workdir)

        entry = FlowEntry.from_flow(flow, priority=priority, flow_info=flow_info)
        entry.save(validate=True)

        flow.build_and_pickle_dump()


class FlowEntry(Document):

    node_id = LongField(required=True)
    workdir = StringField(required=True)
    status = StringField(required=True)
    priority = StringField(required=True, choices=("low", "normal", "high"))
    created = DateTimeField(required=True)
    num_scheduled = LongField(required=True)
    last_schedule = DateTimeField(required=True)
    info = DictField()

    bkp_pickle = FileField(required=True)

    meta = {
        "collection": "queued_flows",
        "indexes": ["status", "priority", "created"],
    }

    @classmethod
    def from_flow(cls, flow, priority="normal", flow_info=None):
        info = flow.get_mongo_info()
        if flow_info is not None:
            info.update(flow_info)
        
        new = cls(
            node_id=flow.node_id,
            workdir=flow.workdir,
            status=str(flow.status),
            priority=priority,
            created=datetime.now(),
            num_scheduled=0,
            last_schedule=datetime.now(),
            info=info,
        )

        new.bkp_pickle.put(flow.pickle_dumps())
        return new

    def pickle_load(self):
        return abilab.Flow.pickle_load(self.workdir)



class LogRecord(Document):
    """Capped collection used for logging"""
    meta = {'max_documents': 1000, 'max_size': 2000000}

    level = StringField(required=True)
    created = DateTimeField(required=True)
    msg = StringField(required=True)


class MongoLogger(object):
    """Logger-like object that saves log messages in a MongoDb Capped collection."""

    def reset(self):
        LogRecord.drop_collection()

    def info(self, msg, *args):
        """Log 'msg % args' with the info severity level"""
        self._log("INFO", msg, args)
                                                                
    def warning(self, msg, *args):
        """Log 'msg % args' with the warning severity level"""
        self._log("WARNING", msg, args)
                                                                
    def critical(self, msg, *args):
        """Log 'msg % args' with the critical severity level"""
        self._log("CRITICAL", msg, args)
                                                                 
    def _log(self, level, msg, args):
        if args:
            try:
                msg = msg % args
            except:
                msg += str(self.args)

        log = LogRecord(level=level, msg=msg, created=datetime.now())
        log.save(validate=False)


class MongoFlowScheduler(object):

    YAML_FILE = "mongo_scheduler.yml"
    USER_CONFIG_DIR = os.path.join(os.getenv("HOME"), ".abinit", "abipy")

    db_name = "abiflows"
    pid_path = os.path.join(os.getenv("HOME"), ".abinit", "abipy", "mongo_scheduler.pid")

    def __init__(self, **kwargs):
        """
        Args:
            workdir:
            max_njobs_inqueue: The launcher will stop submitting jobs when the
                    number of jobs in the queue is >= Max number of jobs
            max_cores: Maximum number of cores.
            sleep_time
            rm_completed_flows
            rm_errored_flows
            fix_qcritical
            validate:
            logmode:
        """
        #TODO: host, port
        from mongoengine import connect
        connect(self.db_name)

        workdir = "/tmp"
        self.workdir = os.path.abspath(workdir)
        self.max_njobs_inqueue = kwargs.pop("max_njobs_inqueue", 200)
        self.max_cores = kwargs.pop("max_cores", 1000)
        self.sleep_time = kwargs.pop("sleep_time", 5)

        self.rm_completed_flows = bool(kwargs.pop("rm_completed_flows", True))
        self.rm_errored_flows = bool(kwargs.pop("rm_errored_flows", True))
        self.fix_qcritical = bool(kwargs.pop("fix_qcritical", True))
        self.validate = bool(kwargs.pop("validate", True))

        if kwargs.pop("logmode", "mongodb") == "mongodb":
            self.logger = MongoLogger()
            self.logger.reset()
        else:
            import logging
            self.logger = logging.getLogger(__name__)

        if kwargs:
            raise ValueError("Unknown options:\n%s" % list(kwargs.keys()))

        self.check_and_write_pid_file()
        self.logger.info("hello")

    @classmethod
    def from_user_config(cls):
        """
        Initialize the :class:`MongoFlowScheduler` from the YAML file 'mongo_launcher.yaml'.
        Search first in the working directory and then in the abipy configuration directory.

        Raises:
            RuntimeError if file is not found.
        """
        # Try in the current directory then in user configuration directory.
        path = os.path.join(os.getcwd(), cls.YAML_FILE)
        if not os.path.exists(path):
            path = os.path.join(cls.USER_CONFIG_DIR, cls.YAML_FILE)

        if not os.path.exists(path):
            raise RuntimeError("Cannot locate %s neither in current directory nor in %s" % (cls.YAML_FILE, path))

        return cls.from_file(path)

    @classmethod
    def from_file(cls, filepath):
        try:
            import yaml
            with open(filepath, "rt") as fh:
               return cls(**yaml.load(fh))

        except Exception as exc:
            raise ValueError("Error while reading MongoFlowLauncher parameters from file %s\nException: %s" % 
                (filepath, exc))

    def check_and_write_pid_file(self):
        """
        This function checks if we already have a running instance of :class:`MongoFlowScheduler`.
        Raises: RuntimeError if the pid file of the scheduler exists.
        """
        if os.path.exists(self.pid_path): 
            raise RuntimeError("""\n\
                pid_path
                %s
                already exists. There are two possibilities:

                   1) There's an another instance of MongoFlowScheduler running
                   2) The previous scheduler didn't exit in a clean way

                To solve case 1:
                   Kill the previous scheduler (use 'kill pid' where pid is the number reported in the file)
                   Then you can restart the new scheduler.

                To solve case 2:
                   Remove the pid_path and restart the scheduler.

                Exiting""" % self.pid_path)

        # Make dir and file if not present.
        if not os.path.exists(os.path.dirname(self.pid_path)):
            os.makedirs(os.path.dirname(self.pid_path))

        d = dict(
            pid=os.getpid(),
            #host=self.host,
            #port=self.port,
            #"db_name"=self.db_name
        )

        with FileLock(self.pid_path):
            with open(self.pid_path, "wt") as fh:
                json.dump(d, fh)

    #def __str__(self):
    #def drop_database()

    def shutdown(self):
        try:
            os.remove(self.pid_path)
        except IOError:
            pass
        import sys
        sys.exit(1)

    def sleep(self):
        time.sleep(self.sleep_time)

    def update_entry(self, entry, flow=None):
        if flow is None:
            flow = entry.pickle_load()
                                                       
        flow.check_status()
        entry.status = str(flow.status)
                                                       
        if flow.status == flow.S_OK:
            return self.move_to_completed(entry, flow)
                                                       
        entry.save(validate=self.validate)

    def update_entries(self):
        for entry in FlowEntry.objects:
            self.update_entry(entry)

    def select(self):
        return [(e, e.pickle_load()) for e in FlowEntry.objects]

    #def find_qcriticals(self):
    #    return [(e, e.pickle_load()) for e in FlowEntry.objects(status="QCritical")]

    #def find_abicriticals(self):
    #    return [(e, e.pickle_load()) for e in FlowEntry.objects(status="AbiCritical")]

    #def restart_unconverged(self):

    def fix_queue_critical(self):
        for entry in FlowEntry.objects(status="QCritical"):
            flow = entry.pickle_load()
            if self.fix_qcritical:
                flow.fix_queue_critical()
                self.update_entry(entry, flow=flow)
            else:
                self.move_to_errored(entry, flow)

    def fix_abicritical(self):
        for entry in FlowEntry.objects(status="AbiCritical"):
            flow = entry.pickle_load()
            flow.fix_abicritical()
            self.update_entry(entry, flow=flow)

    def move_to_completed(self, entry, flow):
        """Move this entry to the completed_flows collection."""
        entry.delete()
        entry.bkp_pickle.delete()

        entry.switch_collection("completed_flows")
        entry.save(validate=self.validate)

        # TODO: Handle possible errors 
        doc = MongoFlow.from_flow(flow)
        doc.save(validate=self.validate)

        if self.rm_completed_flows:
            try:
                flow.rmtree()
            except IOError:
                pass

    def move_to_errored(self, entry, flow):
        """Move this entry to the errored_flows collection"""
        # TODO: options to handle the storage of errored flows.
        entry.delete()
        entry.switch_collection("errored_flows")
        entry.save(validate=self.validate)

        doc = MongoFlow.from_flow(flow)
        doc.switch_collection("errored_flows")
        doc.save(validate=self.validate)

        if self.rm_errored_flows:
            try:
                flow.rmtree()
            except IOError:
                pass

    def rapidfire(self):
        # TODO: restart
        for entry, flow in self.select():
            entry.last_scheduled = datetime.now()
            entry.num_scheduled += 1
            try:
                flow.rapidfire()
            except Exception as exc:
                # Move this entry to the errored_flows collection.
                self.move_to_errored(entry, flow)

            flow.check_status()
            entry.status = str(flow.status)
                                                                   
            if flow.status == flow.S_OK:
                self.move_to_completed(entry, flow)
            else:
                flow.pickle_dump()
                entry.save(validate=self.validate)

    def rollback_entry(self, entry):
        if os.path.exists(entry.workdir):
            shutil.rmtree(entry.workdir, ignore_errors=False, onerror=None)
        
        _, filepath = mkstemp(suffix='.pickle', text=False)
        with open(filepath , "wb") as fh:
            fh.write(entry.bkp_pickle.read())
                                                                                
        flow = abilab.Flow.pickle_load(filepath)
                                                                                
        new_entry = FlowEntry.from_flow(flow, priority=entry.priority, flow_info=entry.info)
        entry.delete()

        new_entry.save(validate=self.validate)
        return new_entry

    def run(self):
        if FlowEntry.objects.count() == 0:
            self.logger.info("No FlowEntries, will sleep for %s" % self.sleep_time)
            #self.sleep()

        self.update_entries()
        self.fix_queue_critical()
        self.fix_abicritical()

        self.rapidfire()

        return FlowEntry.objects.count()

    def start(self):
        while True:
            if self.run() == 0:
                return print("All flows completed")
