# coding: utf-8
"""The scheduler is the object responsible for the submission of the flows in the database."""
from __future__ import print_function, division, unicode_literals

import os
import six
import time
import shutil

from datetime import datetime
from tempfile import mkstemp
from mongoengine import *
from mongoengine import connect
from abipy import abilab
from .models import MongoFlow

import logging
logger = logging.getLogger(__name__)


class FlowUploader(object):

    def __init__(self, **kwargs):
        connect(db="abiflows")
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

        entry = FlowEntry.from_flow(flow, priority=priority, info=flow_info)
        entry.save()

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
        #info = flow.get_mongo_info()
        info = {}
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

    def rollback(self):
        if os.path.exists(self.workdir)
            shutil.rmtree(self.workdir, ignore_errors=False, onerror=None)
        
        _, filepath = mkstemp(suffix='.pickle', text=False)
        with open(filepath , "wb") as fh:
            fh.write(self.bkp_pickle.read())

        flow = abilab.Flow.pickle_load(filepath)

        new = FlowEntry.from_flow(flow, priority=self.priority, info=self.info)
        self.delete()
        #self = new
        return new

    def update(self, flow=None):
        if flow is None:
            flow = self.pickle_load()

        flow.check_status()
        self.status = str(flow.status)

        if flow.status == flow.S_OK:
            return self.move_to_completed(flow)

        self.save()

    def move_to_completed(self, flow):
        self.delete()
        self.bkp_pickle.delete()

        self.switch_collection("completed_flows")
        self.save()

        # Move this entry to the ok_flows collection.
        # TODO: Handle possible errors 
        doc = MongoFlow.from_flow(flow)
        doc.save()
        flow.rmtree()

    def move_to_errored(self, flow):
        # TODO: options to handle the storage of errored flows.
        self.delete()
        self.switch_collection("errored_flows")
        self.save()

        # Move this entry to the ok_flows collection.
        #doc = MongoFlow.from_flow(flow)
        #doc.switch_collection("errored_flows")
        #doc.save()
        #flow.rmtree()

    def rapidfire(self, flow):
        try:
            flow.rapidfire()
        except:
            raise
            # Move this entry to the errored_flows collection.
            #self.move_to_errored(flow)

        self.last_scheduled = datetime.now()
        self.num_scheduled += 1
        flow.check_status()
        self.status = str(flow.status)

        if flow.status == flow.S_OK:
            return self.move_to_completed(flow)

        flow.pickle_dump()
        self.save()


class Scheduler(object):

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
            fix_qcritcal
        """
        from mongoengine import connect
        connect('abiflows')

        #self.db = db
        #self.workdir = os.path.abspath(workdir)
        self.max_njobs_inqueue = kwargs.pop("max_njobs_inqueue", 200)
        self.max_cores = kwargs.pop("max_cores", 1000)
        self.sleep_time = kwargs.pop("sleep_time", 5)

        self.rm_completed_flows = bool(kwargs.pop("rm_completed_flows", True))
        self.rm_errored_flows = bool(kwargs.pop("rm_errored_flows", True))
        self.fix_qcritical = bool(kwargs.pop("fix_qcritical", True))

        if kwargs:
            print("Unknown options:\n%s" % list(kwargs.keys()))

    #@classmethod
    #def from_file(cls, filepath):
    #    import yaml
    #    with open(filepath, "rt") as fh:
    #       return cls(**yaml.load(fh))

    def sleep(self):
        time.sleep(self.sleep_time)

    def check_all(self):
        for entry in FlowEntry.objects:
            entry.update()

    def find_qcriticals(self):
        return [(e, e.pickle_load()) for e in FlowEntry.objects(status="QCritical")]

    def find_abicriticals(self):
        return [(e, e.pickle_load()) for e in FlowEntry.objects(status="AbiCritical")]

    def select(self):
        return [(e, e.pickle_load()) for e in FlowEntry.objects]

    def serve_forever(self):
        while True:
            if self.run() == 0:
                return print("All flows completed")

    def run(self):
        if FlowEntry.objects.count() == 0:
            logger.info("No FlowEntries, will sleep for %s" % self.sleep_time)
            #self.sleep()

        self.check_all()

        for entry, flow in self.find_qcriticals():
            if self.fix_qcritical:
                flow.fix_qcritical()
                entry.update(flow=flow)
            else:
                entry.move_to_errored(flow)

        for entry, flow in self.find_abicriticals():
            flow.fix_abicritical()
            entry.update(flow=flow)

        for entry, flow in self.select():
            entry.rapidfire(flow)

        return FlowEntry.objects.count()
