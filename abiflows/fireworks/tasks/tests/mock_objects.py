from __future__ import print_function, division, unicode_literals, absolute_import

import os

from pymatgen.io.abinit.events import EventReport, ScfConvergenceWarning, RelaxConvergenceWarning, AbinitError

from fireworks import Firework, FireTaskBase, FWAction, explicit_serialize
from abiflows.fireworks.workflows.abinit_workflows import AbstractFWWorkflow
from mongoengine import Document, StringField, IntField

##########################
# Test reports
##########################

def report_ok():
    return EventReport('.')

def report_ScfConvergenceWarning():
    er = EventReport('.', events=[ScfConvergenceWarning(message='Fake warning', src_file=__file__, src_line=0)])
    er.set_run_completed(True,'', ' ')
    return er

def report_RelaxConvergenceWarning():
    return EventReport('.', events=[RelaxConvergenceWarning(message='Fake warning', src_file=__file__, src_line=0)])

def report_AbinitError():
    return EventReport('.', events=[AbinitError(message='Fake warning', src_file=__file__, src_line=0)])

##########################
# Fake Tasks
##########################

@explicit_serialize
class FakeTask(FireTaskBase):
    def run_task(self, fw_spec):
        return FWAction()

fake_fw = Firework([FakeTask()])


@explicit_serialize
class CreateOutputsTask(FireTaskBase):
    """
    Creates temporary files in with the specified extensions in the "indata", "outdata" and "tmpdata" folders.
    """

    prefix = "tmp_"

    def run_task(self, fw_spec):
        dirs = ["indata", "outdata", "tmpdata"]
        for d in dirs:
            os.mkdir(d)
            for e in self.get('extensions', []):
                with open(os.path.join(d, "{}{}".format(self.prefix, e)), "wt") as f:
                    f.write(" ")
        return FWAction()


##########################
# Fake Workflows
##########################

class DataDocument(Document):

    test_field_string = StringField()
    test_field_int = IntField()

class SaveDataWorkflow(AbstractFWWorkflow):

    workflow_class = 'SaveDataWorkflow'
    workflow_module = 'abiflows.fireworks.tasks.tests.mock_objects'

    @classmethod
    def get_mongoengine_results(cls, wf):
        return DataDocument(test_field_string="test_text", test_field_int=5)

