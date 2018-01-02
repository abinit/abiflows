from __future__ import print_function, division, unicode_literals, absolute_import

import os
import tempfile
import unittest
import json


from monty.tempfile import ScratchDir
from fireworks import Workflow, Firework
from fireworks.core.rocket_launcher import rapidfire
from abiflows.core.testing import AbiflowsTest, has_mongodb
from abiflows.fireworks.tasks.tests import mock_objects
from abiflows.fireworks.tasks.utility_tasks import MongoEngineDBInsertionTask, FinalCleanUpTask, get_fw_task_manager
from abiflows.fireworks.utils.fw_utils import FWTaskManager
from abiflows.fireworks.tasks.tests.mock_objects import SaveDataWorkflow
from abiflows.database.mongoengine.utils import DatabaseData


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

test_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                        "test_files", "fw_task_managers")


class TestFinalCleanUpTask(AbiflowsTest):

    @classmethod
    def setUpClass(cls):
        cls.setup_fireworks()

    @classmethod
    def tearDownClass(cls):
        cls.teardown_fireworks(module_dir=MODULE_DIR)

    def tearDown(self):
        if self.lp:
            self.lp.reset(password=None,require_password=False)

    def test_class(self):
        """Test of basic methods"""

        task = FinalCleanUpTask()
        assert "WFK" in task.out_exts
        assert "1WF" in task.out_exts

        task = FinalCleanUpTask(out_exts=["DEN","WFK"])
        assert "WFK" in task.out_exts
        assert "DEN" in task.out_exts
        assert "1WF" not in task.out_exts

        task = FinalCleanUpTask(out_exts="DEN,WFK")
        assert "WFK" in task.out_exts
        assert "DEN" in task.out_exts
        assert "1WF" not in task.out_exts

        self.assertFwSerializable(task)

        with ScratchDir(".") as tmp_dir:

            wfkfile = tempfile.NamedTemporaryFile(suffix="_WFK", dir=tmp_dir)
            denfile = tempfile.NamedTemporaryFile(suffix="DEN", dir=tmp_dir)

            FinalCleanUpTask.delete_files(tmp_dir, exts=["WFK", "1WF"])

            assert not os.path.isfile(wfkfile.name)
            assert os.path.isfile(denfile.name)

    @unittest.skipUnless(has_mongodb(), "A local mongodb is required.")
    def test_run(self):
        create_fw = Firework([mock_objects.CreateOutputsTask(extensions=["WFK", "DEN"])], fw_id=1)
        delete_fw = Firework([FinalCleanUpTask(["WFK", "1WF"])], parents=create_fw, fw_id=2,
                             spec={"_add_launchpad_and_fw_id": True})

        wf = Workflow([create_fw, delete_fw])

        self.lp.add_wf(wf)

        rapidfire(self.lp, self.fworker, m_dir=MODULE_DIR, nlaunches=1)

        # check that the files have been created
        create_fw = self.lp.get_fw_by_id(1)
        create_ldir = create_fw.launches[0].launch_dir

        for d in ["tmpdata", "outdata", "indata"]:
            assert os.path.isfile(os.path.join(create_ldir, d, "tmp_WFK"))
            assert os.path.isfile(os.path.join(create_ldir, d, "tmp_DEN"))

        rapidfire(self.lp, self.fworker, m_dir=MODULE_DIR, nlaunches=1)

        wf = self.lp.get_wf_by_fw_id(1)

        assert wf.state == "COMPLETED"

        for d in ["tmpdata", "indata"]:
            assert not os.path.isfile(os.path.join(create_ldir, d, "tmp_WFK"))
            assert not os.path.isfile(os.path.join(create_ldir, d, "tmp_DEN"))

        assert not os.path.isfile(os.path.join(create_ldir, "outdata", "tmp_WFK"))
        assert os.path.isfile(os.path.join(create_ldir, "outdata", "tmp_DEN"))


class TestMongoEngineDBInsertionTask(AbiflowsTest):

    @classmethod
    def setUpClass(cls):
        cls.setup_fireworks()

    @classmethod
    def tearDownClass(cls):
        cls.teardown_fireworks(module_dir=MODULE_DIR)

    def test_class(self):
        db = DatabaseData("test_db", collection="test_collection", username="user", password="pass")
        task = MongoEngineDBInsertionTask(db)

        self.assertFwSerializable(task)

    @unittest.skipUnless(has_mongodb(), "A local mongodb is required.")
    def test_run(self):
        db = DatabaseData(self.lp.name, collection="test_MongoEngineDBInsertionTask", username=self.lp.username,
                          password=self.lp.password)
        task = MongoEngineDBInsertionTask(db)
        fw = Firework([task], fw_id=1, spec={"_add_launchpad_and_fw_id": True})
        wf = Workflow([fw], metadata={'workflow_class': SaveDataWorkflow.workflow_class,
                                     'workflow_module': SaveDataWorkflow.workflow_module})
        self.lp.add_wf(wf)

        rapidfire(self.lp, self.fworker, m_dir=MODULE_DIR, nlaunches=1)

        wf = self.lp.get_wf_by_fw_id(1)

        assert wf.state == "COMPLETED"

        # retrived the saved object
        # error if not imported locally
        from abiflows.fireworks.tasks.tests.mock_objects import DataDocument
        db.connect_mongoengine()
        with db.switch_collection(DataDocument) as DataDocument:
            data = DataDocument.objects()

            assert len(data) == 1

            assert data[0].test_field_string == "test_text"
            assert data[0].test_field_int == 5


class TestFunctions(AbiflowsTest):

    def test_get_fw_task_manager(self):
        manager_path = os.path.join(test_dir, "fw_manager_ok.yaml")

        fw_spec = {"ftm_file": manager_path, "fw_policy": {"max_restarts":44}}

        ftm = get_fw_task_manager(fw_spec)

        assert isinstance(ftm, FWTaskManager)
        assert ftm.fw_policy.max_restarts == 44
        assert ftm.has_task_manager()

