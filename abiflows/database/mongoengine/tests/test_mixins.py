from __future__ import print_function, division, unicode_literals, absolute_import

import os
import unittest
import filecmp
import tempfile
import abipy.data as abidata
import abipy.abilab as abilab

from datetime import datetime
from abipy.abio.factories import scf_input
from abiflows.core.testing import AbiflowsTest, has_mongodb, TESTDB_NAME
from abiflows.database.mongoengine.mixins import *
from mongoengine import Document
from mongoengine.errors import ValidationError
from fireworks import Workflow, Firework
from fireworks.core.rocket_launcher import rapidfire
from fireworks.user_objects.firetasks.script_task import PyTask


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


class TestMixins(AbiflowsTest):
    """
    For all the classes, try to create an instance and call methods to fill the document.
    If mongodb is available, save the document.
    """

    @classmethod
    def setUpClass(cls):
        cls.si_structure = abilab.Structure.from_file(abidata.cif_file("si.cif"))
        cls.scf_inp = scf_input(cls.si_structure, abidata.pseudos("14si.pspnc"), ecut=2, kppa=10)

    def setUp(self):
        self.setup_mongoengine()
        self.setup_fireworks()

    @classmethod
    def tearDownClass(cls):
        cls.teardown_fireworks(module_dir=MODULE_DIR)
        cls.teardown_mongoengine()

    def test_calculation_metadata(self):
        doc_class = self.get_document_class_from_mixin(CalculationMetadataMixin)
        doc = doc_class(user="user", cluster="cluster", execution_date=datetime(2010, 1, 1, 0, 1, 1))

        if has_mongodb():
            doc.save()
            query_result = doc_class.objects()
            assert len(query_result) == 1

    def test_run_stats(self):
        doc_class = self.get_document_class_from_mixin(RunStatsMixin)
        doc = doc_class(core_num=10, elapsed_time=10.5, maximum_memory_used=1000, number_of_restarts=2,
                        number_of_errors=1, number_of_warnings=2)

        if has_mongodb():
            doc.save()
            query_result = doc_class.objects()
            assert len(query_result) == 1

    def test_spacegroup(self):
        doc_class = self.get_document_class_from_mixin(SpaceGroupMixin)
        doc = doc_class()
        doc.set_space_group_from_structure(self.si_structure)
        assert doc.number == 227

        if has_mongodb():
            doc.save()
            query_result = doc_class.objects()
            assert len(query_result) == 1

    def test_material(self):
        doc_class = self.get_document_class_from_mixin(MaterialMixin)
        doc = doc_class()
        doc.set_material_data_from_structure(self.si_structure)
        assert doc.nelements == 1

        if has_mongodb():
            doc.save()
            query_result = doc_class.objects()
            assert len(query_result) == 1

    def test_calculation_type(self):
        doc_class = self.get_document_class_from_mixin(CalculationTypeMixin)
        doc = doc_class(xc_functional="PBE", pseudo_type="nc", is_hubbard=False, pseudo_dojo_table="table")

        if has_mongodb():
            doc.save()
            query_result = doc_class.objects()
            assert len(query_result) == 1

    def test_ground_state_output(self):
        doc_class = self.get_document_class_from_mixin(GroundStateOutputMixin)
        doc = doc_class(final_energy=1.1, efermi=0.1, total_magnetization=-0.1, structure=self.si_structure.as_dict())

        if has_mongodb():
            doc.save()
            query_result = doc_class.objects()
            assert len(query_result) == 1

    def test_hubbards(self):
        doc_class = self.get_document_class_from_mixin(HubbardMixin)
        doc = doc_class(hubbards={"AA": 1.5})
        with self.assertRaises(ValidationError):
            doc.validate()

        doc = doc_class(hubbards={"Ti": 1.5})

        if has_mongodb():
            doc.save()
            query_result = doc_class.objects()
            assert len(query_result) == 1

    def test_date(self):
        doc_class = self.get_document_class_from_mixin(DateMixin)
        doc = doc_class(created_on=datetime(2010, 1, 1, 0, 1, 1), modified_on=datetime(2011, 1, 1, 0, 1, 1))

        if has_mongodb():
            doc.save()
            query_result = doc_class.objects()
            assert len(query_result) == 1

    def test_directory(self):
        doc_class = self.get_document_class_from_mixin(DirectoryMixin)
        doc = doc_class()

        # create and run a fireworks workflow
        task = PyTask(func="time.sleep", args=[0.5])
        wf = Workflow([Firework(task, spec={'wf_task_index': "1"}, fw_id=1),
                       Firework(task, spec={'wf_task_index': "2"}, fw_id=2)])
        self.lp.add_wf(wf)
        rapidfire(self.lp, self.fworker, m_dir=MODULE_DIR)
        wf = self.lp.get_wf_by_fw_id(1)

        doc.set_dir_names_from_fws_wf(wf)
        assert len(doc.dir_names) == 2

        if has_mongodb():
            doc.save()
            query_result = doc_class.objects()
            assert len(query_result) == 1
            assert len(query_result[0].dir_names) == 2
