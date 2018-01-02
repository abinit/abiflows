from __future__ import print_function, division, unicode_literals, absolute_import

import os
import unittest
import filecmp
import tempfile
import abipy.data as abidata
import abipy.abilab as abilab

from abipy.abio.factories import scf_input
from abiflows.core.testing import AbiflowsTest, has_mongodb, TESTDB_NAME
from abiflows.database.mongoengine.abinit_mixins import *
from mongoengine import Document, EmbeddedDocumentField


class TestAbinitPseudoData(AbiflowsTest):
    @classmethod
    def setUpClass(cls):
        cls.si_structure = abilab.Structure.from_file(abidata.cif_file("si.cif"))
        cls.scf_inp = scf_input(cls.si_structure, abidata.pseudos("14si.pspnc"), ecut=2, kppa=10)

    def setUp(self):
        self.setup_mongoengine()

    @classmethod
    def tearDownClass(cls):
        cls.teardown_mongoengine()

    def test_class(self):
        pseudo = AbinitPseudoData()
        pseudo.set_pseudos_from_abinit_input(self.scf_inp)
        with tempfile.NamedTemporaryFile("wt") as files_file:
            files_file.write("run.abi\nrun.abo\nin\n\out\ntmp\n")
            files_file.writelines([abidata.pseudo("C.oncvpsp").filepath,
                                   abidata.pseudo("Ga.oncvpsp").filepath])
            pseudo.set_pseudos_from_files_file(files_file.name, 2)

    @unittest.skipUnless(has_mongodb(), "A local mongodb is required.")
    def test_save(self):
        class TestDocument(Document):
            meta = {'collection': "test_AbinitPseudoData"}
            pseudo = EmbeddedDocumentField(AbinitPseudoData, default=AbinitPseudoData)

        pseudo = AbinitPseudoData()
        pseudo.set_pseudos_from_abinit_input(self.scf_inp)

        doc = TestDocument()
        doc.pseudo = pseudo
        doc.save()

        query_result = TestDocument.objects()

        assert len(query_result) == 1


class TestAbinitMixins(AbiflowsTest):
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

    @classmethod
    def tearDownClass(cls):
        cls.teardown_mongoengine()

    def test_abinit_basic_input(self):

        doc_class = self.get_document_class_from_mixin(AbinitBasicInputMixin)
        doc = doc_class()
        doc.set_abinit_basic_from_abinit_input(self.scf_inp)

        if has_mongodb():
            doc.save()

            query_result = doc_class.objects()

            assert len(query_result) == 1
            saved_doc = query_result[0]
            assert saved_doc.ecut == doc.ecut
            self.assertArrayEqual(self.scf_inp['ngkpt'], saved_doc.ngkpt)
            self.assertArrayEqual(self.scf_inp['shiftk'], saved_doc.shiftk)

    def test_abinit_gs_output(self):

        doc_class = self.get_document_class_from_mixin(AbinitGSOutputMixin)
        doc = doc_class()

        if has_mongodb():
            doc.structure = self.si_structure.as_dict()
            gsr_path = abidata.ref_file('si_scf_GSR.nc')

            with open(gsr_path, "rb") as gsr:
                doc.gsr.put(gsr)

            doc.save()

            query_result = doc_class.objects()

            assert len(query_result) == 1
            saved_doc = query_result[0]
            with tempfile.NamedTemporaryFile(mode="wb") as db_file:
                db_file.write(saved_doc.gsr.read())
                db_file.seek(0)

                assert filecmp.cmp(gsr_path, db_file.name)

    def test_abinit_dftp_output(self):

        doc_class = self.get_document_class_from_mixin(AbinitDftpOutputMixin)
        doc = doc_class()

        if has_mongodb():
            doc.structure = self.si_structure.as_dict()
            ddb_path = os.path.join(abidata.dirpath, 'refs', 'znse_phonons', 'ZnSe_hex_qpt_DDB')

            # read/write in binary for py3k compatibility with mongoengine
            with open(ddb_path, "rb") as ddb:
                doc.ddb.put(ddb)

            doc.save()

            query_result = doc_class.objects()

            assert len(query_result) == 1
            saved_doc = query_result[0]
            with tempfile.NamedTemporaryFile(mode="wb") as db_file:
                db_file.write(saved_doc.ddb.read())
                db_file.seek(0)

                assert filecmp.cmp(ddb_path, db_file.name)

    def test_abinit_phonon_output(self):

        doc_class = self.get_document_class_from_mixin(AbinitPhononOutputMixin)
        doc = doc_class()

        if has_mongodb():
            doc.structure = self.si_structure.as_dict()
            phbst_path = abidata.ref_file('ZnSe_hex_886.out_PHBST.nc')
            phdos_path = abidata.ref_file('ZnSe_hex_886.out_PHDOS.nc')
            ananc_path = abidata.ref_file('ZnSe_hex_886.anaddb.nc')

            with open(phbst_path, "rb") as phbst:
                doc.phonon_bs.put(phbst)
            with open(phdos_path, "rb") as phdos:
                doc.phonon_dos.put(phdos)
            with open(ananc_path, "rb") as anaddb_nc:
                doc.anaddb_nc.put(anaddb_nc)

            doc.save()

            query_result = doc_class.objects()

            assert len(query_result) == 1
            saved_doc = query_result[0]
            with tempfile.NamedTemporaryFile(mode="wb") as db_file:
                db_file.write(saved_doc.phonon_bs.read())
                db_file.seek(0)
                assert filecmp.cmp(phbst_path, db_file.name)

            with tempfile.NamedTemporaryFile(mode="r+b") as db_file:
                db_file.write(saved_doc.phonon_dos.read())
                db_file.seek(0)
                assert filecmp.cmp(phdos_path, db_file.name)

            with tempfile.NamedTemporaryFile(mode="r+b") as db_file:
                db_file.write(saved_doc.anaddb_nc.read())
                db_file.seek(0)
                assert filecmp.cmp(ananc_path, db_file.name)
