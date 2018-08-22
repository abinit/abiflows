from __future__ import print_function, division, unicode_literals, absolute_import

import os
import unittest
import filecmp
import tempfile
import numpy as np
import abipy.data as abidata
import abipy.abilab as abilab

from abipy.abio.factories import scf_input
from abiflows.core.testing import AbiflowsTest, has_mongodb, TESTDB_NAME
from abiflows.database.mongoengine.abinit_results import *


class TestAbinitResults(AbiflowsTest):
    """
    For all the classes, try to create an instance and call methods to fill the document.
    If mongodb is available, save the document.
    """

    @classmethod
    def setUpClass(cls):
        cls.si_structure = abilab.Structure.from_file(abidata.cif_file("si.cif"))
        cls.scf_inp = scf_input(cls.si_structure, abidata.pseudos("14si.pspnc"), ecut=2, kppa=10)
        cls.out_file = os.path.join(abidata.dirpath, 'refs', 'si_ebands', 'run.abo')

    def setUp(self):
        self.setup_mongoengine()

    @classmethod
    def tearDownClass(cls):
        cls.teardown_mongoengine()

    def test_relax_result(self):

        doc = RelaxResult(history={"test": 1}, mp_id="mp-1", time_report= {"test": 1}, fw_id=1)

        doc.abinit_input.structure = self.si_structure.as_dict()
        doc.abinit_input.last_input = self.scf_inp.as_dict()
        doc.abinit_input.kppa = 1000

        doc.abinit_output.structure = self.si_structure.as_dict()

        if has_mongodb():
            hist_path = abidata.ref_file('sic_relax_HIST.nc')

            with open(hist_path, "rb") as hist:
                # the proxy class and collection name of the hist file field
                proxy_class = RelaxResult.abinit_output.default.hist_files.field.proxy_class
                collection_name = RelaxResult.abinit_output.default.hist_files.field.collection_name
                file_field = proxy_class(collection_name=collection_name)
                file_field.put(hist)

                doc.abinit_output.hist_files = {'test_hist': file_field}

            outfile_path = self.out_file

            # read/write in binary for py3k compatibility with mongoengine
            with open(outfile_path, "rb") as outfile:
                doc.abinit_output.outfile_ioncell.put(outfile)

            doc.save()

            query_result = RelaxResult.objects()

            assert len(query_result) == 1
            saved_doc = query_result[0]
            with tempfile.NamedTemporaryFile(mode="wb") as db_file:
                db_file.write(saved_doc.abinit_output.hist_files["test_hist"].read())
                db_file.seek(0)
                assert filecmp.cmp(hist_path, db_file.name)

    def test_phonon_result(self):

        doc = PhononResult(mp_id="mp-1", time_report= {"test": 1}, fw_id=1, relax_db={"test": 1},
                           relax_id="id_string")

        doc.abinit_input.structure = self.si_structure.as_dict()
        doc.abinit_input.gs_input = self.scf_inp.as_dict()
        doc.abinit_input.ddk_input = self.scf_inp.as_dict()
        doc.abinit_input.dde_input = self.scf_inp.as_dict()
        doc.abinit_input.wfq_input = self.scf_inp.as_dict()
        doc.abinit_input.phonon_input = self.scf_inp.as_dict()
        doc.abinit_input.kppa = 1000
        doc.abinit_input.ngqpt = [4,4,4]
        doc.abinit_input.qppa = 1000
        doc.abinit_output.structure = self.si_structure.as_dict()

        if has_mongodb():
            gsr_path = abidata.ref_file('si_scf_GSR.nc')

            with open(gsr_path, "rb") as gsr:
                doc.abinit_output.gs_gsr.put(gsr)

            gs_outfile_path = self.out_file

            # read/write in binary for py3k compatibility with mongoengine
            with open(gs_outfile_path, "rb") as gs_outfile:
                doc.abinit_output.gs_outfile.put(gs_outfile)

            doc.save()

            query_result = PhononResult.objects()

            assert len(query_result) == 1
            saved_doc = query_result[0]
            with tempfile.NamedTemporaryFile(mode="wb") as db_file:
                db_file.write(saved_doc.abinit_output.gs_gsr.read())
                db_file.seek(0)
                assert filecmp.cmp(gsr_path, db_file.name)

            with tempfile.NamedTemporaryFile(mode="wt") as db_file:
                saved_doc.abinit_output.gs_outfile.unzip(filepath=db_file.name)
                db_file.seek(0)
                assert filecmp.cmp(gs_outfile_path, db_file.name)

    def test_dte_result(self):

        doc = DteResult(mp_id="mp-1", time_report= {"test": 1}, fw_id=1, relax_db={"test": 1},
                           relax_id="id_string")

        doc.abinit_input.structure = self.si_structure.as_dict()
        doc.abinit_input.gs_input = self.scf_inp.as_dict()
        doc.abinit_input.ddk_input = self.scf_inp.as_dict()
        doc.abinit_input.dde_input = self.scf_inp.as_dict()
        doc.abinit_input.dte_input = self.scf_inp.as_dict()
        doc.abinit_input.phonon_input = self.scf_inp.as_dict()
        doc.abinit_input.kppa = 1000
        doc.abinit_input.with_phonons = True
        doc.abinit_output.structure = self.si_structure.as_dict()
        doc.abinit_output.epsinf = np.eye(3).tolist()
        doc.abinit_output.eps0 = np.eye(3).tolist()
        doc.abinit_output.dchide = np.arange(36).reshape((4,3,3)).tolist()
        doc.abinit_output.dchidt = np.arange(36).reshape((2,2,3,3)).tolist()

        if has_mongodb():
            gsr_path = abidata.ref_file('si_scf_GSR.nc')

            with open(gsr_path, "rb") as gsr:
                doc.abinit_output.gs_gsr.put(gsr)

            gs_outfile_path = self.out_file

            # read/write in binary for py3k compatibility with mongoengine
            with open(gs_outfile_path, "rb") as gs_outfile:
                doc.abinit_output.gs_outfile.put(gs_outfile)

            anaddb_nc_path = abidata.ref_file('ZnSe_hex_886.anaddb.nc')

            with open(anaddb_nc_path, "rb") as anaddb_nc:
                doc.abinit_output.anaddb_nc.put(anaddb_nc)

            doc.save()

    def test_dfpt_result(self):

        doc = DfptResult(mp_id="mp-1", time_report={"test": 1}, fw_id=1, relax_db={"test": 1},
                        relax_id="id_string")

        doc.abinit_input.structure = self.si_structure.as_dict()
        doc.abinit_input.gs_input = self.scf_inp.as_dict()
        doc.abinit_input.ddk_input = self.scf_inp.as_dict()
        doc.abinit_input.dde_input = self.scf_inp.as_dict()
        doc.abinit_input.dde_input = self.scf_inp.as_dict()
        doc.abinit_input.wfq_input = self.scf_inp.as_dict()
        doc.abinit_input.strain_input = self.scf_inp.as_dict()
        doc.abinit_input.dte_input = self.scf_inp.as_dict()
        doc.abinit_input.phonon_input = self.scf_inp.as_dict()
        doc.abinit_input.kppa = 1000
        doc.abinit_output.structure = self.si_structure.as_dict()

        if has_mongodb():
            gsr_path = abidata.ref_file('si_scf_GSR.nc')

            with open(gsr_path, "rb") as gsr:
                doc.abinit_output.gs_gsr.put(gsr)

            gs_outfile_path = self.out_file

            # read/write in binary for py3k compatibility with mongoengine
            with open(gs_outfile_path, "rb") as gs_outfile:
                doc.abinit_output.gs_outfile.put(gs_outfile)

            anaddb_nc_path = abidata.ref_file('ZnSe_hex_886.anaddb.nc')

            with open(anaddb_nc_path, "rb") as anaddb_nc:
                doc.abinit_output.anaddb_nc.put(anaddb_nc)

            doc.save()

            query_result = DfptResult.objects()

            assert len(query_result) == 1
            saved_doc = query_result[0]
            with tempfile.NamedTemporaryFile(mode="wb") as db_file:
                db_file.write(saved_doc.abinit_output.gs_gsr.read())
                db_file.seek(0)
                assert filecmp.cmp(gsr_path, db_file.name)

            with tempfile.NamedTemporaryFile(mode="wt") as db_file:
                saved_doc.abinit_output.gs_outfile.unzip(filepath=db_file.name)
                db_file.seek(0)
                assert filecmp.cmp(gs_outfile_path, db_file.name)

            with tempfile.NamedTemporaryFile(mode="wb") as db_file:
                db_file.write(saved_doc.abinit_output.anaddb_nc.read())
                db_file.seek(0)
                assert filecmp.cmp(anaddb_nc_path, db_file.name)
