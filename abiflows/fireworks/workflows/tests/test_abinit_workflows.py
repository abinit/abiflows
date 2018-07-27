"""
Tests for the Abinit workflows classes.
The tests require abipy's factories to run, so abipy should be configured and abinit available.
"""
from __future__ import print_function, division, unicode_literals, absolute_import

import os
import unittest
import abipy.data as abidata
import abipy.abilab as abilab

from abipy.abio.factories import scf_input
from abiflows.core.testing import AbiflowsTest, has_mongodb, TESTDB_NAME
from abiflows.database.mongoengine.utils import DatabaseData
from abiflows.fireworks.workflows.abinit_workflows import *


class TestBaseClassMethods(AbiflowsTest):

    @classmethod
    def setUpClass(cls):
        cls.si_structure = abilab.Structure.from_file(abidata.cif_file("si.cif"))
        cls.scf_inp = scf_input(cls.si_structure, abidata.pseudos("14si.pspnc"), ecut=2, kppa=10)
        cls.setup_fireworks()

    @classmethod
    def tearDownClass(cls):
        cls.teardown_fireworks()

    def setUp(self):
        self.scf_wf = ScfFWWorkflow(self.scf_inp)

    def tearDown(self):
        if self.lp:
            self.lp.reset(password=None,require_password=False)

    def test_add_fws(self):
        assert len(self.scf_wf.wf.fws) == 1
        self.scf_wf.add_final_cleanup(out_exts=["WFK", "DEN"])
        assert len(self.scf_wf.wf.fws) == 2
        self.scf_wf.add_mongoengine_db_insertion(DatabaseData(TESTDB_NAME))
        assert len(self.scf_wf.wf.fws) == 3
        self.scf_wf.add_cut3d_den_to_cube_task()
        assert len(self.scf_wf.wf.fws) == 4

    def test_fireworks_methods(self):
        self.scf_wf.add_metadata(self.si_structure, {"test": 1})
        assert "nsites" in self.scf_wf.wf.metadata

        self.scf_wf.fix_fworker("test_worker")
        self.scf_wf.get_reduced_formula(self.scf_inp)
        self.scf_wf.set_short_single_core_to_spec()
        self.scf_wf.set_preserve_fworker()
        self.scf_wf.add_spec_to_all_fws({"test_spec": 1})

    @unittest.skipUnless(has_mongodb(), "A local mongodb is required.")
    def test_add_to_db(self):
        self.scf_wf.add_to_db(self.lp)


class TestFromFactory(AbiflowsTest):

    @classmethod
    def setUpClass(cls):
        cls.gan_structure = abilab.Structure.from_file(abidata.cif_file("gan.cif"))
        cls.gan_pseudos = [abidata.pseudos("31ga.pspnc").pseudo_with_symbol('Ga'),
               abidata.pseudos("7n.pspnc").pseudo_with_symbol('N')]

    def test_scf_workflow(self):
        ScfFWWorkflow.from_factory(self.gan_structure, self.gan_pseudos, ecut=4, spec={"test": 1},
                                   initialization_info={"test": 1})

        ScfFWWorkflow.from_factory(self.gan_structure, self.gan_pseudos, ecut=4, spec={"test": 1},
                                   initialization_info={"test": 1}, autoparal=True)

    def test_phonon_workflow(self):
        PhononFWWorkflow.from_factory(self.gan_structure, self.gan_pseudos, ecut=4, spec={"test": 1},
                                      initialization_info={"test": 1}, ph_ngqpt=[2,2,2])

        PhononFWWorkflow.from_factory(self.gan_structure, self.gan_pseudos, ecut=4, spec={"test": 1},
                                      initialization_info={"test": 1}, ph_ngqpt=[2,2,2], autoparal=True)

        PhononFullFWWorkflow.from_factory(self.gan_structure, self.gan_pseudos, ecut=4, spec={"test": 1},
                                          initialization_info={"test": 1}, qpoints=[[0.1,0,0]])

    def test_dte_workflow(self):
        DteFWWorkflow.from_factory(self.gan_structure, self.gan_pseudos, ecut=4, smearing=None,
                                   spin_mode="unpolarized", spec={"test": 1},
                                   initialization_info={"test": 1}, extra_abivars={"ixc": 7})

        DteFWWorkflow.from_factory(self.gan_structure, self.gan_pseudos, ecut=4, smearing=None,
                                   spin_mode="unpolarized", spec={"test": 1},
                                   initialization_info={"test": 1}, extra_abivars={"ixc": 7}, autoparal=True)

    def test_relax_workflow(self):
        RelaxFWWorkflow.from_factory(self.gan_structure, self.gan_pseudos, ecut=4, spec={"test": 1},
                                     initialization_info={"test": 1})

        RelaxFWWorkflow.from_factory(self.gan_structure, self.gan_pseudos, ecut=4, spec={"test": 1},
                                     initialization_info={"test": 1}, autoparal=True)

        RelaxFWWorkflow.from_factory(self.gan_structure, self.gan_pseudos, ecut=4, spec={"test": 1},
                                     initialization_info={"test": 1}, target_dilatmx=1.01)

    def test_dfpt_workflow(self):
        # set ixc otherwise the dte part will fail
        extra_abivars = {"ixc": 7}
        DfptFWWorkflow.from_factory(self.gan_structure, self.gan_pseudos, ecut=4, spec={"test": 1},
                                    initialization_info={"test": 1}, ph_ngqpt=[2,2,2], do_ddk=True,
                                    do_dde=True, do_strain=True, do_dte=True, smearing=None,
                                    spin_mode="unpolarized", extra_abivars=extra_abivars)

        DfptFWWorkflow.from_factory(self.gan_structure, self.gan_pseudos, ecut=4, spec=None,
                                    initialization_info=None, ph_ngqpt=[2, 2, 2], do_ddk=True, do_dde=True,
                                    do_strain=False, do_dte=True, extra_abivars=extra_abivars,
                                    smearing=None, spin_mode="unpolarized", autoparal=True)



class TestFromPreviousInput(AbiflowsTest):

    @classmethod
    def setUp(cls):
        cls.gan_structure = abilab.Structure.from_file(abidata.cif_file("gan.cif"))
        cls.gan_pseudos = [abidata.pseudos("31ga.pspnc").pseudo_with_symbol('Ga'),
               abidata.pseudos("7n.pspnc").pseudo_with_symbol('N')]
        cls.scf_inp = scf_input(cls.gan_structure, cls.gan_pseudos, ecut=2, kppa=10, smearing=None,
                                spin_mode="unpolarized")

    def test_phonon_workflow(self):
        PhononFWWorkflow.from_gs_input(gs_input=self.scf_inp, spec={"test": 1},
                                      initialization_info={"test": 1}, ph_ngqpt=[2,2,2])

    def test_dte_workflow(self):
        self.scf_inp['ixc'] = 7
        DteFWWorkflow.from_gs_input(gs_input=self.scf_inp, spec={"test": 1},
                                   initialization_info={"test": 1})

    def test_dfpt_workflow(self):
        self.scf_inp['ixc'] = 7
        DfptFWWorkflow.from_gs_input(gs_input=self.scf_inp, spec=None,
                                     initialization_info=None, ph_ngqpt=[2,2,2],
                                     do_ddk=True,do_dde=True, do_strain=True, do_dte=True)

