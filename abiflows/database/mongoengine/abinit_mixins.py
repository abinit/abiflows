# coding: utf-8
"""
List of mixins to provide standard interfaces with the result databases for abinit calculations.
Most of them should still be considered as examples.
"""
from __future__ import print_function, division, unicode_literals

import os
from mongoengine import *
from abiflows.core.models import AbiFileField, MSONField
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.abinit.pseudos import Pseudo
from abiflows.database.mongoengine.mixins import GroundStateOutputMixin

class AbinitPseudoData(EmbeddedDocument):
    """
    Embedded document providing some fields and function to save abinit pseudopotential data
    """

    pseudos_name = ListField(StringField())
    pseudos_md5 = ListField(StringField())
    pseudos_path = ListField(StringField())

    def set_pseudos_vars(self, pseudos_path):
        # this should be compatible with both version prior and after 0.3 of the pseudo dojo
        pseudos_name = []
        pseudos_md5 = []
        for path in pseudos_path:
            pseudo = Pseudo.from_file(path)
            pseudos_name.append(pseudo.basename)
            pseudos_md5.append(pseudo.md5)

        self.pseudos_name = pseudos_name
        self.pseudos_md5 = pseudos_md5
        self.pseudos_path = pseudos_path

    def set_pseudos_from_files_file(self, files_file_path, num_pseudo):
        pseudos_path = []
        with open(files_file_path) as f:
            lines = f.readlines()
        #remove possible empty lines and white characters and newlines
        lines = [l.strip() for l in lines if l]
        run_dir = os.path.dirname(os.path.abspath(files_file_path))
        for pseudo_line in lines[-num_pseudo:]:
            if os.path.isabs(pseudo_line):
                pseudos_path.append(pseudo_line)
            else:
                pseudos_path.append(os.path.abspath(os.path.join(run_dir, pseudo_line)))

        self.set_pseudos_vars(pseudos_path)


class AbinitBasicInputMixin(object):
    """
    Mixin providing some basic fields that are required to run a calculation.
    """
    #TODO add more variables
    structure = MSONField()
    ecut = FloatField()
    nshiftk = IntField()
    shiftk = ListField(ListField(FloatField()))
    ngkpt = ListField(IntField())
    kptrlatt = ListField(ListField(IntField()))
    dilatmx = FloatField(default=1)
    occopt = IntField()
    tsmear = FloatField()
    pseudopotentials = EmbeddedDocumentField(AbinitPseudoData, default=AbinitPseudoData)


    def set_abinit_basic_from_abinit_input(self, abinit_input):
        """
        create the object from an AbinitInput object
        """
        self.structure = abinit_input.structure.as_dict()
        self.ecut = abinit_input['ecut']
        # kpoints may be defined in different ways
        self.nshiftk = abinit_input.get('nshiftk', None)
        self.shiftk = abinit_input.get('shiftk', None)
        self.ngkpt = abinit_input.get('ngkpt', None)
        self.kptrlatt = abinit_input.get('kptrlatt', None)
        self.dilatmx = abinit_input.get('dilatmx', 1)
        self.occopt = abinit_input.get('occopt', 1)
        self.tsmear = abinit_input.get('tsmear', None)


class AbinitGSOutputMixin(GroundStateOutputMixin):
    """
    Mixin providing generic fiels for abinit ground state calculation
    """
    gsr = AbiFileField(abiext="GSR.nc", abiform="b", help_text="Gsr file produced by the Ground state calculation",
                       db_field='gsr_id', collection_name='gs_gsr_fs')


class AbinitDftpOutputMixin(object):
    """
    Mixin providing generic fiels for dfpt calculation
    """

    ddb = AbiFileField(abiext="DDB", abiform="t", help_text="DDB file produced by a dfpt falculation",
                       db_field='ddb_id', collection_name='ddb_fs')
    structure = MSONField()


class AbinitPhononOutputMixin(AbinitDftpOutputMixin):
    """
    Mixin providing generic fiels for phonon calculation
    """

    phonon_bs = AbiFileField(abiext="PHBST.nc", abiform="b", db_field='phonon_bs_id', collection_name='phonon_bs_fs')
    phonon_dos = AbiFileField(abiext="PHDOS.nc", abiform="b", db_field='phonon_dos_id', collection_name='phonon_dos_fs')
    anaddb_nc = AbiFileField(abiext="anaddb.nc", abiform="b", db_field='anaddb_nc_id', collection_name='anaddb_nc_fs')