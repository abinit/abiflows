from abiflows.database.mongoengine.mixins import MaterialMixin, DateMixin, DirectoryMixin
from abiflows.database.mongoengine.abinit_mixins import AbinitPseudoMixin, AbinitRelaxMixin, AbinitPhononMixin, AbinitBasicMixin
from mongoengine import *
from abiflows.core.models import AbiFileField, MSONField, AbiGzipFileField

class RelaxAbinitData(AbinitBasicMixin, EmbeddedDocument):
    last_input = MSONField(help_text="The last input used for the calculation.")
    # hist_nc_file_ioncell = AbiFileField(abiext="HIST", abiform="b")
    # hist_files =  DictField(field=AbiFileField(abiext="HIST.nc", abiform="b"),
    #                         help_text="Series of HIST files produced during the relaxation. Keys should provide the type"
    #                                   " of relaxation (ion, ioncell) and the ordering")
    outfile_ioncell = AbiGzipFileField(abiext="abo", abiform="t")

class RelaxResult(MaterialMixin, AbinitPseudoMixin, AbinitRelaxMixin, DateMixin, DirectoryMixin, Document):
    """
    Document containing the results for a relaxation workflow consisting of an ion followed by ioncell relaxations.
    """

    history = DictField()
    mp_id = StringField()
    abinit_data = EmbeddedDocumentField(RelaxAbinitData, default=RelaxAbinitData)
    time_report = MSONField()
    fw_id = IntField()
    kppa = IntField()
    hist_files =  DictField(field=AbiFileField(abiext="HIST.nc", abiform="b"),
                            help_text="Series of HIST files produced during the relaxation. Keys should provide the type"
                                      " of relaxation (ion, ioncell) and the ordering")


class PhononAbinitData(AbinitBasicMixin, EmbeddedDocument):
    gs_input = MSONField(help_text="The last input used to calculate the wafunctions.")
    ddk_input = MSONField(help_text="The last input used to calculate one of the ddk.")
    dde_input = MSONField(help_text="The last input used to calculate one of the dde.")
    wfq_input = MSONField(help_text="The last input used to calculate one of the wfq.")
    ph_input = MSONField(help_text="The last input used to calculate one of the phonons.")

    gs_outfile= AbiGzipFileField(abiext="abo", abiform="t")


class PhononResult(MaterialMixin, AbinitPseudoMixin, AbinitPhononMixin, DateMixin, DirectoryMixin, Document):
    """
    Document containing the results for a phonon workflow.
    Includes information from the various steps of the workflow (scf, nscf, ddk, dde, ph, anaddb)
    """

    kppa = IntField()
    ngqpt = ListField(IntField())
    qpoints = DictField()
    qppa = IntField()
    mp_id = StringField()
    # relax_result = ReferenceField(RelaxationResult)
    relax_db = MSONField()
    relax_id = StringField()
    time_report = MSONField()
    fw_id = IntField()
    gs_gsr = AbiFileField(abiext="GSR.nc", abiform="b", help_text="Gsr file produced by the Ground state calculation")
    abinit_data = EmbeddedDocumentField(PhononAbinitData, default=PhononAbinitData)

