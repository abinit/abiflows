from abiflows.database.mongoengine.mixins import MaterialMixin, DateMixin, DirectoryMixin
from abiflows.database.mongoengine.abinit_mixins import AbinitPhononOutputMixin, AbinitBasicInputMixin, \
    AbinitGSOutputMixin, AbinitPhononOutputMixin
from mongoengine import *
from abiflows.core.models import AbiFileField, MSONField, AbiGzipFileField

class RelaxAbinitInput(AbinitBasicInputMixin, EmbeddedDocument):
    last_input = MSONField(help_text="The last input used for the calculation.")
    kppa = IntField()


class RelaxAbinitOutput(AbinitGSOutputMixin, EmbeddedDocument):
    hist_files =  DictField(field=AbiFileField(abiext="HIST.nc", abiform="b"),
                            help_text="Series of HIST files produced during the relaxation. Keys should provide the type"
                                      " of relaxation (ion, ioncell) and the ordering")
    outfile_ioncell = AbiGzipFileField(abiext="abo", abiform="t")


class RelaxResult(MaterialMixin, DateMixin, DirectoryMixin, Document):
    """
    Document containing the results for a relaxation workflow consisting of an ion followed by ioncell relaxations.
    """

    history = DictField()
    mp_id = StringField()
    abinit_input = EmbeddedDocumentField(RelaxAbinitInput, default=RelaxAbinitInput)
    abinit_output = EmbeddedDocumentField(RelaxAbinitOutput, default=RelaxAbinitOutput)
    time_report = MSONField()
    fw_id = IntField()


class PhononAbinitInput(AbinitBasicInputMixin, EmbeddedDocument):
    gs_input = MSONField(help_text="The last input used to calculate the wafunctions.")
    ddk_input = MSONField(help_text="The last input used to calculate one of the ddk.")
    dde_input = MSONField(help_text="The last input used to calculate one of the dde.")
    wfq_input = MSONField(help_text="The last input used to calculate one of the wfq.")
    phonon_input = MSONField(help_text="The last input used to calculate one of the phonons.")
    kppa = IntField()
    ngqpt = ListField(IntField())
    qpoints = DictField()
    qppa = IntField()


class PhononAbinitOutput(AbinitPhononOutputMixin, EmbeddedDocument):
    gs_gsr = AbiFileField(abiext="GSR.nc", abiform="b", help_text="Gsr file produced by the Ground state calculation")
    gs_outfile= AbiGzipFileField(abiext="abo", abiform="t")


class PhononResult(MaterialMixin, DateMixin, DirectoryMixin, Document):
    """
    Document containing the results for a phonon workflow.
    Includes information from the various steps of the workflow (scf, nscf, ddk, dde, ph, anaddb)
    """

    mp_id = StringField()
    # relax_result = ReferenceField(RelaxationResult)
    relax_db = MSONField()
    relax_id = StringField()
    time_report = MSONField()
    fw_id = IntField()
    abinit_input = EmbeddedDocumentField(PhononAbinitInput, default=PhononAbinitInput)
    abinit_output = EmbeddedDocumentField(PhononAbinitOutput, default=PhononAbinitOutput)

