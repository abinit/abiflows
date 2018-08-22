from __future__ import print_function, division, unicode_literals, absolute_import

from abiflows.database.mongoengine.mixins import MaterialMixin, DateMixin, DirectoryMixin, CustomFieldMixin
from abiflows.database.mongoengine.abinit_mixins import AbinitPhononOutputMixin, AbinitBasicInputMixin, \
    AbinitGSOutputMixin, AbinitPhononOutputMixin, AbinitDftpOutputMixin
from mongoengine import *
from abiflows.core.models import AbiFileField, MSONField, AbiGzipFileField

class RelaxAbinitInput(AbinitBasicInputMixin, EmbeddedDocument):
    """
    EmbeddedDocument containing the typical inputs for a relaxation workflow.

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: RelaxAbinitInput
    """

    last_input = MSONField(help_text="The last input used for the calculation.")
    kppa = IntField()


class RelaxAbinitOutput(AbinitGSOutputMixin, EmbeddedDocument):
    """
    EmbeddedDocument containing the typical outputs for a relaxation workflow.

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: RelaxAbinitOutput
    """

    hist_files =  DictField(field=AbiFileField(abiext="HIST.nc", abiform="b", collection_name='relax_hist_fs'),
                            help_text="Series of HIST files produced during the relaxation. Keys should provide the type"
                                      " of relaxation (ion, ioncell) and the ordering", db_field='relax_hist_files_ids')
    outfile_ioncell = AbiGzipFileField(abiext="abo", abiform="t", db_field='outfile_ioncell_id',
                                       collection_name='relax_outfile_fs')


class RelaxResult(MaterialMixin, DateMixin, DirectoryMixin, CustomFieldMixin, Document):
    """
    Document containing the results for a relaxation workflow consisting of an ion followed by ioncell relaxations.

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: RelaxResult
    """

    history = DictField()
    mp_id = StringField()
    abinit_input = EmbeddedDocumentField(RelaxAbinitInput, default=RelaxAbinitInput)
    abinit_output = EmbeddedDocumentField(RelaxAbinitOutput, default=RelaxAbinitOutput)
    time_report = MSONField()
    fw_id = IntField()


class PhononAbinitInput(AbinitBasicInputMixin, EmbeddedDocument):
    """
    EmbeddedDocument containing the typical inputs for a phonon workflow.

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: PhononAbinitInput
    """

    gs_input = MSONField(help_text="The last input used to calculate the wafunctions.")
    ddk_input = MSONField(help_text="The last input used to calculate one of the ddk.")
    dde_input = MSONField(help_text="The last input used to calculate one of the dde.")
    wfq_input = MSONField(help_text="The last input used to calculate one of the wfq.")
    phonon_input = MSONField(help_text="The last input used to calculate one of the phonons.")
    kppa = IntField()
    ngqpt = ListField(IntField())
    qpoints = ListField()
    qppa = IntField()


class PhononAbinitOutput(AbinitPhononOutputMixin, EmbeddedDocument):
    """
    EmbeddedDocument containing the typical outputs for a phonon workflow.

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: PhononAbinitOutput
    """

    gs_gsr = AbiFileField(abiext="GSR.nc", abiform="b", help_text="Gsr file produced by the Ground state calculation",
                          db_field='gs_gsr_id', collection_name='phonon_gs_gsr_fs')
    gs_outfile= AbiGzipFileField(abiext="abo", abiform="t", db_field='gs_outfile_id',
                                 collection_name='phonon_gs_outfile_fs')


class PhononResult(MaterialMixin, DateMixin, DirectoryMixin, CustomFieldMixin, Document):
    """
    Document containing the results for a phonon workflow.
    Includes information from the various steps of the workflow (scf, nscf, ddk, dde, ph, anaddb)

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: PhononResult
    """

    mp_id = StringField()
    # relax_result = ReferenceField(RelaxationResult)
    relax_db = MSONField()
    relax_id = StringField()
    time_report = MSONField()
    fw_id = IntField()
    abinit_input = EmbeddedDocumentField(PhononAbinitInput, default=PhononAbinitInput)
    abinit_output = EmbeddedDocumentField(PhononAbinitOutput, default=PhononAbinitOutput)


class DteAbinitInput(AbinitBasicInputMixin, EmbeddedDocument):
    """
    EmbeddedDocument containing the typical inputs for a DTE workflow.

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: DteAbinitInput
    """

    gs_input = MSONField(help_text="The last input used to calculate the wafunctions.")
    ddk_input = MSONField(help_text="The last input used to calculate one of the ddk.")
    dde_input = MSONField(help_text="The last input used to calculate one of the dde.")
    dte_input = MSONField(help_text="The last input used to calculate one of the dte.")
    phonon_input = MSONField(help_text="The last input used to calculate one of the phonons.")
    kppa = IntField()
    with_phonons = BooleanField(help_text="Whether the phonon perturbations have been included or not")


class DteAbinitOutput(AbinitDftpOutputMixin, EmbeddedDocument):
    """
    EmbeddedDocument containing the typical outputs for a DTE workflow.

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: DteAbinitOutput
    """

    gs_gsr = AbiFileField(abiext="GSR.nc", abiform="b", help_text="Gsr file produced by the Ground state calculation",
                          db_field='gs_gsr_id', collection_name='phonon_gs_gsr_fs')
    gs_outfile= AbiGzipFileField(abiext="abo", abiform="t", db_field='gs_outfile_id',
                                 collection_name='phonon_gs_outfile_fs')
    anaddb_nc = AbiFileField(abiext="anaddb.nc", abiform="b", db_field='anaddb_nc_id', collection_name='anaddb_nc_fs')

    epsinf = ListField(ListField(FloatField()), help_text="Macroscopic dielectric tensor")
    eps0 = ListField(ListField(FloatField()), help_text="Relaxed ion Macroscopic dielectric tensor")
    dchide= ListField(ListField(ListField(FloatField())), help_text="Non-linear optical susceptibilities tensor")
    dchidt= ListField(ListField(ListField(ListField(FloatField()))),
                      help_text="First-order change in the linear dielectric susceptibility")


class DteResult(MaterialMixin, DateMixin, DirectoryMixin, CustomFieldMixin, Document):
    """
    Document containing the results for a dte workflow.
    Includes information from the various steps of the workflow (scf, ddk, dde, ph, dte, anaddb)

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: DteResult
    """

    mp_id = StringField()
    relax_db = MSONField()
    relax_id = StringField()
    time_report = MSONField()
    fw_id = IntField()
    abinit_input = EmbeddedDocumentField(DteAbinitInput, default=DteAbinitInput)
    abinit_output = EmbeddedDocumentField(DteAbinitOutput, default=DteAbinitOutput)


class DfptAbinitInput(AbinitBasicInputMixin, EmbeddedDocument):
    """
    EmbeddedDocument containing the typical inputs for a dfpt workflow.

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: DfptAbinitInput
    """

    gs_input = MSONField(help_text="The last input used to calculate the wafunctions.")
    ddk_input = MSONField(help_text="The last input used to calculate one of the ddk.")
    dde_input = MSONField(help_text="The last input used to calculate one of the dde.")
    wfq_input = MSONField(help_text="The last input used to calculate one of the wfq.")
    phonon_input = MSONField(help_text="The last input used to calculate one of the phonons.")
    strain_input = MSONField(help_text="The last input used to calculate one of the strain.")
    dte_input = MSONField(help_text="The last input used to calculate one of the dte.")
    kppa = IntField()
    ngqpt = ListField(IntField())
    qpoints = ListField()
    qppa = IntField()


class DfptAbinitOutput(AbinitPhononOutputMixin, EmbeddedDocument):
    """
    EmbeddedDocument containing the typical outputs for a dfpt workflow.

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: DfptAbinitOutput
    """

    gs_gsr = AbiFileField(abiext="GSR.nc", abiform="b", help_text="Gsr file produced by the Ground state calculation",
                          db_field='gs_gsr_id', collection_name='phonon_gs_gsr_fs')
    gs_outfile= AbiGzipFileField(abiext="abo", abiform="t", db_field='gs_outfile_id',
                                 collection_name='phonon_gs_outfile_fs')


class DfptResult(MaterialMixin, DateMixin, DirectoryMixin, CustomFieldMixin, Document):
    """
    Document containing the results for a dfpt workflow.
    Includes information from the various steps of the workflow (scf, nscf, ddk, dde, ph, strain, dte, anaddb)

    .. rubric:: Inheritance Diagram
    .. inheritance-diagram:: DfptResult
    """

    mp_id = StringField()
    relax_db = MSONField()
    relax_id = StringField()
    time_report = MSONField()
    fw_id = IntField()
    abinit_input = EmbeddedDocumentField(DfptAbinitInput, default=DfptAbinitInput)
    abinit_output = EmbeddedDocumentField(DfptAbinitOutput, default=DfptAbinitOutput)

    has_phonons = BooleanField(help_text="Whether the phonon perturbations have been included or not", default=False)
    has_ddk = BooleanField(help_text="Whether the ddk perturbations have been included or not", default=False)
    has_dde = BooleanField(help_text="Whether the dde perturbations have been included or not", default=False)
    has_strain = BooleanField(help_text="Whether the strain perturbations have been included or not", default=False)
    has_dte = BooleanField(help_text="Whether the dte perturbations have been included or not", default=False)
