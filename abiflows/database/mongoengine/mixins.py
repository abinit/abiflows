# coding: utf-8
"""
List of mixins to provide standard interfaces with the result databases.
Most of them should still be considered as examples.
"""
from __future__ import print_function, division, unicode_literals, absolute_import

from monty.dev import deprecated
from mongoengine import *
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.periodic_table import Element
from abiflows.core.models import AbiFileField, MSONField


class CalculationMetadataMixin(object):
    """
    Mixin providing fields for the metadata of the calculation
    """

    user = StringField(help_text="Username of the author of the calculation")
    cluster = StringField(help_text="Cluster where the calculation was performed")
    execution_date = DateTimeField(help_text="Date of the conclusion of the calculation")


class RunStatsMixin(object):
    """
    Mixin providing fields for the statistics of execution of the calculation
    """

    core_num = IntField(help_text="Number of cores used to run the calculation")
    elapsed_time = FloatField(help_text="Time in seconds required to perform the calculation")
    maximum_memory_used = FloatField(help_text="Maximum memory used during the calculation in kb")
    number_of_restarts = IntField(help_text="Number of times the calculation has been restarted")
    number_of_errors = IntField(help_text="Number of errors corrected by error handlers")
    number_of_warnings = IntField(help_text="Number of warnings in the last run")


class SpaceGroupMixin(object):
    """
    Mixin providing fields to save space group data.
    """

    crystal_system = StringField()
    hall = StringField()
    number = IntField()
    point_group = StringField()
    source = StringField()
    symbol = StringField()
    #TODO tolerances?

    @deprecated(message="fill_from_structure has been renamed set_space_group_from_structure.")
    def fill_from_structure(self, structure):
        self.set_space_group_from_structure(structure)

    def set_space_group_from_structure(self, structure):
        spga = SpacegroupAnalyzer(structure=structure)
        self.crystal_system = spga.get_crystal_system()
        self.hall = spga.get_hall()
        self.number = spga.get_space_group_number()
        self.source = "spglib"
        self.symbol = spga.get_space_group_symbol()


class SpaceGroupDocument(SpaceGroupMixin, EmbeddedDocument):
    """
    Embedded document to describe the spacegroup of a material
    """
    pass


class MaterialMixin(object):
    """
    Mixin providing the fields describing the material examined in the calculation
    """

    pretty_formula = StringField(help_text="A nice string formula where the element amounts are normalized")
    full_formula = StringField(help_text="A string of the full explicit formula for the unit cell")
    spacegroup = EmbeddedDocumentField(SpaceGroupDocument)
    unit_cell_formula = DictField(help_text="A dict of the full (unnormalized) formula of the unit cell")
    reduced_cell_formula = DictField(help_text="A dict of the normalized cell formula")
    anonymous_formula = StringField(help_text="A string of the anonymous formula for the material")
    elements = ListField(StringField(), help_text="A list of element symbols present in this material")
    nelements = IntField(help_text="Number of elements in this material's chemical system")
    nsites = IntField(help_text="Number of sites in the structure")
    chemsys = StringField(help_text="A string chemical system associated with the material")

    def set_material_data_from_structure(self, structure, space_group=True, symprec=1e-3, angle_tolerance=5):
        """
        Sets the fields of the Document using a Structure and Spglib to determine the space group properties

        Args:
            structure: A |Structure|
            space_group: if True sets the spacegroup fields using spglib_.
            symprec (float): Tolerance for symmetry finding.
            angle_tolerance (float): Angle tolerance for symmetry finding.
        """

        comp = structure.composition
        el_amt = structure.composition.get_el_amt_dict()
        self.unit_cell_formula = comp.as_dict()
        self.reduced_cell_formula = comp.to_reduced_dict
        self.elements = list(el_amt.keys())
        self.nelements = len(el_amt)
        self.pretty_formula = comp.reduced_formula
        self.anonymous_formula = comp.anonymized_formula
        self.nsites = comp.num_atoms
        self.chemsys = "-".join(sorted(el_amt.keys()))
        if space_group:
            sym = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
            self.spacegroup = SpaceGroupDocument(crystal_system=sym.get_crystal_system(), hall=sym.get_hall(),
                                                 number=sym.get_space_group_number(), point_group=sym.get_point_group_symbol(),
                                                 symbol=sym.get_space_group_symbol(), source="spglib")


class CalculationTypeMixin(object):
    """
    Mixin providing fields for basic details of the calculation
    """

    xc_functional = StringField(help_text="Specific exchange-correlation functional used for the calculation")
    pseudo_type = StringField(help_text="Type of the pseudopotential", choices=["nc", "us", "paw"])
    is_hubbard = BooleanField(help_text="True if calculation was performed with hubbard correction")
    pseudo_dojo_table = StringField(help_text="String describing the version of pseudo dojo table")
    # other possibile fields: magnetization_type (unpolarized, polarized, non collinear), has_spin_orbit


class GroundStateOutputMixin(object):
    """
    Mixin providing generic fields for ground state calculation
    """

    #TODO would it be convenient to make these FloatWithUnitField?
    final_energy = FloatField(help_text="Final energy obtained computed in eV")
    efermi = FloatField(help_text="Computed Fermi energy in eV")
    total_magnetization = FloatField(help_text="Total magnetization computed") # does pymatgen have a default unit for this?

    structure = MSONField(required=True, help_text="The structure used for the calculation. If a relaxation the final structure.")


class HubbardsField(DictField):
    """
    Dictfield to validate the passed values
    """

    def validate(self, value):
        if not all(Element.is_valid_symbol(k) for k in value.keys()):
            self.error('Keys should be element symbols')
        if not all(isinstance(v, (float, int)) for v in value.values()):
            self.error('Values should be numbers')
        super(DictField, self).validate(value)


class HubbardMixin(object):
    """
    Mixin providing hubbard fields
    """
    hubbards = HubbardsField(help_text="Dict containing the values of the Hubbard U values for each element")


class DateMixin(object):
    """
    Mixin providing timestamp fields
    """

    created_on = DateTimeField()
    modified_on = DateTimeField()


class DirectoryMixin(object):
    """
    Mixin providing the field to store the directories used during the workflow
    """

    dir_names = DictField(help_text="Keys are the folder containing the various steps of the calculation, with a "
                                    "description of the step as value. Task name as value is the best option. "
                                    "None is accettable")

    def set_dir_names_from_fws_wf(self, wf):
        """
        Fills the the dir_names of the mixin using the COMPLETED launches of a firework workflow.
        Needs access to the full data of the workflow: Fireworks and Launches.
        """

        #FIXME should the imports be moved to the top? Would it be better to not rely on external objects/functions?
        from abiflows.fireworks.utils.fw_utils import get_last_completed_launch

        d = {}

        for fw in wf.fws:
            task_index = fw.spec.get('wf_task_index', None)
            if task_index is None and "unnamed" not in fw.name.lower():
                task_index = fw.name

            launch = get_last_completed_launch(fw)

            # skip the fws that have not run yet
            if launch is None:
                continue

            d[launch.launch_dir] = task_index

        self.dir_names = d


class CustomFieldMixin(object):
    """
    Mixin providing a DictField for storing user specific additional properties
    """

    custom = DictField(help_text="Dict storing additional custom properties to identify the calculation.")
