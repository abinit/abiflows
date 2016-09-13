from pymatgen.io.vasp.sets import MPRelaxSet, MITNEBSet


class MPNEBSet(MPRelaxSet, MITNEBSet):

    def __init__(self, structures, unset_encut=False, **kwargs):
        if len(structures) < 3:
            raise ValueError("You need at least 3 structures for an NEB.")
        kwargs["sort_structure"] = False
        MPRelaxSet.__init__(self, structures[0], **kwargs)
        self.structures = self._process_structures(structures)
        self.unset_encut = False
        if unset_encut:
            self.config_dict["INCAR"].pop("ENCUT", None)

        if "EDIFF" not in self.config_dict["INCAR"]:
            self.config_dict["INCAR"]["EDIFF"] = self.config_dict[
                "INCAR"].pop("EDIFF_PER_ATOM")

        # NEB specific defaults
        defaults = {'IMAGES': len(structures) - 2, 'IBRION': 1, 'ISYM': 0,
                    'LCHARG': False, "LDAU": False}
        self.config_dict["INCAR"].update(defaults)