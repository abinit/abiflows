from __future__ import print_function, division, unicode_literals, absolute_import

import numpy as np
import os


from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.io.vasp.outputs import Poscar
from pymatgen.core.structure import Structure
from pymatgen.core.sites import PeriodicSite
from itertools import chain


class MPNEBSet(MPRelaxSet):

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

    @property
    def poscar(self):
        return Poscar(self.structures[0])

    @property
    def poscars(self):
        return [Poscar(s) for s in self.structures]

    def _process_structures(self, structures):
        """
        Remove any atom jumps across the cell
        """
        input_structures = structures
        structures = [input_structures[0]]
        for s in input_structures[1:]:
            prev = structures[-1]
            for i in range(len(s)):
                t = np.round(prev[i].frac_coords - s[i].frac_coords)
                if np.sum(t) > 0.5:
                    s.translate_sites([i], t, to_unit_cell=False)
            structures.append(s)
        return structures

    def write_input(self, output_dir, make_dir_if_not_present=True,
                    write_cif=False, write_path_cif=False,
                    write_endpoint_inputs=False):
        """
        NEB inputs has a special directory structure where inputs are in 00,
        01, 02, ....

        Args:
            output_dir (str): Directory to output the VASP input files
            make_dir_if_not_present (bool): Set to True if you want the
                directory (and the whole path) to be created if it is not
                present.
            write_cif (bool): If true, writes a cif along with each POSCAR.
            write_path_cif (bool): If true, writes a cif for each image.
            write_endpoint_inputs (bool): If true, writes input files for
                running endpoint calculations.
        """

        if make_dir_if_not_present and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.incar.write_file(os.path.join(output_dir, 'INCAR'))
        self.kpoints.write_file(os.path.join(output_dir, 'KPOINTS'))
        self.potcar.write_file(os.path.join(output_dir, 'POTCAR'))

        for i, p in enumerate(self.poscars):
            d = os.path.join(output_dir, str(i).zfill(2))
            if not os.path.exists(d):
                os.makedirs(d)
            p.write_file(os.path.join(d, 'POSCAR'))
            if write_cif:
                p.structure.to(filename=os.path.join(d, '{}.cif'.format(i)))
        if write_endpoint_inputs:
            end_point_param = MPRelaxSet(
                self.structures[0],
                user_incar_settings=self.user_incar_settings)

            for image in ['00', str(len(self.structures) - 1).zfill(2)]:
                end_point_param.incar.write_file(os.path.join(output_dir, image, 'INCAR'))
                end_point_param.kpoints.write_file(os.path.join(output_dir, image, 'KPOINTS'))
                end_point_param.potcar.write_file(os.path.join(output_dir, image, 'POTCAR'))
        if write_path_cif:
            sites = set()
            l = self.structures[0].lattice
            for site in chain(*(s.sites for s in self.structures)):
                sites.add(PeriodicSite(site.species_and_occu, site.frac_coords, l))
            path = Structure.from_sites(sorted(sites))
            path.to(filename=os.path.join(output_dir, 'path.cif'))


class MPcNEBSet(MPNEBSet):
    def __init__(self, structures, unset_encut=False, **kwargs):
        super(MPcNEBSet, self).__init__(structures, unset_encut=False, **kwargs)
        # cNEB specific defaults
        defaults = {'LCLIMB': True}
        self.config_dict["INCAR"].update(defaults)