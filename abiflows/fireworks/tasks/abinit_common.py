from __future__ import print_function, division, unicode_literals, absolute_import

import json
import os

from math import ceil
from monty.dev import deprecated
import abipy.abio.inputs as abinit_inputs

__author__ = 'waroquiers'

TMPDIR_NAME = "tmpdata"
OUTDIR_NAME = "outdata"
INDIR_NAME = "indata"
STDERR_FILE_NAME = "run.err"
LOG_FILE_NAME = "run.log"
FILES_FILE_NAME = "run.files"
OUTPUT_FILE_NAME = "run.abo"
OUTNC_FILE_NAME = "out_OUT.nc"
INPUT_FILE_NAME = "run.abi"
MPIABORTFILE = "__ABI_MPIABORTFILE__"
DUMMY_FILENAME = "__DUMMY__"
ELPHON_OUTPUT_FILE_NAME = "run.abo_elphon"
DDK_FILES_FILE_NAME = "ddk.files"
HISTORY_JSON = "history.json"


module_dir = os.path.dirname(os.path.abspath(__file__))


@deprecated(abinit_inputs.Cut3DInput, "Switch to the new version defined in abipy")
class Cut3DInput(object):

    def __init__(self, cut3d_input):
        self.cut3d_input = cut3d_input

    def write_input(self, filename):
        with open(filename, 'w') as f:
            f.write('\n'.join(self.cut3d_input))

    @classmethod
    def den_to_cube(cls, density_filepath, cube_filename):
        lines = [density_filepath]          # Path to the _DEN file
        # input.append('1')                   # _DEN file is a binary file. Not needed after abinit 8
        # input.append('0')                   # Use the total density. Not needed
        lines.append('14')                  # Option to convert _DEN file to a .cube file
        lines.append(cube_filename)         # Name of the output .cube file
        lines.append('0')                   # No more analysis
        return cls(cut3d_input=lines)

    @classmethod
    def hirshfeld(cls, density_filepath, all_el_dens_paths):
        lines = [density_filepath]  # Path to the _DEN file
        lines.append('11')  # Option to convert _DEN file to a .cube file
        for p in all_el_dens_paths:
            lines.append(p)

        return cls(lines)

    @classmethod
    def hirshfeld_from_fhi_path(cls, density_file_path, structure, fhi_all_el_path):
        all_el_dens_paths = []
        for e in structure.composition.elements:
            all_el_dens_paths.append(os.path.join(fhi_all_el_path), "0.{}-{}.8.density.AE")

        return cls.hirshfeld(density_file_path, all_el_dens_paths)

def unprime_nband(nband, number_of_primes=10):
    allowed_nbands = []
    with open('{}/n1000multiples_primes.json'.format(module_dir), 'r') as f:
        dd = json.load(f)
        if 'numbers{:d}primes'.format(number_of_primes) not in dd:
            raise ValueError('Number of primes is wrong ...')
        allowed_nbands = dd['numbers{:d}primes'.format(number_of_primes)]
    if nband <= 1000:
        if nband in allowed_nbands:
            return nband
        return min([larger_nband for larger_nband in allowed_nbands if larger_nband > nband])
    elif nband <= 10000:
        nband10 = int(ceil(float(nband)/10.0))
        if nband10 in allowed_nbands:
            return nband10*10
        return 10*min([larger_nband for larger_nband in allowed_nbands if larger_nband > nband10])