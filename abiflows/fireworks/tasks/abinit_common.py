__author__ = 'waroquiers'

import json
from math import ceil
import os

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


class Cut3DInput(object):

    def __init__(self, cut3d_input):
        self.cut3d_input = cut3d_input

    def write_input(self, filename):
        with open(filename, 'w') as f:
            f.write('\n'.join(self.cut3d_input))

    @classmethod
    def den_to_cube(cls, density_filepath, cube_filename):
        input = [density_filepath]          # Path to the _DEN file
        input.append('1')                   # _DEN file is a binary file
        input.append('0')                   # Use the total density
        input.append('14')                  # Option to convert _DEN file to a .cube file
        input.append(cube_filename)         # Name of the output .cube file
        input.append('0')                   # No more analysis
        return cls(cut3d_input=input)

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