__author__ = 'waroquiers'

TMPDIR_NAME = "tmpdata"
OUTDIR_NAME = "outdata"
INDIR_NAME = "indata"
STDERR_FILE_NAME = "run.err"
LOG_FILE_NAME = "run.log"
FILES_FILE_NAME = "run.files"
OUTPUT_FILE_NAME = "run.abo"
INPUT_FILE_NAME = "run.abi"
MPIABORTFILE = "__ABI_MPIABORTFILE__"
DUMMY_FILENAME = "__DUMMY__"
ELPHON_OUTPUT_FILE_NAME = "run.abo_elphon"
DDK_FILES_FILE_NAME = "ddk.files"
HISTORY_JSON = "history.json"


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