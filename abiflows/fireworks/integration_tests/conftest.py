"""Configuration file for pytest."""
from __future__ import print_function, division, unicode_literals, absolute_import

import os
import pytest
import abipy.abilab as abilab
import abipy.data as abidata

from pymongo import MongoClient
from abipy.abio.factories import ebands_input, ion_ioncell_relax_input, scf_for_phonons, phonons_from_gsinput
from abipy.data.benchmark_structures import simple_semiconductors, simple_metals
from abiflows.database.mongoengine.utils import DatabaseData
from fireworks import LaunchPad, FWorker
try:
    from pymatgen.ext.matproj import MPRester
except ImportError:
    from pymatgen.matproj.rest import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


TESTDB_NAME = 'abiflows_unittest'
TESTCOLLECTION_NAME = 'test_results'
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def lp(request):
    lp = LaunchPad(name=TESTDB_NAME, strm_lvl='ERROR')
    lp.reset(password=None, require_password=False)

    def fin():
        lp.connection.drop_database(TESTDB_NAME)

    # request.addfinalizer(fin)
    return lp


@pytest.fixture(scope="module")
def fworker():
    return FWorker()


@pytest.fixture(scope="function")
def cleandb(request, lp):
    def fin():
        lp.reset(password=None, require_password=False)
    request.addfinalizer(fin)


@pytest.fixture(params=[False, True])
def use_autoparal(request):
    """
    This fixture allows to run some of the test both with autoparal True and False
    """
    return request.param


def get_si_structure():
    cif_file = abidata.cif_file("si.cif")
    structure = abilab.Structure.from_file(cif_file)

    return structure

def get_gan_structure():
    cif_file = abidata.cif_file("gan.cif")
    structure = abilab.Structure.from_file(cif_file)

    return structure


@pytest.fixture(scope="function")
def input_scf_si_low():
    pseudos = abidata.pseudos("14si.pspnc")
    structure = get_si_structure()

    return ebands_input(structure, pseudos, kppa=100, ecut=6, spin_mode="unpolarized",
                        accuracy="low").split_datasets()[0]

@pytest.fixture(scope="function")
def input_ebands_si_low():
    pseudos = abidata.pseudos("14si.pspnc")
    structure = get_si_structure()

    return ebands_input(structure, pseudos, kppa=100, ecut=6, ndivsm=3, spin_mode="unpolarized",
                        accuracy="low").split_datasets()


@pytest.fixture(scope="function")
def inputs_relax_si_low():
    pseudos = abidata.pseudos("14si.pspnc")
    structure = get_si_structure()

    # negative strain. This will allow to trigger dilatmx error
    structure.apply_strain(-0.005)
    structure.translate_sites(0, [0.001, -0.003, 0.005])
    structure.translate_sites(1, [0.007, 0.006, -0.005])
    return ion_ioncell_relax_input(structure, pseudos, kppa=100, ecut=4, spin_mode="unpolarized",
                                   accuracy="low", smearing=None).split_datasets()


@pytest.fixture(scope="function")
def input_scf_phonon_si_low():
    pseudos = abidata.pseudos("14si.pspnc")
    structure = get_si_structure()

    scf_in = scf_for_phonons(structure, pseudos, kppa=100, ecut=4, spin_mode="unpolarized", accuracy="low",
                             smearing=None)

    return  scf_in

@pytest.fixture(scope="function")
def input_scf_phonon_gan_low():
    pseudos = [abidata.pseudos("31ga.pspnc").pseudo_with_symbol('Ga'),
               abidata.pseudos("7n.pspnc").pseudo_with_symbol('N')]
    structure = get_gan_structure()

    scf_in = scf_for_phonons(structure, pseudos, kppa=100, ecut=4, spin_mode="unpolarized", accuracy="low",
                             smearing=None)

    return  scf_in

@pytest.fixture(scope="function")
def db_data():
    """
    Creates an instance of DatabaseData with TESTDB_NAME and TESTCOLLECTION_NAME.
    Drops the collection.
    """
    db_data = DatabaseData(TESTDB_NAME, collection=TESTCOLLECTION_NAME)
    connection = db_data.connect_mongoengine()
    connection[TESTDB_NAME].drop_collection(TESTCOLLECTION_NAME)

    return db_data


@pytest.fixture(scope="function", params=simple_semiconductors)
def benchmark_input_scf(request):
    pseudos = abidata.pseudos("14si.pspnc", "6c.pspnc", "3li.pspnc", "9f.pspnc",
                              "12mg.pspnc", "8o.pspnc", "31ga.pspnc", "7n.pspnc")
    rest = MPRester()
    structure = rest.get_structure_by_material_id(request.param)
    try:
        return ebands_input(structure, pseudos, kppa=100, ecut=6).split_datasets()[0]
    except:
        #to deal with missing pseudos
        pytest.skip('Cannot create input for material {}.'.format(request.param))


@pytest.fixture()
def fwp(tmpdir):
    """
    Parameters used to initialize Flows.
    """
    # Temporary working directory
    fwp.workdir = str(tmpdir)

    # Create the TaskManager.
    fwp.manager = abilab.TaskManager.from_file(os.path.join(os.path.dirname(__file__), "manager.yml"))

    fwp.scheduler = abilab.PyFlowScheduler.from_file(os.path.join(os.path.dirname(__file__), "scheduler.yml"))

    return fwp



