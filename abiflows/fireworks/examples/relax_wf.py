from fireworks import FWorker
import os
import pseudo_dojo
from abiflows.fireworks.workflows.abinit_workflows import RelaxFWWorkflow
from abiflows.database.mongoengine.utils import DatabaseData
from pymatgen import MPRester

# use the pseudo dojo table of pseudopotentials. These are good pseudopotentials. If you want to use
# some other kind of pseudos you will need to provide explicitly the cutoff for the calculation
pseudo_table = pseudo_dojo.OfficialDojoTable.from_djson_file(
    os.path.join(pseudo_dojo.dojotable_absdir("ONCVPSP-PBE-PDv0.4"), 'standard.djson'))
pseudo_path = pseudo_dojo.dojotable_absdir("ONCVPSP-PBE-PDv0.4")

# connection data of the output MongoDB database
# it can be the same database used for fireworks with other collections or a different one
db = DatabaseData(host='database_address', port=27017, collection='collection_name',
                  database='database_name', username='username', password='password')

# in case you are using multiple workers for the same fireworks db (i.e. different clusters or queues) it may be a good idea
# setting the worker explicitly. Here I just get the name
fworker = FWorker.from_file(os.path.join(os.getenv("HOME"), ".fireworks", "my_fworker.yaml"))

# Get the structure from the Materials Project. mp-149 is silicon.
mp_id = 'mp-149'
structure = MPRester().get_structure_by_material_id(mp_id)

# check if the pseudo is available and just selects those that are needed for the specific structure
try:
    pseudos = pseudo_table.get_pseudos_for_structure(structure)
except BaseException as e:
    print("no pseudo")
    exit(1)

# density of k-points per reciprocal atom. set to 1500 for phonons.
kppa = 1500

# this will be read at the end of the workflow to store this information in the database. It is not mandatory
initialization_info = dict(kppa=kppa, mp_id=mp_id)

# use a more strict tolmxf in case this might be needed, for example for phonon calculations.
tolmxf = 1e-6

#override some default parameters from the factory function
extra_abivars = dict(tolmxf=tolmxf, ionmov=2, chksymbreak=1, ntime=30, nstep=100)
# uncomment this if you want to try paral_kgb=1
#extra_abivars['paral_kgb'] = 1


# this will create a fireworks workflow object (still not added to fireworks database)
# check the function for the different options available.
# The OneSymmetric option will set a single shift that respects the symmetry of the crystal.
# The target_dilatmx means that the dilatmx parameter will be automatically progressively
# reduced and relaxation restarted until the desired value has been used.
gen = RelaxFWWorkflow.from_factory(structure, pseudo_table, kppa=kppa, spin_mode="unpolarized", extra_abivars=extra_abivars,
                                   autoparal=True, initialization_info=initialization_info, target_dilatmx=1.01,
                                   smearing=None, shift_mode='OneSymmetric', ecut=5)

# add to the workflow a step that automatically adds the results to the database in the collection specified above.
gen.add_mongoengine_db_insertion(db)

# add a step to the workflow that cleans up files with this extensions once the other calculations are completed.
# The list of extensions is customizable and these are usually file that won't be needed again
gen.add_final_cleanup(["WFK", "1WF", "DEN", "WFQ", "DDB"])

# This will specify that all the steps will be forced to be executed on the same worker
# and will set the worker to the one chosen before for the existing fireworks. This step is not mandatory.
gen.fix_fworker(fworker.name)

# adds the workflow to the fireworks database. It will use the fireworks LaunchPad that has been set by default.
# If a different one should be used it can be passed as an argument.
fw_id_maps = gen.add_to_db()

print("{} submitted".format(mp_id))
