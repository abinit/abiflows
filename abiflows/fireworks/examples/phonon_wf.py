from abipy.abilab import Structure
from abiflows.fireworks.workflows.abinit_workflows import DfptFWWorkflow
from abiflows.database.mongoengine.utils import DatabaseData
from abiflows.database.mongoengine.abinit_results import RelaxResult

# data for the database where the relaxed structures were stored
source_db = DatabaseData(host='database_address', port=27017, collection='collection_name',
                         database='database_name', username='username', password='password')

# data for the database where the phonon results will be stored.
# note that these can be in different databases or in the same.
# The collections should be different
db = DatabaseData(host='database_address', port=27017, collection='another_collection_name',
                  database='database_name', username='username', password='password')

# Open the connection to the database
source_db.connect_mongoengine()

# in case you are using multiple workers for the same fireworks db (i.e. different clusters or queues)
# it may be a good idea to set the worker explicitly. One can just get the name from the configuration:
# fworker = FWorker.from_file(os.path.join(os.getenv("HOME"), ".fireworks", "my_fworker.yaml"))
# or you can also just write the name of the fworker explicitely
fworker_name = 'name_of_the_fworker'

mp_id = 'mp-149'

# This context manager is required to use the collection name selected in source_db
# By default mongoengine uses the name of the class (in this case RelaxResult) as
# name of the collection to query.
with source_db.switch_collection(RelaxResult) as RelaxResult:
    # download from the database the relaxed structure
    # This relies on mongoengine (http://mongoengine.org/) to interact with the database.
    # See the module abiflows.database.mongoengine.abinit_results for the objects used to store the results
    relaxed_results = RelaxResult.objects(mp_id=mp_id)

    # Assume that there is one and only one result matching the query. In real cases you might want to check this.
    # At this point is an instance of a RelaxResult object
    relaxed = relaxed_results[0]

    # load the relaxed Structure
    structure = Structure.from_dict(relaxed.abinit_output.structure)
    # use the same k-point sampling as the one of the relax
    kppa = relaxed.abinit_input.kppa
    ngkpt = relaxed.abinit_input.ngkpt

    # The AbinitInput object used for the relax is stored in the database.
    # We get it to use the same approximations used during the relaxation.
    relax_input = relaxed.abinit_input.last_input.to_mgobj()

    # We use the same k and q point grid
    qppa = kppa
    extra_abivars = dict(chkprim=1, nstep=100, chksymbreak=1)
    # as for the relax workflow, information stored in the database for the calculation. In particular information
    # about the source structure.
    initialization_info = dict(kppa=kppa, mp_id=mp_id,
                               relax_db=source_db.as_dict_no_credentials(), relax_id=relaxed.id,
                               relax_tol_val=1e-6, qppa=qppa)

    # In this case the base is the input file of the of the relax workflow.
    # Use the DfptFWWorkflow that allow to calculate the different kind of Dfpt perturbations
    # with abinit in a single workflow. In this case only the phonons.
    gen = DfptFWWorkflow.from_gs_input(structure=structure, gs_input=relax_input, extra_abivars=extra_abivars, autoparal=True,
                                       initialization_info=initialization_info, do_ddk=True, do_dde=True, ph_ngqpt=[1,1,1],
                                       do_strain=False)

    # add to the workflow a step that automatically adds the results to the database in the collection specified above.
    gen.add_mongoengine_db_insertion(db)

    # add a step to the workflow that cleans up files with this extensions once the other calculations are completed.
    # The list of extensions is customizable and these are usually file that won't be needed again.
    # Here we do not delete the DDB files.
    gen.add_final_cleanup(["WFK", "1WF", "WFQ", "1POT", "1DEN"])

    # This will specify that all the steps will be forced to be executed on the same worker
    # and will set the worker to the one chosen before for the existing fireworks. This step is not mandatory.
    gen.fix_fworker(fworker_name)

    # adds the workflow to the fireworks database. It will use the fireworks LaunchPad that has been set by default.
    # If a different one should be used it can be passed as an argument.
    gen.add_to_db()
