.. _setup:

=====
Setup
=====

The first step to be able to run workflows in abiflows is to properly install and configure
all the software and modules that it depends on. Several configuration files should be
prepared in this procedure, related to the different components that are required to run
the workflows. You can either put these files in their respective default paths or create
a single folder where to collect all these files and point the different modules to this
folder, as explained in the next sections. The second option should be favoured especially
if you are planning to run with different sets of configurations (e.g. different parallelization
configurations, different databases, ...). In the following we will refer to this generic
configuration folder as ``af_config``, but this is just an example and there is no reference
to this specific name in Abiflows.

Abinit
======

First of all you need a compiled version of `Abinit <https://www.abinit.org>`_.
Latest versions tend to have a larger support for the python code handling its
inputs and outputs so you are encouraged to use the most recent production version
available. In any case the version should not be older than version 8.4.

When running on a cluster you will likely need to load modules relative or set some
variables when running the DFT code. These configurations should be available
at run time. If these are not automatically loaded (e.g. they are not in your
``.bashrc`` file), you will need to set it up from the fireworks configuration files.

Abipy
=====

`Abipy <http://abinit.github.io/abipy/>`_ is used in abiflows to handle the input and outputs
of Abinit, but is also a key tool to execute the small executables belonging to the Abinit suite,
like ``mrgddb`` or ``anaddb``. In addition it provides the base for the automatic determination
of the optimal parallelization configuration for the calculations. In order to do this
Abipy needs to be properly configured. In particular this means that you need to prepare
a ``manager.yml`` file according to the instructions given in the
`Abipy user guide <http://abinit.github.io/abipy/workflows/taskmanager.html>`_.
If you are a standard Abipy user you can probably keep you ``manager.yml`` file as it
is. Its default location is ``$HOME/.abinit/abipy``, as in Abipy, but otherwise you can
specify a different one for Abiflows, as explained below.

Fireworks
=========

The next step is to configure Fireworks and its database.
The main source of information for setting up Fireworks is its official documentation:

* `<https://materialsproject.github.io/fireworks/installation.html>`_
* `<https://materialsproject.github.io/fireworks/config_tutorial.html>`_
* `<https://materialsproject.github.io/fireworks/worker_tutorial.html>`_

However another useful reference is the atomate installation instructions:

* `<https://atomate.org/installation.html#configure-database-connections-and-computing-center-parameters>`_

Here we provide a quick summary of the main steps that you need to perform and the specific
configurations that you might want to set for Abiflows.

Several files should be created to properly run fireworks. These are:

* ``FW_config.yaml``: contains general configurations and paths to the following files.
* ``my_launchpad.yaml``: details of the connection to the DB.
* ``my_fworker.yaml``: the definition of the ``FireWorker``.
* ``my_qadapter.yaml``: the definition of the parameters for the queue.

By default fireworks will look for these configuration files in the ``~/.fireworks``
folder. In case you opt for using the specific configuration folder you can put all these
files your ``af_config`` folder This should be explicitly indicated when using fireworks scripts through
the command line options (``-c``, ``-l``, ``-q``, ``-w``) or setting the environmental variable:

.. code-block:: bash

    export FW_CONFIG_FILE=/path/to/base_config/FW_config.yaml

When needing a new set of configurations create a copy of ``af_config`` and modify the
required parameters and all the paths to the new folder.

.. note::

    It is important that the files are properly filled with paths to the actual configuration
    folder, so for the ``rocket_launch`` key in ``my_qadapter.yaml`` you will need to set
    the full path to ``af_config`` (or whichever name is used for the folder)

    .. code-block:: yaml

        rocket_launch: rlaunch -c /path/to/af_config rapidfire

After setting up the fireworks files check that the connection with the database is working
properly, running one of the ``lpad`` commands, for example

.. code-block:: bash

    lpad get_fws -d count

If you everything is configured correctly you should get the number of fireworks in the database
(``0`` for an empty database). At this point you should initialize your database running

.. code-block:: bash

    lpad reset

Runtime configurations
----------------------

When needing to set some specific option in the job that will be executed you need to set this
in the ``my_qadapter.yaml`` file, and in particular in the ``pre_rocket`` keyword. This should
be a line of commands that will be executed before running the ``rlaunch`` command in the
submission script. You should consider adding here everything that will be needed to execute
the python code and to correctly run abinit. This is (a partial) list of things that might want
to use the ``pre_rocket`` for:

* loading the python environment.
* loading the cluster modules needed to run Abinit.
* calling some configuration script to set up the parallelization environment.
* add the Abinit bin folder to the ``PATH``.
* set the ``FW_CONFIG_FILE`` environmental variable.
* set the ``FW_TASK_MANAGER`` environmental variable (see below).

Fireworks offline mode
----------------------

Note that sometimes it might be convenient (or even necessary) to run the jobs on the cluster
nodes in what is called *offline mode* in `Fireworks <https://materialsproject.github.io/fireworks/offline_tutorial.html>`_.
Not all the operations implemented in abiflows are compatible with this mode though.
In particular the insertion in the database and the final cleanup of temporary files requires
a connection to the Fireworks database to be executed. If you want/need to run in offline mode
you have to make sure that the Fireworks containing these operations will run in standard mode,
for example on the front-end of the cluster. This can be done by setting up two different
workers (with their full set of configuration files, if needed). More details about how to do
this can be found in the `Fireworks documentation <https://materialsproject.github.io/fireworks/controlworker.html#controlling-the-worker-that-executes-a-firework>`_.

Abiflows
========

Abiflows has only one configuration file, whose default name is ``fw_manager.yaml`` and whose
default location is in the ``$HOME/.abinit/abipy`` folder. The easiest way to point to a different
one is to set the ``FW_TASK_MANAGER`` environmental variable with the full path to the file.
With the example of the ``fw_config`` folder you should run:

.. code-block:: bash

    export FW_TASK_MANAGER=/path/to/base_config/fw_manager.yaml

The ``fw_manager.yaml`` is a yaml file where you can set some configurations for the execution of
the workflow. For backward compatibility reasons the file should be structured with a main keyword,
`fw_policy`, containing the different options that you might want to customize, i.e.:

    .. code-block:: yaml

        fw_policy:
            abinit_cmd: abinit
            mrgddb_cmd: mrgddb

Note that in some cases the values of these options can be overwritten by setting the value in the
``spec`` of a specific Firework.

.. _setup_fw_manager_opt:

fw.manager.yaml options
-----------------------

This is the list of options that can be currently set:

**abipy_manager**: (default: ``None``) the full path to the ``manager.yml`` file that contains the information
of the Abipy task manager.

**max_restarts**: (default: ``10``) the maximum number of restarts allowed for fixing errors or for
not converged calculations.

**autoparal**: (default: ``False``) whether to use the autoparal or not, if not explicitly set in the
workflow at creation time.

**abinit_cmd**: (default: ``abinit``) the full path to the abinit executable. If only ``abinit`` is
used it is expected to be in the PATH.

**mrgddb_cmd**: (default: ``mrgddb``) the full path to the mrgddb executable. If only ``mrgddb`` is
used it is expected to be in the PATH.

**anaddb_cmd**: (default: ``anaddb``) the full path to the anaddb executable. If only ``anaddb`` is
used it is expected to be in the PATH.

**cut3d_cmd**: (default: ``cut3d``) the full path to the cut3d executable. If only ``cut3d`` is
used it is expected to be in the PATH.

**mpirun_cmd**: (default: ``mpirun``) the command to be used for mpi parallelization.

**short_job_timelimit**: (default: ``600``) the number of seconds used for generating the job in the queue
for a *short* firework, e.g. the insertion in the database, the cleaunp, running ``mrgddb``.

**recover_previous_job**: (default: ``True``) if True before running a Firework it will try to check if
there is already a completed calculation in the folder and not execute it. This is useful when trying
to recover a calculation that completed successfully but was not registered correctly (e.g. the DB
was offline or a connection problem happened at the moment of completing the Firework).

**walltime_command**: (default: ``None``) a string containing a command that will return the remaining
number of seconds in the queue job. Passed to the ``--timelimit`` in Abinit. If ``None`` or the command
fails not time limit is set.

**timelimit_buffer**: (default: ``120``) number of seconds given as additional buffer for the time limit with
respect to what is extracted from ``walltime_command``.

**continue_unconverged_on_rerun**: (default: True) if a job did not converge within the number of restarts
specified in ``max_restarts`` the job ends in a ``FIZZLED`` state. If ``continue_unconverged_on_rerun`` is
set to ``True``, when rerunning that Firework the calculation will start from the final configuration of
the previous execution and will not start from scratch.

**allow_local_restart**: (default: ``False``) if True instead of creating a detour when fixing an error or
restarting an unconverged calculation it will continue in the same job.

**rerun_same_dir**: (default: ``False``) if True, when a calculation did not converge or an error is
fixed, the new Firework created to run will be launched in the same folder as the current one
(similarly to what is done in Abipy).

**copy_deps**: (default: ``False``) if True the abinit output files from previous steps, that are required
for the current step, will be copied instead of being linked.

Pseudo dojo
===========

While there is no constraint on the pseudopotentials that can be used and they can be just
given as list of strings with the paths to the pseudopotentials for each workflow, the best
solution when running high-throughput calculations is to rely on table of pseudopotentials with hints
for the values of the cutoff. In our case abipy provides full support for the pseudopotentials
tables available in the the `pseudo dojo <http://www.pseudo-dojo.org/>`_. Even though this is not a
strict dependency you are thus also encouraged to install the ``pseudo_dojo`` module. For more details see the
`pseudo dojo github page <https://github.com/abinit/pseudo_dojo/>`_.
