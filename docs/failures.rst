===================
Errors and failures
===================

A key point for being able to run workflows in an high throughput regime is to have
some automatic tool that deals with the most common errors that might show up during
an DFT calculation, and this is indeed present in Abiflows. Such a system, however,
cannot be expected to solve all the problems of various nature that may happen.
So you should consider that, when running fireworks workflows, you can encounter two
main kinds of failures. In the following we will distinguish the possible failures in
**soft and hard failures**. We will call a failure *soft* if a firework
is in the ``FIZZLED`` state. Conversely, we will identify a failure as *hard* if
the job was killed by some external action leaving the firework in a ``RUNNING``
state as a *lost run*. A third kind of failure that can happen is the case of
a *lost run* that is in an inconsistent state and needs to be adjusted.
Here we will provide a guide about how to deal with the different kinds of failures.

Before proceeding to the following section about error handling and dealing with failures
specific to Abiflows you might want to review the basic
`tutorial on failures <https://materialsproject.github.io/fireworks/failures_tutorial.html>`_
in the fireworks documentation.

Error Handling
==============

As mentioned above Abiflows can try to solve some errors that might be encountered during
a DFT calculation. The error messages produced by Abinit are encoded in a specific (YAML)
format inside the output file, so these can be easily parsed and identified, without relying
on different analysis of the textual messages.

The default approach is that whenever an error is encountered, and thus Abinit stops,
Abiflows will try to fix the problem and the default behaviour is to generate a *detour* that
will continue the calculation with the required modifications. This can altered setting the
``allow_local_restart`` keyword to ``True`` in the :ref:`setup_fw_manager_opt`. The creation
of a detour is usually preferable, since this may allow to accommodate for a change in the
parallelization options in the restart (if autoparal is used) and to reduce the chances of
hitting the maximum time required by a job on a cluster.

The system will not try to fix an unlimted number of errors (or the same errors for an
unlimited number of times). A maximum number of restarts for the same Firework is set
to 10 by default and can be customized with the ``max_restarts`` keyword in the
``fw_manager.yaml`` configuration file.

The error handling relies on the event handlers implemented in Abipy and shares the same
functionalities as the workflows implemented there. If you encounter an error that is not
handled and for which an error handler could be implemented, you can try to develop your
own handler and include it in Abipy or try to get in touch with the Abipy developers.

Soft failures
=============

When Abinit produces an error that is not handled, or the maximum number of restarts
is reached, the Firework will be end up in a ``FIZZLED`` state. Technically this means
that an exception was raised in the python code. This can thus come from Abiflows
recognizing that it cannot fix the error, but can also happen due to partial failure
of the system (e.g. a full disk will cause an I/O error but the system might still be
able to update the state of the fireworks database) or even to some bug in the python
code.

If the failure originates from a temporary problem of the system, like a problem of
one node that leads the calculation to crash, simply rerunning the Firework should solve
the issue. For this you can use the standard fireworks commands, e.g.

.. code-block:: bash

    lpad rerun_fws -i xxx

If the Firework fizzled because the maximum number of restarts has been reached you
can consider to inspect the calculation and decide if you want it to restart some more
times, if you evaluate that there is a possibility to reach the end of the calculation.
In this case you need to increase the number of restarts allowed in the ``fw_manager.yaml``
file and simply rerun the calculation.

However, if the problem arose from an Abinit error that cannot be fixed, rerunning the
Firework will simply lead to the same error. In this case you should probably consider
the calculation as lost and either create a new workflow with different initial
configurations or just discard the system. For an advanced user there might be an
additional option. If you think that you can fix the problem by adjusting some
abinit input variable you might consider updating the document in the ``fireworks``
collection of the fireworks database corresponding to the failed fireworks and modify
the data corresponding to the ``AbinitInput`` object.

.. note::
    If you decide to discard a calculation it might be convenient to delete
    the corresponding workflow from the fireworks database, so that it does
    not show up again when you look for ``FIZZLED`` fireworks. This can be
    achieved running:

    .. code-block:: bash

        lpad delete_wflows -i xxx

    Using the fw_id of the failed firework. This will just delete the workflow
    from the database, but it is also possible to delete all the related
    folders from the file system using the ``--ldirs`` option. This is usually
    the best solution

    .. code-block:: bash

        lpad delete_wflows -i xxx --ldirs

Hard failures
=============

When your job is killed by the queue manager, being it for some problem
of the cluster or because your calculation exceeded the resources available in the
job (e.g. memory or wall time) the Firework will remain in the ``RUNNING`` state.
First of all you need to identify these *lost runs*. This can be done with the standard
fireworks procedure with the command

.. code-block:: bash

    lpad detect_lostruns

This will provide a list of the ids of the *lost* fireworks (i.e. the fireworks
that did not *ping* the database for a specified amount of time). If you are confident
that all the lost jobs are due to a temporary problem or to a whim of the cluster you can
just rerun all the lost Fireworks:

.. code-block:: bash

    lpad detect_lostruns --rerun

If instead you suspect that there might be a problem in some of your jobs, the correct
way of proceeding will be to go to the launch directory of the job and inspect the
files produced by the queue manager. These usually contain information about the reason
of the failure and would probably explicitly mention if the error is coming from
an exceeded memory or wall time. If that's the case simply resubmitting your job will
probably end up with the same outcome and you might want to be sure that your job has enough
resources when you rerun it.

For this you might consider creating a specific fireworker with additional resources
and make the job run with that
`fireworker <https://materialsproject.github.io/fireworks/controlworker.html#controlling-the-worker-that-executes-a-firework>`_.
Alternatively, you might want to set or change the ``_queueadapter`` keyword in the spec of
the Firework, as is explained in the
``specific section of the fireworks manual <https://materialsproject.github.io/fireworks/queue_tutorial_pt2.html>`_.
If the *autoparal* is enabled the ``_queueadapter`` will already be present and you will
need to update the appropriate keywords to increase the resources that will be requested
to the queue manager.

Database issues
---------------

An additional problem that could leave your job as a lost run is when the database
becomes temporarily not available, due to a failure of the database server or to an
issue in the connection. In this case it might be that the calculation completed
successfully, but it could not update the results in the database. If you end up
in this situation the standard solution of simply rerunning the job is perfectly
viable, but you will lose the computational time used for the first run. A better
solution is to rerun the firework with the following command:

.. code-block:: bash

    lpad rerun_fws -i xxx --task-level --previous-dir

This will rerun the Firework and will make sure that it reruns in the same folder
as the first run. At this point Abiflows will notice that there is a completed
output in the folder and use that one instead of running Abinit again.

.. warning::

    Remember that there should be a completed Abinit output file in the folder.
    Obviously a partial output cannot be recovered in any way. In addition
    consider that the new launch of the Firework will only last a few seconds and
    this is the time that will be registered in the fireworks (and in the results)
    database. Consider this point if you plan to keep statistics about your
    computations run time.

Inconsistent Fireworks
======================

As mentioned above, there is one particular case for which your jobs might
be identified by the ``lpad detect_lostruns`` command, when there in an
inconsistency between the state of the Firework and the state of the Launch.
The output will look like this::

    2019-01-01 00:00:00,000 INFO Detected 0 lost FWs: []
    2019-01-01 00:00:00,000 INFO Detected 2 inconsistent FWs: [123,124]
    You can fix inconsistent FWs using the --refresh argument to the detect_lostruns command

This may happen when fireworks has problems in refreshing the whole state of
the Workflow and the dependencies of a Firework. A concrete example where this
can show up is when there are workflows with a large number of Fireworks that
all have a common child. This is the case for example for the ``mrgddb`` step
in a :ref:`dfpt_workflow` workflow.

This is only a small issue due to the particular configuration of the workflow
and does not require the job to be rerun. To solve this you simply need to run
the command:

.. code-block:: bash

    lpad detect_lostruns --refresh

Depending on the size of the workflow and on the number of inconsistent Fireworks,
this may take a while, but is not a computationally intensive operation.
