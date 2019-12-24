.. _results_db:

================
Results database
================

Abiflows provides an automated way to store the output of the workflow in a
MongoDB database. The default approach is that of storing the output of the
whole workflow in a single document of a MongoDB collection. Each type of
workflow produces one specific output, storing the most relevant information
produced by that workflow. In the following we describe the infrastructure
used to store and retrieve the output documents.

Workflow output
===============

The approach in the storage of the results is that in the MongoDB document only
data that are expected to be used are query parameters are included. A few examples
of what this includes are the composition and symmetry of the structure and the main
configuration options of the calculation. Other values that can be found in the
document are some basic output values that are likely to be accessed more frequently
or to be used as a filtering as well, like the final magnetization of the system or
the statistics of execution of the calculations.

On the other hand the fully detailed values of the results rarely needs to be queried
on. In the Abiflows output these values are kept in the native machine readable format
produced directly by Abinit, i.e. in NetCDF files. These files contain the full set of
data produced by the calculations and Abipy offers a full support to analyse and plot
their content. These files are stored in MongoDB GridFS as they are and referenced in the
main document. Usually some files in text format are also stored, both for necessity,
when the mainly supported version of an output file is a text one (e.g. the DDB file),
and for completeness. These files usually cover all the relevant outputs produced
by the workflow.

Mongoengine documents
=====================

In order to provide a common base for the standard keywords that are stored in the
database and for the structure of the documents produced by the different workflows,
the interaction with the output database in Abiflows is handled using
`mongoengine <http://mongoengine.org/>`_. For this purpose a set of
mongoengine ``Document`` objects are implemented in the
:mod:`abiflows.database.mongoengine.abinit_results` module.

From a technical point of view, the results documents are obtained composing different
mixins, each of which define a specific set of properties that are likely to be queried
and that are usually shared by different kinds of workflows (e.g. the
:class:`abiflows.database.mongoengine.mixins.MaterialMixin`). In this way the uniformity
of the notation for common keywords is guaranteed across the different kind of outputs.

The queries to the documents can be done using the standard mongoengine approach:

.. code-block:: python

    for result in RelaxResult.objects(nsites=5):
        print(result.pretty_formula)

where it is possible to query the properties in the ``Document`` object. See the
`mongoengine user guide <http://docs.mongoengine.org/guide/index.html>`_ for more
details.

Database connection definition
==============================

The information required to connect to the database (e.g. address, username, ...)
are stored in a specific object :class:`abiflows.database.mongoengine.utils.DatabaseData`
so that it can be used to define where to store the results and to access, passing
it to the task responsible to generate the document with the output of the workflow.

In addition, since mongoengine uses the name of the class as default name for the collection
where to store and retrieve the data, ``DatabaseData`` offers a shortcut for the
`switch_collection <http://docs.mongoengine.org/guide/connecting.html#context-managers>`_
method. You can thus use it to query the database, as shown in the :ref:`examples_wf_phonons`
example:

.. code-block:: python

    db = DatabaseData(host='db_address', port=27017, collection='collection_name',
                      database='db_name', username='user', password='pass')


    with source_db.switch_collection(RelaxResult) as RelaxResult:
        for result in RelaxResult.objects(nsites=5):
            print(result.pretty_formula)


.. note::

    Even though it might be convenient to rely on the mongoengine documents to interact
    with the results produced by Abiflows, this is by no means a strict requirement
    and the database can be queried using the standard MongoDB connections and queries.
