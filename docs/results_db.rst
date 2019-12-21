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
mongoengine `Document` objects are implemented in the
:mod:`abiflows.database.mongoengine` module.
