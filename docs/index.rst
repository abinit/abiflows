.. :Release: |version| :Date: |today|

.. htmlonly::
    :Release: |version|
    :Date: |today|


.. image:: https://badge.fury.io/py/abiflows.svg
    :target: https://badge.fury.io/py/abiflows

.. image:: https://travis-ci.org/abinit/abiflows.svg?branch=develop
    :target: https://travis-ci.org/abinit/abiflows

.. image:: https://coveralls.io/repos/github/abinit/abiflows/badge.svg?branch=develop
    :target: https://coveralls.io/github/abinit/abiflows?branch=develop

.. image:: https://img.shields.io/badge/license-GPL-blue.svg


AbiFlows is a framework for Python >= 3.6 that provides workflows to automate ab-initio calculations
with Abinit based on the `Fireworks <https://materialsproject.github.io/fireworks/>`_ framework.
The package relies on external libraries, mainly `Abipy <http://abinit.github.io/abipy/>`_ and
`Pymatgen <http://www.pymatgen.org>`_, to handle the input generation and parse the output files.
The package also offers a set of tools to store the results produced by the workflows in
MongoDB documents and to easily retrieve them.

Abiflows is mainly meant for running Abinit calculations in the high-throughput regime, when it
is important to rely on a database both for handling many calculations and to store the results.
While it provides a wide support for different types of  calculations, this necessarily comes at
the price of some limitations on the configurations that one can set for Abinit. If you are
interested in running just a small set of calculations, we suggest to consider the
lightweight implementation provided by the `Abipy workflows <http://abinit.github.io/abipy/flow_gallery/index.html>`_.

AbiFlows is free to use. However, we also welcome your help to improve this library by making your own contributions.
Please report any bugs and issues at AbiFlows `Github page <https://github.com/abinit/abiflows>`_.


User Guide
==========

.. toctree::
   :maxdepth: 1

   installation
   setup
   workflows
   results_db
   failures
   changelog

References
==========

.. toctree::
   :maxdepth: 1

   useful_links
   zzbiblio

Citing Abiflows
===============

If you use abiflows in your research, please consider citing the
`following work <https://doi.org/10.1016/j.cpc.2019.107042>`_::

    @article{key,
     author = {Gonze, Xavier and Amadon, Bernard and Antonius, Gabriel and Arnardi, Frédéric and Baguet, Lucas and Beuken, Jean-Michel and Bieder, Jordan and Bottin, François and Bouchet, Johann and Bousquet, Eric and Brouwer, Nils and Bruneval, Fabien and Brunin, Guillaume and Cavignac, Théo and Charraud, Jean-Baptiste and Chen, Wei and Côté, Michel and Cottenier, Stefaan and Denier, Jules and Geneste, Grégory and Ghosez, Philippe and Giantomassi, Matteo and Gillet, Yannick and Gingras, Olivier and Hamann, Donald R. and Hautier, Geoffroy and He, Xu and Helbig, Nicole and Holzwarth, Natalie and Jia, Yongchao and Jollet, François and Lafargue-Dit-Hauret, William and Lejaeghere, Kurt and Marques, Miguel A.L. and Martin, Alexandre and Martins, Cyril and Miranda, Henrique P.C. and Naccarato, Francesco and Persson, Kristin and Petretto, Guido and Planes, Valentin and Pouillon, Yann and Prokhorenko, Sergei and Ricci, Fabio and Rignanese, Gian-Marco and Romero, Aldo H. and Schmitt, Michael Marcus and Torrent, Marc and van Setten, Michiel J. and Van Troeye, Benoit and Verstraete, Matthieu J. and Zérah, Gilles and Zwanziger, Josef W.},
     doi = {10.1016/j.cpc.2019.107042},
     pages = {107042},
     source = {Crossref},
     url = {http://dx.doi.org/10.1016/j.cpc.2019.107042},
     volume = {248},
     journal = {Computer Physics Communications},
     publisher = {Elsevier BV},
     title = {The Abinitproject: {Impact,} environment and recent developments},
     issn = {0010-4655},
     year = {2020},
     month = mar,
    }

API
===

.. toctree::
   :maxdepth: 1

   api/modules.rst

Indices and tables
==================

  :ref:`genindex`
  :ref:`modindex`
  :ref:`search`

License
=======

AbiFlows is released under the GPL License. The terms of the license are as follows:

.. include:: ../LICENSE
