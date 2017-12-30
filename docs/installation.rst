================
Getting AbiFlows
================

.. contents::
   :backlinks: top

--------------
Stable version
--------------

The version at the Python Package Index (PyPI) is always the latest stable release
that can be installed with::

    $ pip install abiflows

Note that you may need to install pymatgen and other critical dependencies manually.
In this case, please consult the detailed installation instructions provided by the
`pymatgen howto <http://pymatgen.org/index.html#standard-install>`_ to install pymatgen.

The installation process is greatly simplified if you install the required 
python packages through one of the following python distributions:

  * `Anaconda <https://continuum.io/downloads>`_
  * `Canopy <https://www.enthought.com/products/canopy>`_

.. _developmental_version:

---------------------
Developmental version
---------------------

Getting the developmental version of AbiFlows is easy.
You can clone it from the  `github repository <https://github.com/abinit/abiflows>`_ using this command:

.. code-block:: console

   $ git clone https://github.com/abinit/abiflows

After cloning the repository, type::

    $ python setup.py install

or alternately::

    $ python setup.py develop

to install the package in developmental mode 
(Develop mode is the recommended approach if you are planning to implement new features.
In this case you may also opt to first fork AbiFlow on Git and then clone your own fork.
This will allow you to push any changes to you own fork and also get them merged in the main branch).

The documentation of the **developmental** version is hosted on `github pages <http://abinit.github.io/abiflow>`_.

The Github version include test files for complete unit testing.
To run the suite of unit tests, make sure you have ``pytest`` installed and then type::

    $ pytest

in the AbiFlows root directory.

Note that several unit tests check the integration between AbiPy and Abinit.
In order to run the tests, you need a working set of Abinit executables and  
a ``manager.yml`` configuration file.

A pre-compiled sequential version of Abinit for Linux and OSx can be installed directly from the anaconda cloud with::

    $ conda install abinit -c abinit

Contributing to AbiFlows is relatively easy.
Just send us a `pull request <https://help.github.com/articles/using-pull-requests/>`_.
When you send your request, make ``develop`` the destination branch on the repository
AbiFlows uses the `Git Flow <http://nvie.com/posts/a-successful-git-branching-model/>`_ branching model.
The ``develop`` branch contains the latest contributions, and ``master`` is always tagged and points
to the latest stable release.

If you choose to share your developments please take some time to develop some unit tests of at least the
basic functionalities of your code


---------------
Troubleshooting
---------------
