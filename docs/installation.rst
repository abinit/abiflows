================
Getting AbiFlows
================

.. contents::
   :backlinks: top

--------------
Stable version
--------------

The version at the `Python Package Index <https://pypi.python.org/pypi/abiflows>`_  (PyPI) is always 
the latest **stable** release that can be installed with::

    pip install abiflows

Note that you may need to install pymatgen_ and other critical dependencies manually.
In this case, please consult the detailed installation instructions provided in the
`pymatgen howto <http://pymatgen.org/index.html#standard-install>`_ to install pymatgen.

The installation process is greatly simplified if you install the required 
packages through the Anaconda_ distribution.

Download the anaconda installer from the `official web-site <https://www.continuum.io/downloads>`_.
and choose the version that matches your OS.

Run the bash script in the terminal and follow the instructions.
By default, the installer creates the ``anaconda`` directory in your home.
Anaconda will add one line to your ``.bashrc`` to enable access to the anaconda executables.
Once the installation is completed, execute::

    source ~/anaconda/bin/activate root

to activate the ``root`` environment.
The output of ``which python`` should show that you are using the python interpreter provided by anaconda.

Use the conda_ command-line interface to install the required packages.

Visit `materials.sh <http://materials.sh>`_ for instructions on how to use the
matsci channel to install pymatgen and other packages.


.. _developmental_version:

---------------------
Developmental version
---------------------

Getting the developmental version of AbiFlows is easy.
You can clone it from the `github repository <https://github.com/abinit/abiflows>`_ using::

    git clone https://github.com/abinit/abiflows

After cloning the repository, type::

    python setup.py install

or alternately::

    python setup.py develop

to install the package in developmental mode 
(Develop mode is the recommended approach if you are planning to implement new features.
In this case you may also opt to first fork AbiFlow on Git and then clone your own fork.
This will allow you to push any changes to you own fork and also get them merged in the main branch).

The documentation of the **developmental** version is hosted on `github pages <http://abinit.github.io/abiflow>`_.

The Github version include test files for complete unit testing.
To run the suite of unit tests, make sure you have pytest_ installed and issue::

    pytest

in the AbiFlows root directory.

Note that several unit tests check the integration between AbiPy and Abinit.
In order to run the tests, you need a working set of Abinit executables and  
a ``manager.yml`` configuration file.
For further information on the syntax of the configuration file, please consult the :ref:`taskmanager` section.

A pre-compiled sequential version of Abinit for Linux and OSx can be installed directly from the abinit-channel_ with::

    conda install abinit -c abinit

Examples of configuration files to configure and compile Abinit on clusters can be found 
in the abiconfig_ package.

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
