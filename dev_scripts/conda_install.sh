#!/bin/bash
set -e  # exit on first error

echo "Installing AbiPy dependencies with conda."
echo "Adding conda-forge, matsci and abinit to channels"
echo "Working in CONDA_PREFIX: ${CONDA_PREFIX} ..."
conda config --add channels conda-forge
conda config --add channels abinit

conda install -c abinit apscheduler==2.1.0

echo "Installing requirements listed requirements.txt and requirements-optional.txt ..."
# https://github.com/ContinuumIO/anaconda-issues/issues/542
conda install -y -c anaconda setuptools
#conda install -c blaze flask-mongoengine

# We are gonna use the github version from gmatteo
#sed -i '/pymatgen/d' requirements.txt
#sed -i '/abipy/d' requirements.txt
conda install -y --file ./requirements.txt
conda install -y --file ./requirements-optional.txt

#echo "Installing bader executable (http://theory.cm.utexas.edu/henkelman/code/bader/) from matsci ..."
#conda install -y -c matsci bader

# Install abinit from abinit conda channel.
conda install -y -c abinit abinit=${ABINIT_VERSION}
abinit --version
abinit --build

echo "Installation completed"
