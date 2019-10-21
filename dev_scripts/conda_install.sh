#!/bin/bash
set -e  # exit on first error

conda install -c abinit apscheduler==2.1.0

echo "Installing requirements listed requirements.txt and requirements-optional.txt ..."
# https://github.com/ContinuumIO/anaconda-issues/issues/542
conda install -y -c anaconda setuptools

# We are gonna use the github version from gmatteo
#sed -i '/pymatgen/d' requirements.txt
#sed -i '/abipy/d' requirements.txt
conda install -y --file ./requirements.txt
conda install -y --file ./requirements-optional.txt

# Install abinit from abinit conda channel.
#conda install -y -c abinit abinit=${ABINIT_VERSION}
#abinit --version
#abinit --build

echo "Installation completed"
