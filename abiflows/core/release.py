# coding: utf-8
"""Release data for the abiflows project."""
from __future__ import print_function, division, unicode_literals, absolute_import

from collections import OrderedDict

# Name of the package for release purposes.  This is the name which labels
# the tarballs and RPMs made by distutils, so it's best to lowercase it.
name = 'abiflows'

# version information.  An empty _version_extra corresponds to a full
# release.  'dev' as a _version_extra string means this is a development version
_version_major = 0
_version_minor = 4
_version_micro = ''  # use '' for first of series, number for 1 and above
#_version_extra = 'dev'
_version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro: _ver.append(_version_micro)
if _version_extra: _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

version = __version__  # backwards compatibility name

# The minimum Abinit version compatible with AbiFlows
#min_abinit_version = "8.0.8"

description = "Framework for high-throughput calculations with ABINIT"

long_description = \
    """
    The latest development version is always available from site <https://github.com/abinit/abiflows>
    """

license = 'GPL'

author = 'The Abinit group'
author_email = 'matteo.giantomassi@uclouvain.be'
maintainer = "Matteo Giantomassi"
maintainer_email = author_email
authors = OrderedDict([
    ('Guido', ('G. Petretto', 'nobody@nowhere')),
    ('David', ('D. Waroquiers', 'nobody@nowhere')),
    ('Matteo', ('M. Giantomassi', 'nobody@nowhere')),
    ('Michiel', ('M. J. van Setten', 'nobody@nowhere')),
])

url = "https://github.com/abinit/abiflows"
download_url = "https://github.com/abinit/abiflows"
platforms = ['Linux', 'darwin']
keywords = ["ABINIT", "ab-initio", "density-function-theory", "first-principles", "electronic-structure", "pymatgen"]
classifiers=[
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: Chemistry",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
