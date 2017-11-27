# coding: utf-8
"""
Common test support for abiflows test scripts.

This single module should provide all the common functionality for abiflows tests
in a single location, so that test scripts can just import it and work right away.
This module heavily depends on the abipy.testing module/
"""
from __future__ import print_function, division, unicode_literals, absolute_import

import os
import numpy
import subprocess
import json
import tempfile
import shutil
import glob
import unittest
import numpy.testing.utils as nptu

from monty.os.path import which
from monty.string import is_string

from abipy.core.testing import AbipyTest

from fireworks.core.launchpad import LaunchPad
from fireworks.core.fworker import FWorker

import logging
logger = logging.getLogger(__file__)

root = os.path.dirname(__file__)

__all__ = [
    "AbiflowsTest"
]


TESTDB_NAME = "abiflows_unittest"


def has_mongodb(host='localhost', port=27017, name='mongodb_test', username=None, password=None):
   try:
       from pymongo import MongoClient
       connection = MongoClient(host, port, j=True)
       db = connection[name]
       if username:
           db.authenticate(username, password)

       return True
   except:
       return False


def has_fireworks():
   """True if fireworks is installed."""
   try:
       import fireworks
       return True
   except ImportError:
       return False


class AbiflowsTest(AbipyTest):
    """Extends AbipyTest with methods specific for the testing of workflows"""

    def assertFwSerializable(self, obj):
       assert '_fw_name' in obj.to_dict()
       self.assertDictEqual(obj.to_dict(), obj.__class__.from_dict(obj.to_dict()).to_dict())

    @classmethod
    def setup_fireworks(cls):
        """
        Sets up the fworker and launchpad if a connection to a local mongodb is available.
        cls.lp is set to None if not available
        """

        cls.fworker = FWorker()
        try:
            cls.lp = LaunchPad(name=TESTDB_NAME, strm_lvl='ERROR')
            cls.lp.reset(password=None, require_password=False)
        except:
            cls.lp = None

    @classmethod
    def teardown_fireworks(cls, module_dir=None):
        """
        Removes the fireworks test database if cls.lp is present and deletes all the launcher directories
        """
        if cls.lp:
            cls.lp.connection.drop_database(TESTDB_NAME)

        if module_dir:
            for ldir in glob.glob(os.path.join(module_dir,"launcher_*")):
                shutil.rmtree(ldir)
