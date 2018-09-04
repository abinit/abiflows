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

from mongoengine import connect, Document
from mongoengine.connection import get_db, get_connection
from monty.os.path import which
from monty.string import is_string
from abipy.core.testing import AbipyTest
from fireworks.core.launchpad import LaunchPad
from fireworks.core.fworker import FWorker
from fireworks.core.rocket_launcher import rapidfire
from abiflows.fireworks.utils.fw_utils import get_fw_by_task_index

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

    @classmethod
    def setup_mongoengine(cls):
        try:
            cls._connection = connect(db=TESTDB_NAME)
            cls._connection.drop_database(TESTDB_NAME)
            cls.db = get_db()
        except:
            cls.db = None
            cls._connection = None

    @classmethod
    def teardown_mongoengine(cls):
        if cls._connection:
            cls._connection.drop_database(TESTDB_NAME)


    def get_document_class_from_mixin(self, mixin_cls):
        """
        Utility function to generate a mongoengine Document class from the mixin.
        Needed to save the object in the db with mongoengine
        """

        class TestDocument(mixin_cls, Document):
            meta = {'collection': "test_{}".format(mixin_cls.__name__)}

        return TestDocument


class AbiflowsIntegrationTest(object):
    """
    Provides utility methods and variables for integration tests, that can't subclass unittest.TestCase
    """

    # variable to enable/disable the checks on the numerical quantities as output of the workflow
    check_numerical_values = True

    @staticmethod
    def assertArrayAlmostEqual(actual, desired, decimal=7, err_msg='',
                               verbose=True):
        """
        Tests if two arrays are almost equal to a tolerance. The CamelCase
        naming is so that it is consistent with standard unittest methods.
        """
        return nptu.assert_almost_equal(actual, desired, decimal, err_msg, verbose)


def check_restart_task_type(lp, fworker, tmpdir, fw_id, task_tag):

    # resume the task for tag
    wf = lp.get_wf_by_fw_id(fw_id)
    fw = get_fw_by_task_index(wf, task_tag, index=1)
    assert fw is not None
    assert fw.state == "PAUSED"
    lp.resume_fw(fw.fw_id)

    # run the FW
    rapidfire(lp, fworker, m_dir=str(tmpdir), nlaunches=1)

    # the job should have a detour for the restart
    wf = lp.get_wf_by_fw_id(fw_id)
    fw = get_fw_by_task_index(wf, task_tag, index=2)
    assert fw is not None
    assert fw.state == "READY"

    # run all the following and check that the last is correctly completed (if convergence is not achieved
    # the final state should be FIZZLED)
    rapidfire(lp, fworker, m_dir=str(tmpdir))

    wf = lp.get_wf_by_fw_id(fw_id)
    fw = get_fw_by_task_index(wf, task_tag, index=-1)

    assert fw.state == "COMPLETED"
