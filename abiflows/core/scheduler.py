# coding: utf-8
"""The scheduler is the object responsible for the submission of the flows in the database."""
from __future__ import print_function, division, unicode_literals

import os
import six

import logging
logger = logging.getLogger(__name__)


#class Scheduler(six.with_metaclass(abc.ABCMeta)):
class Scheduler(object):

    def __init__(self, kwargs):
        """
        Args:
            max_njobs_inqueue: The launcher will stop submitting jobs when the
                    number of jobs in the queue is >= Max number of jobs
            max_cores: Maximum number of cores
        """
        self.max_njobs_inqueue = kwargs.pop("max_njobs_inqueue", 200)
        self.max_cores = kwargs.pop("max_cores", 1000)

    #def fix_qcritical(self):
    #def fix_abicritical(self):
