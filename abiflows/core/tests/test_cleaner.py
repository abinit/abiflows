# coding: utf-8
from __future__ import print_function, division, unicode_literals, absolute_import

import os
import shutil

from abiflows.core.mastermind_abc import Cleaner
from abiflows.core.testing import AbiflowsTest


test_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                        "test_files")

class TestCleaner(AbiflowsTest):

    def test_cleaner(self):
        # Keep current working directory, create tmp directory and change to tmp directory
        cwd = os.getcwd()
        tmp_dir = 'tmp'
        os.makedirs(tmp_dir)
        os.chdir(tmp_dir)

        tmp_abs_dir = os.getcwd()

        # Create a list of files and directories
        os.makedirs('outputs')
        os.makedirs('outputs/formatted')
        os.makedirs('outputs/text')
        os.makedirs('results')
        os.makedirs('temporary')

        os.mknod('somefile.txt')
        os.mknod('somefile.txt.backup')

        os.mknod('outputs/text/text1.abc')
        os.mknod('outputs/text/text2.abc')
        os.mknod('outputs/text/text3.abc')
        os.mknod('outputs/text/text1.def')
        os.mknod('outputs/text/text15.def')

        os.mknod('outputs/formatted/formatted1.txt')
        os.mknod('outputs/formatted/formatted2.txt')
        os.mknod('outputs/formatted/formatted3.log')
        os.mknod('outputs/formatted/formatted4.log')
        os.mknod('outputs/formatted/formatted5.log')
        os.mknod('outputs/formatted/formatted6.bin')
        os.mknod('outputs/formatted/formatted7.bog')
        os.mknod('outputs/formatted/formatted8.beg')

        os.mknod('temporary/item.log')
        os.mknod('temporary/result.txt')

        # Create a first cleaner
        cleaner1 = Cleaner(dirs_and_patterns=[{'directory': 'outputs/text',
                                               'patterns': ['text?.abc']}])
        cleaner1.clean(root_directory=tmp_abs_dir)

        # Check that the first cleaner did his job correctly
        self.assertTrue(os.path.exists('outputs/text/text1.def'))
        self.assertTrue(os.path.exists('outputs/text/text15.def'))
        self.assertFalse(os.path.exists('outputs/text/text1.abc'))
        self.assertFalse(os.path.exists('outputs/text/text2.abc'))
        self.assertFalse(os.path.exists('outputs/text/text3.abc'))

        # Create a second cleaner
        cleaner2 = Cleaner(dirs_and_patterns=[{'directory': '.',
                                               'patterns': ['temporary']},
                                              {'directory': 'outputs/formatted',
                                               'patterns': ['*[1-4].log', '*.b?g']}])
        cleaner2.clean(root_directory=tmp_abs_dir)

        # Check that the first cleaner did his job correctly
        self.assertTrue(os.path.exists('outputs/formatted/formatted1.txt'))
        self.assertTrue(os.path.exists('outputs/formatted/formatted2.txt'))
        self.assertFalse(os.path.exists('outputs/formatted/formatted3.log'))
        self.assertFalse(os.path.exists('outputs/formatted/formatted4.log'))
        self.assertTrue(os.path.exists('outputs/formatted/formatted5.log'))
        self.assertTrue(os.path.exists('outputs/formatted/formatted6.bin'))
        self.assertFalse(os.path.exists('outputs/formatted/formatted7.bog'))
        self.assertFalse(os.path.exists('outputs/formatted/formatted8.beg'))
        self.assertFalse(os.path.exists('temporary'))
        self.assertTrue(os.path.exists('outputs/formatted'))

        # Change back to the initial working directory and remove the tmp directory
        os.chdir(cwd)
        shutil.rmtree(tmp_dir)
