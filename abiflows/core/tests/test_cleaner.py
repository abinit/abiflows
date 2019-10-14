# coding: utf-8
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
        tmp_dir = '_tmp_cleaner'
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
        os.chdir(tmp_dir)

        tmp_abs_dir = os.getcwd()

        # Create a list of files and directories
        os.makedirs('outputs')
        os.makedirs('outputs/formatted')
        os.makedirs('outputs/text')
        os.makedirs('results')
        os.makedirs('temporary')

        open('somefile.txt', "w").close()
        open('somefile.txt.backup', "w").close()

        open('outputs/text/text1.abc', "w").close()
        open('outputs/text/text2.abc', "w").close()
        open('outputs/text/text3.abc', "w").close()
        open('outputs/text/text1.def', "w").close()
        open('outputs/text/text15.def', "w").close()

        open('outputs/formatted/formatted1.txt', "w").close()
        open('outputs/formatted/formatted2.txt', "w").close()
        open('outputs/formatted/formatted3.log', "w").close()
        open('outputs/formatted/formatted4.log', "w").close()
        open('outputs/formatted/formatted5.log', "w").close()
        open('outputs/formatted/formatted6.bin', "w").close()
        open('outputs/formatted/formatted7.bog', "w").close()
        open('outputs/formatted/formatted8.beg', "w").close()

        open('temporary/item.log', "w").close()
        open('temporary/result.txt', "w").close()

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
