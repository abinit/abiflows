# coding: utf-8
from __future__ import print_function, division, unicode_literals, absolute_import

from abiflows.core.testing import AbiflowsTest
from abiflows.fireworks.tasks.src_tasks_abc import SRCCleanerOptions


class TestSRCCleanerOptions(AbiflowsTest):
    def test_steps_to_clean(self):
        # Clean all option
        opt = SRCCleanerOptions.clean_all()
        to_clean = opt.steps_to_clean(this_step_index=4, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [1, 2, 3, 4])
        to_clean = opt.steps_to_clean(this_step_index=1, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [1])

        # Clean all except last option
        opt = SRCCleanerOptions.clean_all_except_last()
        to_clean = opt.steps_to_clean(this_step_index=6, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [1, 2, 3, 4, 5])
        to_clean = opt.steps_to_clean(this_step_index=1, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [])

        # Test errors raised
        self.assertRaises(ValueError, SRCCleanerOptions, when_to_clean='EACH_STEP',
                          current_src_states_allowed=['RECOVERABLE', 'FINALIZED'],
                          which_src_steps_to_clean='the_one_before_the_7_previous_one')
                                        #should be "the_one_before_the_7_previous_ones"
        self.assertRaises(ValueError, SRCCleanerOptions, when_to_clean='EACH_STEP',
                          current_src_states_allowed=['RECOVERABLES', 'FINALIZED'],
                          which_src_steps_to_clean='the_one_before_the_7_previous_ones')
                                        #"RECOVERABLES" should be "RECOVERABLE"
        self.assertRaises(ValueError, SRCCleanerOptions, when_to_clean='EACH_STEP',
                          current_src_states_allowed=['RECOVERABLE', 'FINALIZED'],
                          which_src_steps_to_clean='single_0')
                                        #a single step index to be cleaned should be >= 1
        self.assertRaises(ValueError, SRCCleanerOptions, when_to_clean='ALL_STEPS',
                          current_src_states_allowed=['RECOVERABLE', 'FINALIZED'],
                          which_src_steps_to_clean='single_0')
                                    #when_to_clean should be one of 'EACH_STEP', 'LAST_STEP', 'EACH_STEP_EXCEPT_LAST'

        # Testing all
        opt = SRCCleanerOptions(when_to_clean='EACH_STEP',
                                current_src_states_allowed=['RECOVERABLE', 'FINALIZED'],
                                which_src_steps_to_clean='all')
        to_clean = opt.steps_to_clean(this_step_index=4, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [1, 2, 3, 4])
        to_clean = opt.steps_to_clean(this_step_index=1, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [1])

        # Testing this_one
        opt = SRCCleanerOptions(when_to_clean='EACH_STEP',
                                current_src_states_allowed=['FINALIZED'],
                                which_src_steps_to_clean='this_one')
        to_clean = opt.steps_to_clean(this_step_index=4, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [4])
        to_clean = opt.steps_to_clean(this_step_index=1, this_step_state='RECOVERABLE')
        self.assertEqual(to_clean, [])
        to_clean = opt.steps_to_clean(this_step_index=1, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [1])

        # Testing all_before_this_one
        opt = SRCCleanerOptions(when_to_clean='EACH_STEP',
                                current_src_states_allowed=['FINALIZED'],
                                which_src_steps_to_clean='all_before_this_one')
        to_clean = opt.steps_to_clean(this_step_index=4, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [1, 2, 3])
        to_clean = opt.steps_to_clean(this_step_index=1, this_step_state='RECOVERABLE')
        self.assertEqual(to_clean, [])
        to_clean = opt.steps_to_clean(this_step_index=1, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [])
        to_clean = opt.steps_to_clean(this_step_index=2, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [1])

        # Testing all_before_the_previous_one
        opt = SRCCleanerOptions(when_to_clean='EACH_STEP',
                                current_src_states_allowed=['FINALIZED'],
                                which_src_steps_to_clean='all_before_the_previous_one')
        to_clean = opt.steps_to_clean(this_step_index=4, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [1, 2])
        to_clean = opt.steps_to_clean(this_step_index=1, this_step_state='RECOVERABLE')
        self.assertEqual(to_clean, [])
        to_clean = opt.steps_to_clean(this_step_index=1, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [])
        to_clean = opt.steps_to_clean(this_step_index=2, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [])
        to_clean = opt.steps_to_clean(this_step_index=3, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [1])

        # Testing the_one_before_this_one
        opt = SRCCleanerOptions(when_to_clean='EACH_STEP',
                                current_src_states_allowed=['RECOVERABLE', 'FINALIZED'],
                                which_src_steps_to_clean='the_one_before_this_one')
        to_clean = opt.steps_to_clean(this_step_index=6, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [5])
        to_clean = opt.steps_to_clean(this_step_index=2, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [1])

        # Testing the_one_before_the_previous_one
        opt = SRCCleanerOptions(when_to_clean='EACH_STEP',
                                current_src_states_allowed=['RECOVERABLE', 'FINALIZED'],
                                which_src_steps_to_clean='the_one_before_the_previous_one')
        to_clean = opt.steps_to_clean(this_step_index=6, this_step_state='UNRECOVERABLE')
        self.assertEqual(to_clean, [])
        to_clean = opt.steps_to_clean(this_step_index=6, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [4])
        to_clean = opt.steps_to_clean(this_step_index=2, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [])

        # Testing all_before_the_N_previous_ones
        opt = SRCCleanerOptions(when_to_clean='EACH_STEP',
                                current_src_states_allowed=['FINALIZED'],
                                which_src_steps_to_clean='all_before_the_3_previous_ones')
        to_clean = opt.steps_to_clean(this_step_index=4, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [])
        to_clean = opt.steps_to_clean(this_step_index=1, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [])
        to_clean = opt.steps_to_clean(this_step_index=3, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [])
        to_clean = opt.steps_to_clean(this_step_index=5, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [1])
        to_clean = opt.steps_to_clean(this_step_index=8, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [1, 2, 3, 4])
        to_clean = opt.steps_to_clean(this_step_index=8, this_step_state='RECOVERABLE')
        self.assertEqual(to_clean, [])

        # Testing the_one_before_the_N_previous_ones
        opt = SRCCleanerOptions(when_to_clean='EACH_STEP',
                                current_src_states_allowed=['FINALIZED'],
                                which_src_steps_to_clean='the_one_before_the_2_previous_ones')
        to_clean = opt.steps_to_clean(this_step_index=4, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [1])
        to_clean = opt.steps_to_clean(this_step_index=1, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [])
        to_clean = opt.steps_to_clean(this_step_index=3, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [])
        to_clean = opt.steps_to_clean(this_step_index=5, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [2])
        to_clean = opt.steps_to_clean(this_step_index=8, this_step_state='FINALIZED')
        self.assertEqual(to_clean, [5])
        to_clean = opt.steps_to_clean(this_step_index=8, this_step_state='RECOVERABLE')
        self.assertEqual(to_clean, [])



        # ['all', 'this_one', 'all_before_this_one', 'all_before_the_previous_one',
        #                                 'the_one_before_this_one', 'the_one_before_the_previous_one']