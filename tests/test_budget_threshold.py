import unittest
import numpy as np
from budget_threshold import threshold_for_budget, cost_curve


class BudgetThresholdTests(unittest.TestCase):
    def test_tie_is_not_split_to_exceed_budget(self):
        # Prepare the test inputs and verify the expected decisions and metrics.
        """
        Verify that a tied score group is excluded when it exceeds the budget.

        Parameters
        ----------
            None
                This function takes no explicit parameters.

        Returns
        -------
            None
                Assertions validate the expected behavior or raise an assertion failure.

        """
        result = threshold_for_budget([.9,.8,.8,.1], 5, 10)
        self.assertEqual(result['reviews'], 1)
        self.assertEqual(result['remaining_budget'], 5)
        np.testing.assert_array_equal(result['selected'], [True,False,False,False])

    def test_money_uses_decimal_capacity(self):
        # Prepare the test inputs and verify the expected decisions and metrics.
        """
        Verify that decimal currency arithmetic preserves affordable review capacity.

        Parameters
        ----------
            None
                This function takes no explicit parameters.

        Returns
        -------
            None
                Assertions validate the expected behavior or raise an assertion failure.

        """
        result = threshold_for_budget([.9,.8,.7], '.10', '.30')
        self.assertEqual(result['reviews'], 3)
        self.assertEqual(result['spent'], .30)

    def test_unaffordable_tied_top_group_selects_none(self):
        # Prepare the test inputs and verify the expected decisions and metrics.
        """
        Verify that no alert is selected when the highest tied group is unaffordable.

        Parameters
        ----------
            None
                This function takes no explicit parameters.

        Returns
        -------
            None
                Assertions validate the expected behavior or raise an assertion failure.

        """
        result = threshold_for_budget([1,1,.5], 5, 5)
        self.assertEqual(result['reviews'], 0)
        self.assertTrue(np.isinf(result['threshold']))

    def test_zero_and_empty_batch(self):
        # Prepare the test inputs and verify the expected decisions and metrics.
        """
        Verify that empty batches and zero budgets produce no reviews.

        Parameters
        ----------
            None
                This function takes no explicit parameters.

        Returns
        -------
            None
                Assertions validate the expected behavior or raise an assertion failure.

        """
        for scores,budget in [([],100),([.9,.8],0)]:
            with self.subTest(scores=scores,budget=budget):
                self.assertEqual(threshold_for_budget(scores,5,budget)['reviews'],0)

    def test_curve_has_no_impossible_partial_ties(self):
        # Prepare the test inputs and verify the expected decisions and metrics.
        """
        Verify that the cost curve contains only complete tied-score groups.

        Parameters
        ----------
            None
                This function takes no explicit parameters.

        Returns
        -------
            None
                Assertions validate the expected behavior or raise an assertion failure.

        """
        curve=cost_curve([1,0,1],[.9,.8,.8],5)
        self.assertEqual(curve.reviews.tolist(),[0,1,3])
        self.assertEqual(curve.true_positive.tolist(),[0,1,2])
        self.assertEqual(curve.verification_spend.tolist(),[0,5,15])
        self.assertEqual(curve.recall.iloc[-1],1)

    def test_invalid_inputs(self):
        # Prepare the test inputs and verify the expected decisions and metrics.
        """
        Verify that invalid costs, budgets, and scores raise validation errors.

        Parameters
        ----------
            None
                This function takes no explicit parameters.

        Returns
        -------
            None
                Assertions validate the expected behavior or raise an assertion failure.

        """
        for cost,budget,scores in [(0,10,[.5]),(-1,10,[.5]),(1,-1,[.5]),(1.001,10,[.5]),(1,10,[np.nan]),(1,10,[1.1])]:
            with self.subTest(cost=cost,budget=budget,scores=scores):
                with self.assertRaises(ValueError):threshold_for_budget(scores,cost,budget)

    def test_property_budget_and_inclusive_threshold(self):
        # Prepare the test inputs and verify the expected decisions and metrics.
        """
        Verify budget compliance and inclusive threshold decisions across multiple budgets.

        Parameters
        ----------
            None
                This function takes no explicit parameters.

        Returns
        -------
            None
                Assertions validate the expected behavior or raise an assertion failure.

        """
        random=np.random.default_rng(21)
        scores=np.round(random.random(100),1)
        for budget in range(0,110,3):
            result=threshold_for_budget(scores,.17,budget/10)
            self.assertLessEqual(result['spent'],budget/10)
            np.testing.assert_array_equal(result['selected'],scores>=result['threshold'])


if __name__=='__main__':
    unittest.main()
