import unittest
import numpy as np
from verapak.utilities import point_tools
from verapak.dimension_ranking import gradient_based
from verapak.dimension_ranking.largest_first import LargestFirstDimSelection
from verapak.abstraction.center_point import CenterPoint
from verapak.abstraction.fallback import FallbackStrategy
from verapak.abstraction.fgsm import FGSM
from verapak.abstraction.modfgsm import ModFGSM
from verapak.abstraction.random_point import RandomPoint
from verapak.partitioning.tools import hierarchicalDimensionRefinement
from verapak.partitioning.largest_first import LargestFirstPartitioningStrategy
import verapak_utils


class PointToolsTests(unittest.TestCase):
    def setUp(self):
        self.region_boundary = [np.array([1, 3.5, 6, -1, -2.3, 4]),
                                np.array([1.1, 3.6, 6.1, -0.9, -2.2, 4.1])]
        self.region_covering = [np.array([0.5, 3.0, 5.5, -1.5, -2.8, 3.5]),
                                np.array([1.1, 3.6, 6.1, -0.9, -2.2, 4.1])]
        self.region_outside_covering = [np.array([2.5, 2, 3.5, -6.5, 5.2, -2.5]),
                                        np.array([3.5, 3, 4.5, -5.5, 6.2, -1.5])]
        self.region_novalid = [np.array([1.1, 3.6, 6.1, -0.9, -2.2, 4.1]),
                               np.array([1.9, 4.4, 6.9, -0.1, -1.2, 4.9])]
        self.region_boundary_both = [np.array([1, 3.5, 6, -1, -2.3, 4]),
                                     np.array([2, 4.5, 7, 0, -1.3, 5])]
        self.granularity = 1
        self.region_multiple = [self.region_outside_covering[0],
                                self.region_outside_covering[0] + 5*self.granularity]
        self.valid_point = np.array([[1, 3.5, 6], [-1, -2.3, 4]])
        self.valid_point2 = np.array([[3, 2.5, 4], [-6, 5.7, -2]])

    def test_valid_point_in_region(self):
        res = point_tools.get_valid_point_in_region(
            self.region_boundary, self.granularity, self.valid_point)
        self.assertTrue(np.all(self.valid_point == res))
        res = point_tools.get_valid_point_in_region(
            self.region_covering, self.granularity, self.valid_point)
        self.assertTrue(np.all(self.valid_point == res))
        self.assertTrue(point_tools.is_point_valid(self.valid_point2,
                                                   self.granularity, self.valid_point))
        res = point_tools.get_valid_point_in_region(
            self.region_outside_covering, self.granularity, self.valid_point)
        self.assertTrue(np.all(self.valid_point2 == res))
        res = point_tools.get_valid_point_in_region(
            self.region_novalid, self.granularity, self.valid_point)
        self.assertTrue(res is None)

    def test_num_valid_points(self):
        res = point_tools.get_amount_valid_points(
            self.region_boundary, self.granularity, self.valid_point)
        self.assertEqual(res, 1)
        res = point_tools.get_amount_valid_points(
            self.region_novalid, self.granularity, self.valid_point)
        self.assertEqual(res, 0)
        res = point_tools.get_amount_valid_points(
            self.region_boundary_both, self.granularity, self.valid_point)
        self.assertEqual(res, 1)
        res = point_tools.get_amount_valid_points(
            self.region_multiple, self.granularity, self.valid_point)
        self.assertEqual(res, 5**6)

    def tearDown(self):
        pass


class GradientDimSelectionTests(unittest.TestCase):
    def setUp(self):
        self.grad_strategy = gradient_based.GradientBasedDimSelection(
            lambda x: x)

    def test_dim_selection_strategy(self):
        region_lb = np.array([1, 2, 3, 4, 5, 6])
        a = self.grad_strategy.rank_indices_impl([region_lb, region_lb + 4])
        pass


class LargestFirstDimSelectionTests(unittest.TestCase):
    def setUp(self):
        self.dim_selection_strat = LargestFirstDimSelection()
        pass

    def test_lf_dim_selection(self):
        region = [np.array([1, 1, 1, 1]), np.array([2, 5, 3, 6])]
        res = self.dim_selection_strat.rank_indices_impl(region)
        expected = [3, 1, 2, 0]
        self.assertCountEqual(res, expected)
        self.assertListEqual(res, expected)
        region1 = [region[0].reshape(2, 2), region[1].reshape(2, 2)]
        res = self.dim_selection_strat.rank_indices_impl(region1)
        self.assertCountEqual(res, expected)
        self.assertListEqual(res, expected)
        pass


class PointSetTests(unittest.TestCase):
    def test_point_set(self):
        point_set = verapak_utils.PointSet()
        self.assertEqual(point_set.size(), 0)

    def test_region_set(self):
        region_set = verapak_utils.RegionSet()
        self.assertEqual(region_set.size(), 0)


class PartitioningToolsTest(unittest.TestCase):
    def setUp(self):
        self.region1 = [np.array([1.0, 1.0, 1.0, 1.0]),
                        np.array([3.0, 3.0, 3.0, 3.0])]
        self.dim_select_strategy = lambda x: [i for i in range(x[0].size)]

    def test_enumeration(self):
        res = hierarchicalDimensionRefinement(
            self.region1, self.dim_select_strategy, 4, 2)
        self.assertEqual(len(res), 2**4)
        res = hierarchicalDimensionRefinement(
            self.region1, self.dim_select_strategy, 4, 3)
        self.assertEqual(len(res), 3**4)
        res = hierarchicalDimensionRefinement([self.region1[0].reshape(
            2, 2), self.region1[1].reshape(2, 2)], self.dim_select_strategy, 4, 4)
        self.assertEqual(len(res), 4**4)


class LargestFirstPartitioningTest(unittest.TestCase):
    def setUp(self):
        self.strat = LargestFirstPartitioningStrategy(2, 2)

    def test_lf_partitioning(self):
        region1 = [np.array([1.0, 1.0, 1.0, 1.0]),
                   np.array([3.0, 3.0, 5.0, 5.0])]
        res = self.strat.partition_impl(region1)
        acc = np.full(region1[0].shape, True)
        for i in res:
            acc = acc & (i[0] == region1[0]) & (i[1] == region1[1])
        self.assertTrue(acc[0] and acc[1])
        self.assertFalse(acc[2])
        self.assertFalse(acc[3])
        lbs = [list(x[0]) for x in res]
        ubs = [list(x[1]) for x in res]
        self.assertTrue([1.0, 1.0, 3.0, 3.0] in lbs)
        self.assertTrue([1.0, 1.0, 3.0, 1.0] in lbs)
        self.assertTrue([1.0, 1.0, 1.0, 3.0] in lbs)
        self.assertTrue([1.0, 1.0, 1.0, 1.0] in lbs)

        self.assertTrue([3.0, 3.0, 5.0, 5.0] in ubs)
        self.assertTrue([3.0, 3.0, 5.0, 3.0] in ubs)
        self.assertTrue([3.0, 3.0, 3.0, 5.0] in ubs)
        self.assertTrue([3.0, 3.0, 3.0, 3.0] in ubs)

        region2 = [np.array([1.0, 1.0, 1.0, 1.0]),
                   np.array([2.0, 5.0, 3.0, 4.0])]
        res = self.strat.partition_impl(region2)
        acc = np.full(region2[0].shape, True)
        for i in res:
            acc = acc & (i[0] == region2[0]) & (i[1] == region2[1])
        self.assertTrue(acc[0] and acc[2])
        self.assertFalse(acc[1])
        self.assertFalse(acc[3])

class ModFGSMTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_modfgsm(self):
        pass

if __name__ == "__main__":
    unittest.main()
