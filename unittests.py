from verapak import strategy_registry
import unittest
import numpy as np
from verapak.utilities import point_tools
from verapak.dimension_ranking import gradient_based
from verapak.dimension_ranking.largest_first import LargestFirstDimSelection
from verapak.abstraction.center_point import CenterPoint
from verapak.abstraction.fgsm import FGSM
from verapak.abstraction.modfgsm import ModFGSM
from verapak.abstraction.rfgsm import RFGSM
from verapak.abstraction.random_point import RandomPoint
from verapak.partitioning.tools import hierarchicalDimensionRefinement
from verapak.partitioning.largest_first import LargestFirstPartitioningStrategy
from verapak.verification.discrete_search import DiscreteSearch
from verapak.verification import ve
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
    def setUp(self):
        self.p1 = np.array([1, 4, 6, 3])
        self.r1 = [self.p1, self.p1 + 1]
        self.r2 = [self.p1 + 1, self.p1 + 2]
        self.r3 = [self.p1 + 2, self.p1 + 4]

    def test_point_set(self):
        point_set = verapak_utils.PointSet()
        self.assertEqual(point_set.size(), 0)

    def test_region_set(self):
        region_set = verapak_utils.RegionSet()
        self.assertEqual(region_set.size(), 0)
        region_set.insert(*self.r1)
        region_set.insert(*self.r2)
        self.assertEqual(region_set.size(), 2)
        region_set.insert(*self.r3)
        self.assertEqual(region_set.size(), 3)
        success, region = region_set.pop_front()
        self.assertEqual(region_set.size(), 2)
        region_set.pop_front()
        region_set.pop_front()
        success, region = region_set.pop_front()
        self.assertFalse(success)
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
        self.granularity = np.array([1, 1, 1, 1])
        self.fgsm = ModFGSM(
            gradient_function=lambda x: x, granularity=np.array([1, 1, 1, 1], dtype=np.float64))
        self.rfgsm = RFGSM(gradient_function=lambda x: 0.1 * x,
                           granularity=np.array([1.0, 1.0, 1.0, 1.0]), balance_factor=0.5)

    def test_modfgsm(self):
        region1 = [np.array([0, 0, 0, 0], dtype=np.float64),
                   np.array([4, 4, 4, 4], dtype=np.float64)]
        res = self.fgsm.abstraction_impl(region1, 4)
        self.assertEqual(len(res), 4)
        res = self.fgsm.abstraction_impl(region1, 8)
        self.assertEqual(len(res), 8)
        res = self.rfgsm.abstraction_impl(region1, 4)
        self.assertEqual(len(res), 4)


class VerificationToolsTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_discrete_point_enumerator(self):
        r1 = [np.array([0, 0, 0, 0], dtype=np.float32),
              np.array([2, 2, 2, 2], dtype=np.float32)]
        g = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        vp = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        all_points = list(point_tools.enumerate_valid_points(r1, g, vp))
        self.assertEqual(len(all_points), 2**4)
        g = np.array([0.5, 0.5, 0.5, 0.5])
        all_points = list(point_tools.enumerate_valid_points(r1, g, vp))
        self.assertEqual(len(all_points), 4**4)
        unique_points = np.unique(all_points, axis=0)
        self.assertEqual(len(unique_points), len(all_points))
        vp = np.array([0.1, 0.1, 0.1, 0.1])
        all_points = list(point_tools.enumerate_valid_points(r1, g, vp))
        self.assertEqual(len(all_points), 4**4)
        unique_points = np.unique(all_points, axis=0)
        self.assertEqual(len(unique_points), len(all_points))


class VerificationEngineTests(unittest.TestCase):
    def setUp(self):
        def sa(x):
            return True
        g = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        vp = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self.discrete_verifier = DiscreteSearch(10000, g, vp)
        pass

    def test_discrete_exhaustive_search(self):
        r1 = [np.array([0, 0, 0, 0], dtype=np.float32),
              np.array([2, 2, 2, 2], dtype=np.float32)]

        def sp(p):
            return np.all(p < 2.0) and np.all(p >= 0.0)
        res, val = self.discrete_verifier.verification_impl(r1, sp)
        self.assertEqual(res, ve.SAFE)


if __name__ == "__main__":
    unittest.main()
