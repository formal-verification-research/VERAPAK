#include "region_set.hpp"
#include "numpy_helpers.hpp"

bool operator<(grid::point const &p, grid::region const &r) {
  for (auto i = 0u; i < p.size(); ++i) {
    if (p[i] < r[i].first)
      return true;
    if (p[i] >= r[i].second)
      return false;
  }
  return false;
}

bool operator<(grid::region const &r, grid::point const &p) {
  for (auto i = 0u; i < p.size(); ++i) {
    if (r[i].second <= p[i])
      return true;
    if (r[i].first > p[i])
      return false;
  }
  return false;
}

bool operator<(grid::region const &r1, grid::region const &r2) {
  for (auto i = 0u; i < r1.size(); ++i) {
    if (r1[i].second <= r2[i].first)
      return true;
    if (r1[i].first >= r2[i].second)
      return false;
  }
  return false;
}

bool operator<(grid::point const &p1, grid::point const &p2) {
  for (auto i = 0u; i < p1.size(); ++i) {
    if (p1[i] < p2[i])
      return true;
    if (p1[i] > p2[i])
      return false;
  }
  return false;
}

bool operator<(numpy::ndarray const &p1, numpy::ndarray const &p2) {
  auto v1 = numpyArrayToPoint(p1);
  auto v2 = numpyArrayToPoint(p2);
  return v1 < v2;
}

#include <iostream>

bool RegionSet::insert(numpy::ndarray const &l, numpy::ndarray const &u, python::tuple &a) {
  auto region = pointPairAndAttributesToRegionPair(l, u, a);
  auto [iter, inserted] = region_set_internal.insert(region);
  return inserted;
}

python::tuple
RegionSet::get_and_remove_region_containing_point(numpy::ndarray const &p) {
  auto point = numpyArrayToPoint(p);
  auto iter = region_set_internal.find(point);
  if (iter == region_set_internal.end()) {
    return python::make_tuple(false, python::object());
  }
  auto python_region = regionPairToPointPairAndAttributes(*iter);
  region_set_internal.erase(iter);
  return python::make_tuple(true, python_region);
}

std::size_t RegionSet::size() { return region_set_internal.size(); }

python::tuple RegionSet::pop_random() {
  if (region_set_internal.empty()) {
    return python::make_tuple(false, python::object());
  }
  auto iter = region_set_internal.begin(); 
  std::advance(iter, rand() % region_set_internal.size());
  auto python_region = regionPairToPointPairAndAttributes(*iter);
  region_set_internal.erase(iter);
  return python::make_tuple(true, python_region);
}

bool PointSet::insert(numpy::ndarray const &p) {
  return point_set_internal.insert(p.copy()).second;
}

point_set::iterator PointSet::begin() { return point_set_internal.begin(); }

point_set::iterator PointSet::end() { return point_set_internal.end(); }

std::size_t PointSet::size() { return point_set_internal.size(); }
