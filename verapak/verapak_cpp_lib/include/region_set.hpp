#ifndef REGION_SET_HPP_INCLUDED
#define REGION_SET_HPP_INCLUDED

#include <map>
#include <set>
#include <utility>
#include <vector>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace python = boost::python;
namespace numpy = boost::python::numpy;

namespace grid {
struct region_less_compare;
using numeric_type_t = double;
using point = std::vector<numeric_type_t>;
//using region_element = std::pair<numeric_type_t, numeric_type_t>;
//using region = std::vector<region_element>;
using region = std::pair<point, point>;
using region_pair = std::pair<region, python::tuple>;
} // namespace grid

bool operator<(grid::point const &, grid::region const &);
bool operator<(grid::region const &, grid::point const &);
bool operator<(grid::region const &, grid::region const &);
bool operator<(grid::point const &, grid::point const &);

bool operator<(numpy::ndarray const &, numpy::ndarray const &);

namespace grid {
struct region_less_compare {
  using is_transparent = void;
  bool operator()(grid::point const &p, grid::region const &r) const {
    return p < r;
  }
  bool operator()(grid::region const &r, grid::point const &p) const {
    return r < p;
  }
  bool operator()(grid::region const &r1, grid::region const &r2) const {
    return r1 < r2;
  }
  bool operator()(grid::point const &p1, grid::point const &p2) const {
    return p1 < p2;
  }
  bool operator()(grid::point const& p, grid::region_pair const& r) const {
      return p < r.first;
  }
  bool operator()(grid::region_pair const& r, grid::point const& p) const {
      return r.first < p;
  }
  bool operator()(grid::region_pair const& r1, grid::region_pair const& r2) const {
      return r1.first < r2.first;
  }
};

struct ndarray_less_compare {
    using is_transparent = void;
    bool operator()(numpy::ndarray const& a, numpy::ndarray const& b) {
        return a < b;
    }
};
} // namespace grid



using region_set = std::set<grid::region_pair, grid::region_less_compare>;
using region_key_point_map =
    std::map<grid::region, grid::point, grid::region_less_compare>;
using region_stack = std::vector<grid::region>;
using point_stack = std::vector<grid::point>;
using point_set = std::set<numpy::ndarray, grid::ndarray_less_compare>;

struct RegionSet {
  region_set region_set_internal;
  bool insert(numpy::ndarray const &, numpy::ndarray const &, python::tuple const &);
  std::size_t size();
  python::tuple get_and_remove_region_containing_point(numpy::ndarray const &);
  python::tuple pop_random();
};

struct PointSet {
    point_set point_set_internal;
    bool insert(numpy::ndarray const&);
    std::size_t size();
    point_set::iterator begin();
    point_set::iterator end();
};

#endif

