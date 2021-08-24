#ifndef REGION_SET_HPP_INCLUDED
#define REGION_SET_HPP_INCLUDED

#include <map>
#include <set>
#include <utility>
#include <vector>

namespace grid {
struct region_less_compare;
using numeric_type_t = long double;
using region_element = std::pair<numeric_type_t, numeric_type_t>;
using region = std::vector<region_element>;
using point = std::vector<numeric_type_t>;
} // namespace grid

bool operator<(grid::point const &, grid::region const &);
bool operator<(grid::region const &, grid::point const &);
bool operator<(grid::region const &, grid::region const &);
bool operator<(grid::point const &, grid::point const &);

using region_set = std::set<grid::region, grid::region_less_compare>;
using region_key_point_map =
    std::map<grid::region, grid::point, grid::region_less_compare>;
using region_stack = std::vector<grid::region>;

#endif

