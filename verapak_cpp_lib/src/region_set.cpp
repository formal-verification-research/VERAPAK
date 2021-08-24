#include "region_set.hpp"


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
};
} // namespace grid


