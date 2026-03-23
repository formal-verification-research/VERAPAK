#ifndef VERAPAK_UTILS_HPP
#define VERAPAK_UTILS_HPP

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <vector>
#include <stdexcept>
#include <cmath> // For NAN

namespace py = boost::python;
namespace np = boost::python::numpy;

/* ===== Point ===== */
typedef np::ndarray Point;

/* ===== PointSet ===== */
typedef std::vector<Point> PointSet;

/* ===== Region ===== */
struct Region {
    Point low;
    Point high;

    // Constructor
    Region(
        const Point& low,
        const Point& high
    );

    // Function to check if a point is contained in the region
    bool contains_point(const Point& point) const;

    bool operator==(const Region& other) const;

    int size() const;
    py::tuple shape() const;
};

/* ===== RegionSet ===== */
typedef std::vector<Region> RegionSet;

// Function to get the region that contains the specified point
Region& RegionSet_get_region_containing_point(RegionSet& self, Point point);

/* ===== BOOST_PYTHON_MODULE declaration ===== */
//BOOST_PYTHON_MODULE(verapak_utils);

#endif // VERAPAK_UTILS_HPP
