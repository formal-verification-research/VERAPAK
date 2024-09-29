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

/* ===== RegionData ===== */
struct RegionData {
    float confidence;
    float confidence_parent;
    float confidence_grandparent;
    bool siblings_equal;
    int recursion_depth;

    // Constructor
    RegionData(
        float confidence = NAN,
        float confidence_parent = NAN,
        float confidence_grandparent = NAN,
        bool siblings_equal = true,
        int recursion_depth = 0
    );
};

/* ===== Region ===== */
struct Region {
    Point low;
    Point high;
    RegionData data;

    // Constructor
    Region(
        const Point& low,
        const Point& high,
        const RegionData& data = RegionData()
    );

    // Function to check if a point is contained in the region
    bool contains_point(const Point& point);

    bool operator==(const Region& other) const;
};

/* ===== RegionSet ===== */
typedef std::vector<Region> RegionSet;

// Function to get the region that contains the specified point
Region& RegionSet_get_region_containing_point(RegionSet& self, Point point);

// Function to get a random region from the RegionSet
Region& RegionSet_get_random(RegionSet& self);

/* ===== BOOST_PYTHON_MODULE declaration ===== */
//BOOST_PYTHON_MODULE(verapak_utils);

#endif // VERAPAK_UTILS_HPP
