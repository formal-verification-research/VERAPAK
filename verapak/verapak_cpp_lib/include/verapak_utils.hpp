#ifndef VERAPAK_UTILS_HPP
#define VERAPAK_UTILS_HPP

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <vector>
#include <stdexcept>
#include <cmath> // For NAN
#include <boost/optional.hpp>

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
    boost::optional<np::ndarray> adversarial_example;
    bool initialized;

    // Constructor
    RegionData(
        float confidence = NAN,
        float confidence_parent = NAN,
        float confidence_grandparent = NAN,
        bool siblings_equal = false,
        int recursion_depth = 0,
	boost::optional<np::ndarray> adversarial_example = boost::none
    );

    RegionData make_child() const;

    bool is_initialized() const;
    boost::optional<float> get_confidence() const;
    boost::optional<bool> get_siblings_equal() const;
    boost::optional<np::ndarray> get_adversarial_example() const;
    void set_confidence(float value);
    void set_siblings_equal(bool value);
    void set_adversarial_example(boost::optional<np::ndarray> value);
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
    // Inheritance Constructor
    Region(
        const Point& low,
        const Point& high,
        const Region& parent
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
