#include "verapak_utils.hpp"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <random>
#include <stdexcept>
#include <cmath>  // For NAN

namespace py = boost::python;
namespace np = boost::python::numpy;

/* ===== RegionData Implementation ===== */
RegionData::RegionData(
        float confidence,
        float confidence_parent,
        float confidence_grandparent,
        bool siblings_equal,
        int recursion_depth
) :
    confidence(confidence),
    confidence_parent(confidence_parent),
    confidence_grandparent(confidence_grandparent),
    siblings_equal(siblings_equal),
    recursion_depth(recursion_depth) {}

/* ===== Region Implementation ===== */
Region::Region(
        const Point& low,
        const Point& high,
        const RegionData& data
) :
    low(low),
    high(high),
    data(data) {}

bool Region::contains_point(const Point& point) {
    int ndim = point.get_nd();
    for (int i = 0; i < ndim; ++i) {
        double point_value = boost::python::extract<double>(point[i]);
        double low_value = boost::python::extract<double>(low[i]);
        double high_value = boost::python::extract<double>(high[i]);

        if (point_value <= low_value || point_value >= high_value) {
            return false;
        }
    }
    return true;
}

bool Region::operator==(const Region& other) const {
    return (low == other.low && high == other.high);
}

/* ===== RegionSet Functions Implementation ===== */
Region& RegionSet_get_region_containing_point(RegionSet& self, Point point) {
    for (Region& region : self) {
        if (region.contains_point(point)) {
            return region;
        }
    }
    throw std::out_of_range("No region contains the given point.");
}

Region& RegionSet_get_random(RegionSet& self) {
    if (self.empty()) {
        throw std::out_of_range("Cannot get a value from an empty vector");
    }
    auto iter = self.begin();
    std::advance(iter, rand() % self.size());
    return *iter;
}

/* ===== BOOST_PYTHON_MODULE ===== */
BOOST_PYTHON_MODULE(verapak_utils) {
    Py_Initialize();
    np::initialize();

    // Point class
    py::class_<Point>("Point", py::init<np::ndarray>());

    // RegionData class
    py::class_<RegionData>("RegionData", py::init<float, float, float, bool, int>())
        .def_readwrite("confidence", &RegionData::confidence)
        .def_readwrite("confidence_parent", &RegionData::confidence_parent)
        .def_readwrite("confidence_grandparent", &RegionData::confidence_grandparent)
        .def_readwrite("siblings_equal", &RegionData::siblings_equal)
        .def_readwrite("recursion_depth", &RegionData::recursion_depth);

    // Region class
    py::class_<Region>("Region", py::init<Point, Point, RegionData>())
        .def_readwrite("low", &Region::low)
        .def_readwrite("high", &Region::high)
        .def_readwrite("data", &Region::data)
        .def("__contains__", &Region::contains_point);

    // RegionSet class
    py::class_<RegionSet>("RegionSet")
        .def(py::vector_indexing_suite<std::vector<Region>>())
        .def("get_region_containing_point", &RegionSet_get_region_containing_point,
			py::return_internal_reference<>()) // Warning - if this region is destroyed by C++, it will cause issues in Python
        .def("pop_random", &RegionSet_get_random,
			py::return_internal_reference<>()); // Warning - if this region is destroyed by C++, it will cause issues in Python

    // PointSet class
    py::class_<PointSet>("PointSet")
        .def(py::vector_indexing_suite<std::vector<Point>>());
}

