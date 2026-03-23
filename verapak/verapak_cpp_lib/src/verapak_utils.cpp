#include "verapak_utils.hpp"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <random>
#include <stdexcept>
#include <cmath>  // For NAN

namespace py = boost::python;
namespace np = boost::python::numpy;

/* ===== Region Implementation ===== */
Region::Region(
        const Point& low,
        const Point& high
) :
    low(low),
    high(high)

bool Region::contains_point(const Point& point) const {
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

int Region::size() const {
    return py::extract<int>(low.attr("size"));
}
py::tuple Region::shape() const {
    return py::extract<py::tuple>(low.attr("shape"));
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

/* ===== BOOST_PYTHON_MODULE ===== */
BOOST_PYTHON_MODULE(verapak_utils) {
    Py_Initialize();
    np::initialize();

    // Point class
    py::class_<Point>("Point", py::init<np::ndarray>());

    // Region class
    py::class_<Region>("Region", py::init<Point, Point>())
        .def_readwrite("low", &Region::low)
        .def_readwrite("high", &Region::high)
        .def("__contains__", &Region::contains_point)
        .add_property("size", &Region::size)
        .add_property("shape", &Region::shape);

    // RegionSet class
    py::class_<RegionSet>("RegionSet")
        .def(py::vector_indexing_suite<std::vector<Region>>())
        .def("get_region_containing_point", &RegionSet_get_region_containing_point,
                        py::return_internal_reference<>()); // Warning - if this region is destroyed by C++, it will cause issues in Python
}
