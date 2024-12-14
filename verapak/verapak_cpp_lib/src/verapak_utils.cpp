#include "verapak_utils.hpp"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <random>
#include <stdexcept>
#include <cmath>  // For NAN
#include <boost/optional.hpp>

namespace py = boost::python;
namespace np = boost::python::numpy;

/* ===== RegionData Implementation ===== */
RegionData::RegionData(
        float confidence,
        float confidence_parent,
        float confidence_grandparent,
        bool siblings_equal,
        int recursion_depth,
        boost::optional<np::ndarray> adversarial_example
) :
    confidence(confidence),
    confidence_parent(confidence_parent),
    confidence_grandparent(confidence_grandparent),
    siblings_equal(siblings_equal),
    recursion_depth(recursion_depth),
    adversarial_example(adversarial_example),
    initialized(false) {}

RegionData RegionData::make_child() const {
    return RegionData(
        NAN, // confidence
        confidence, // confidence_parent
        confidence_parent, // confidence_grandparent
        true, // MUST BE SET LATER!!!
        0, // recursion_depth
        boost::none // adversarial_example
    );
}

boost::optional<float> RegionData::get_confidence() const {
    if (initialized) return confidence;
    else return boost::none;
}
boost::optional<bool> RegionData::get_siblings_equal() const {
    if (initialized) return siblings_equal;
    else return boost::none;
}
boost::optional<np::ndarray> RegionData::get_adversarial_example() const {
    return adversarial_example;
}
void RegionData::set_confidence(float value) {
    initialized = true;
    confidence = value;
}
void RegionData::set_siblings_equal(bool value) {
    siblings_equal = value;
}
void RegionData::set_adversarial_example(boost::optional<np::ndarray> value) {
    adversarial_example = value;
}

const RegionData RegionData_EMPTY = RegionData();

/* ===== Region Implementation ===== */
Region::Region(
        const Point& low,
        const Point& high,
        const RegionData& data
) :
    low(low),
    high(high),
    data(data) {}
Region::Region(
        const Point& low,
        const Point& high,
        const Region& parent
) :
    low(low),
    high(high),
    data(parent.data.make_child()) {}

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

/* ===== Python Type Converters ===== */
struct optional_float_to_python {
    static PyObject* convert(const boost::optional<float>& opt) {
        if (opt) {
            return PyFloat_FromDouble(opt.get());
        } else {
            Py_INCREF(Py_None);
            return Py_None;
        }
    }
};
struct optional_ndarray_to_python {
    static PyObject* convert(const boost::optional<np::ndarray>& opt) {
        if (opt) {
            return opt.get().ptr();
        } else {
            Py_INCREF(Py_None);
            return Py_None;
        }
    }
};
struct optional_ndarray_from_python {
    static void* convertible(PyObject* obj) {
        if (obj == Py_None) {
            return obj;
        }
        try {
            return py::extract<np::ndarray>(obj).check() ? obj : 0;
        } catch (const std::exception& e) {
        }
        return nullptr;
    }

    static void construct(PyObject* obj, py::converter::rvalue_from_python_stage1_data* data) {
        // Get the pointer to the storage area for the C++ object
        void* storage = (
            (py::converter::rvalue_from_python_storage<boost::optional<np::ndarray> >*) data
        )->storage.bytes;
        if (obj == Py_None) {
            // If the Python object is None, construct an empty optional
            new (storage) boost::optional<np::ndarray>();
        } else {
            // Otherwise, construct the boost::optional<np::ndarray> from the ndarray
            const np::ndarray arr = py::extract<np::ndarray>(obj);
            new (storage) boost::optional<np::ndarray>(arr);
        }
        data->convertible = storage;
    }

    static void register_converter() {
        py::converter::registry::push_back(
            &convertible, &construct, py::type_id<boost::optional<np::ndarray>>());
    }
};

/* ===== BOOST_PYTHON_MODULE ===== */
BOOST_PYTHON_MODULE(verapak_utils) {
    Py_Initialize();
    np::initialize();

    // Point class
    py::class_<Point>("Point", py::init<np::ndarray>());

    // RegionData class
    py::class_<RegionData>("RegionData", py::init<float, float, float, bool, int>())
        .def(py::init<>())
        .add_property("confidence", &RegionData::get_confidence, &RegionData::set_confidence)
        .def_readwrite("confidence_parent", &RegionData::confidence_parent)
        .def_readwrite("confidence_grandparent", &RegionData::confidence_grandparent)
        .add_property("siblings_equal", &RegionData::get_siblings_equal, &RegionData::set_siblings_equal)
        .def_readwrite("recursion_depth", &RegionData::recursion_depth)
        .add_property("adversarial_example", &RegionData::get_adversarial_example, &RegionData::set_adversarial_example)
        .def_readonly("initialized", &RegionData::initialized)
        .def_readonly("EMPTY", &RegionData_EMPTY)
	.def("make_child", &RegionData::make_child);
    py::to_python_converter<boost::optional<float>, optional_float_to_python>();
    py::to_python_converter<boost::optional<np::ndarray>, optional_ndarray_to_python>();
    optional_ndarray_from_python::register_converter();

    // Region class
    py::class_<Region>("Region", py::init<Point, Point, RegionData>())
        .def(py::init<Point, Point, Region>())
        .def_readwrite("low", &Region::low)
        .def_readwrite("high", &Region::high)
        .def_readwrite("data", &Region::data)
        .def("__contains__", &Region::contains_point)
        .add_property("size", &Region::size)
        .add_property("shape", &Region::shape);

    // RegionSet class
    py::class_<RegionSet>("RegionSet")
        .def(py::vector_indexing_suite<std::vector<Region>>())
        .def("get_region_containing_point", &RegionSet_get_region_containing_point,
                        py::return_internal_reference<>()); // Warning - if this region is destroyed by C++, it will cause issues in Python

    // PointSet class
    py::class_<PointSet>("PointSet")
        .def(py::vector_indexing_suite<std::vector<Point>>());
}

