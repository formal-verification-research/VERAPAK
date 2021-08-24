#include "numpy_helpers.hpp"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

BOOST_PYTHON_MODULE(verapak_utils) {
  Py_Initialize();
  numpy::initialize();
  python::class_<RegionSet>("RegionSet").def("insert", &RegionSet::insert);
}
