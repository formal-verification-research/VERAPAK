#include "greet.hpp"
#include "numpy_helpers.hpp"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

void wrapConvert(boost::python::numpy::ndarray const &in) {
  ndarrayToFlatVector(in);
}

BOOST_PYTHON_MODULE(verapak_utils) {
  Py_Initialize();
  boost::python::numpy::initialize();
  boost::python::def("greet", greet);
  boost::python::def("wrapConvert", wrapConvert);
}
