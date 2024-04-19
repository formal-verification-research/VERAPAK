#include "numpy_helpers.hpp"
#include <iostream>

grid::point numpyArrayToPoint(numpy::ndarray const &in) {
  numpy::ndarray flat_in = in.reshape(python::make_tuple(-1));
  flat_in = flat_in.astype(numpy::dtype::get_builtin<grid::numeric_type_t>());
  auto size = flat_in.shape(0);
  grid::point retVal;
  retVal.reserve(size);
  auto *data_begin =
      reinterpret_cast<grid::numeric_type_t *>(flat_in.get_data());
  auto *data_end = data_begin + size;
  std::copy(data_begin, data_end, std::back_inserter(retVal));
  return retVal;
}

numpy::ndarray pointToNumpyArray(grid::point const &in) {
  auto dtype = numpy::dtype::get_builtin<grid::numeric_type_t>();
  auto shape = python::make_tuple(in.size());
  auto stride = python::make_tuple(sizeof(grid::numeric_type_t));
  auto retVal =
      numpy::from_data(in.data(), dtype, shape, stride, python::object());

  return retVal.copy();
}

grid::region_pair pointPairAndAttributesToRegionPair(numpy::ndarray const &lower,
                               numpy::ndarray const &upper,
                               python::tuple const &attributes) {
  auto flat_lower = lower.reshape(python::make_tuple(-1));
  auto flat_upper = upper.reshape(python::make_tuple(-1));
  auto size_lower = flat_lower.shape(0);
  auto size_upper = flat_upper.shape(0);
  if (size_lower != size_upper) {
    PyErr_SetString(PyExc_TypeError,
                    "Region upper and lower bound point sizes do not match");
    python::throw_error_already_set();
  }
  flat_lower =
      flat_lower.astype(numpy::dtype::get_builtin<grid::numeric_type_t>());
  flat_upper =
      flat_upper.astype(numpy::dtype::get_builtin<grid::numeric_type_t>());
  auto region = std::make_pair(numpyArrayToPoint(lower), numpyArrayToPoint(upper));
  if (region.first > region.second) {
    PyErr_SetString(PyExc_TypeError, "Lower bound is greater than upper bound");
    python::throw_error_already_set();
  }
  return std::make_pair(region, attributes);
}

python::tuple regionPairToPointPairAndAttributes(grid::region_pair const &in) {
  auto numpy_lower = pointToNumpyArray(in.first.first);
  auto numpy_upper = pointToNumpyArray(in.first.second);
  return python::make_tuple(numpy_lower, numpy_upper, in.second);
}
