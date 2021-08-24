#include "numpy_helpers.hpp"
#include <iostream>

grid::point ndarrayToFlatVector(boost::python::numpy::ndarray const &in) {
  boost::python::numpy::ndarray flat_in =
      in.reshape(boost::python::make_tuple(-1));
  flat_in = flat_in.astype(
      boost::python::numpy::dtype::get_builtin<grid::numeric_type_t>());
  auto size = flat_in.shape(0);
  grid::point retVal;
  retVal.reserve(size);
  auto *data_begin =
      reinterpret_cast<grid::numeric_type_t *>(flat_in.get_data());
  auto *data_end = data_begin + size;
  std::copy(data_begin, data_end, std::back_inserter(retVal));
  std::cout << "in c++: ";
  std::copy(retVal.begin(), retVal.end(),
            std::ostream_iterator<long double>(std::cout, " "));
  std::cout << "\n";
  return retVal;
}
