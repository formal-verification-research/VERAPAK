#ifndef NUMPY_HELPERS_HPP_INCLUDED
#define NUMPY_HELPERS_HPP_INCLUDED

#include "region_set.hpp"

#include <boost/python/numpy.hpp>

grid::point ndarrayToFlatVector(boost::python::numpy::ndarray const &);

#endif

