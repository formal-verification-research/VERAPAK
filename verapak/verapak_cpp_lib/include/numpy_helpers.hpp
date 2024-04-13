#ifndef NUMPY_HELPERS_HPP_INCLUDED
#define NUMPY_HELPERS_HPP_INCLUDED

#include "region_set.hpp"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <optional>

namespace python = boost::python;
namespace numpy = boost::python::numpy;

grid::point numpyArrayToPoint(numpy::ndarray const &);

numpy::ndarray pointToNumpyArray(grid::point const &);

grid::region_pair pointPairAndAttributesToRegionPair(numpy::ndarray const &, numpy::ndarray const &, python::tuple const &);

python::tuple regionPairToPointPairAndAttributes(grid::region_pair const &);

#endif

