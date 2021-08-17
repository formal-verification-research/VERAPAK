#include "greet.hpp"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

BOOST_PYTHON_MODULE(verapak_utils)
{
    Py_Initialize();
    boost::python::numpy::initialize();
    using namespace boost::python;
    def("greet", greet);
}
