project(VERAPAK)
cmake_minimum_required(VERSION 3.10.2)

# Find Python 3: automatically finds includes and libraries 
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

find_package(Boost 1.88.0 COMPONENTS python system numpy REQUIRED)

# Add your source include directory
include_directories(${CMAKE_SOURCE_DIR}/verapak/verapak_cpp_lib/include)


file(GLOB SOURCES verapak/verapak_cpp_lib/src/*.cpp)
add_library(verapak_utils SHARED ${SOURCES})

target_link_libraries(verapak_utils PUBLIC
    Python3::Python     # Correct way to link to Python development library
    Boost::python       # Correct way to link to Boost.Python
    Boost::system       # Correct way to link to Boost.System
    Boost::numpy
)

set_target_properties(verapak_utils PROPERTIES PREFIX "")
set_property(TARGET verapak_utils PROPERTY CXX_STANDARD 17)
install(TARGETS verapak_utils DESTINATION /usr/local/lib/python3.8/dist-packages)
