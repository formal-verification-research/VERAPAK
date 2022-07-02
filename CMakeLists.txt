project(VERAPAK)
cmake_minimum_required(VERSION 3.10.2)

include_directories(${CMAKE_SOURCE_DIR}/verapak/verapak_cpp_lib/include)
include_directories(/usr/include/python3.6)
include_directories(/boost_1_77_0)

link_directories(/usr/lib/x86_64-linux-gnu)
link_directories(/usr/local/lib)

file(GLOB SOURCES verapak/verapak_cpp_lib/src/*.cpp)
add_library(verapak_utils SHARED ${SOURCES})

target_link_libraries(verapak_utils boost_python36 boost_system python3.6m boost_numpy36)

set_target_properties(verapak_utils PROPERTIES PREFIX "")
set_property(TARGET verapak_utils PROPERTY CXX_STANDARD 17)

install(TARGETS verapak_utils DESTINATION /usr/local/lib/python3.6/dist-packages)
