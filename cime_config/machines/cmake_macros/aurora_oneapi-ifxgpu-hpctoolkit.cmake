include(aurora_oneapi-ifxgpu.cmake)
include(oneapi-ifxgpu.cmake)

# Append -g and -03 for hpctoolkit
string(APPEND CMAKE_C_FLAGS " -g -O3")
string(APPEND CMAKE_CXX_FLAGS " -g -O3")
string(APPEND CMAKE_Fortran_FLAGS " -g -O3")

