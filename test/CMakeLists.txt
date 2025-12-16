set(CMAKE_CXX_STANDARD 17)


add_executable(benchmark_hnsw benchmark_hnsw.cpp)
add_executable(benchmark_ipnsw benchmark_ipnsw.cpp)
add_executable(benchmark_mobius benchmark_mobius.cpp)
add_executable(benchmark_ipnsw+ benchmark_ipnsw+.cpp)
add_executable(benchmark_fargo benchmark_fargo.cpp)
add_executable(benchmark_dblsh benchmark_dblsh.cpp)
add_executable(benchmark_napg benchmark_napg.cpp)
add_executable(benchmark_rabitq benchmark_rabitq.cpp)
add_executable(benchmark_mag benchmark_mag.cpp)

add_executable(benchmark_nsg benchmark_nsg.cpp)
target_link_libraries(benchmark_nsg PRIVATE efanna2e)

add_executable(benchmark_vamana benchmark_vamana.cpp)
target_include_directories(benchmark_vamana PRIVATE ${MKL_INCLUDE_DIR})
target_link_libraries(benchmark_vamana PRIVATE ${MKL_LIBRARIES})
target_compile_definitions(benchmark_vamana PRIVATE MKL_ILP64)


