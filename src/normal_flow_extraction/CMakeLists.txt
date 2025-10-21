find_package(Eigen3 REQUIRED)

add_library(normal_flow_extraction 
SHARED
    normal_flow_extraction.cu
    math_utils.cu
)

target_include_directories(normal_flow_extraction
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(normal_flow_extraction
PUBLIC
    opencv_core
    Eigen3::Eigen
)

set_target_properties(normal_flow_extraction PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)