project(PathTracing-PBR)

# Header Files
set(CXX_HEADERS 
	vendor/stb_image.h
	vendor/stb_image_write.h
	image/Image.h
	geometry/Vertex.h
	geometry/Mesh.h
)

set(CUDA_HEADERS
	kernel/kernel.cuh
)

# Source Files
set(CXX_SOURCES
	main.cpp
	image/Image.cpp
)

set(CUDA_SOURCES
	kernel/kernel.cu
)

include_directories(vendor/glm)

# Add CUDA executable
add_executable(${PROJECT_NAME} ${CXX_SOURCES} ${CXX_HEADERS} ${CUDA_SOURCES} ${CUDA_HEADERS})

# Set CUDA properties
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA executable with CUDA libraries
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})