cmake_minimum_required(VERSION 3.10)
project(YOLOv11-ONNXRuntime-CPP)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Path to ONNX Runtime
set(ONNXRUNTIME_DIR "path/to/onnxruntime")

# Path to OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${ONNXRUNTIME_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Link directories
link_directories(${ONNXRUNTIME_DIR}/lib)

# Source files
set(SOURCES
    main.cpp
    engine.cpp
)

# Executable
add_executable(yolo_inference ${SOURCES})

# Link libraries
target_link_libraries(yolo_inference ${OpenCV_LIBS} onnxruntime onnxruntime_providers_cuda)
