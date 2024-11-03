# YOLOv11 ONNXRuntime Engine in c++

## Overview

This project is a C++ implementation of a YOLOv11 inference engine using the ONNX Runtime. It is heavily based on the project [yolov8-onnx-cpp by FourierMourier](https://github.com/FourierMourier/yolov8-onnx-cpp) and is updated from my original project [YOLOv8-ONNXRuntime-CPP by K4HVH](https://github.com/K4HVH/YOLOv8-ONNXRuntime-CPP). The primary goal of this implementation is to provide a streamlined and efficient object detection pipeline that can be easily modified to suit various client needs. This version has been heavily optimised, using AVX instructions to allow for extremely high-speed inferencing on modern systems.

## Features

- **High Performance:** Optimized for speed to allow running inference in a loop at maximum speed.
- **Simplicity:** Simplified codebase, focusing solely on object detection.
- **Flexibility:** Easy to modify and extend to fit specific requirements.
- **Greater Accuracy:** YOLOv11 has greater accuracy at the same inferencing speed compared to YoloV8.
- **Faster Inferencing:** In testing, YOLOv11-ONNXRuntime-CPP runs 4 faster for the same size model compared to YOLOv8-ONNXRuntime-CPP.

## Prerequisites

- **ONNX Runtime:** Make sure to have ONNX Runtime installed.
- **OpenCV:** Required for image processing and display.
- **C++ Compiler:** Compatible with C++11.
- **AVX Instruction Set:** This project uses AVX512 or AVX2 for maximum performance

## Getting Started

### Installation
1. **Clone the repository:**

```sh
git clone https://github.com/K4HVH/YOLOv11-ONNXRuntime-CPP
cd YOLOv11-ONNXRuntime-CPP
```

2. **Install dependencies:**
   
Ensure that ONNX Runtime and OpenCV are installed on your system. You can find installation instructions for ONNX Runtime [here](https://onnxruntime.ai/).

### Compilation

1. **Configure the project:**
   
Edit the `CMakeLists.txt` in the project root directory. Replace `"path/to/onnxruntime"` with the actual path to your ONNX Runtime installation directory.

``` scss
# Path to ONNX Runtime
set(ONNXRUNTIME_DIR "path/to/onnxruntime")
```

2. **Build the project:**

```sh
mkdir build
cd build
cmake ..
make
```

### Running the Inference

1. **Run the executable:**

```sh
./yolo_inference
```

2. **Test with your image:**
   
Modify the `imagePath` variable in `main.cpp` to point to your test image.

## Project Structure
**main.cpp:** Entry point of the application. It initializes the inferencer and runs the detection on a sample image.
**engine.hpp:** Header file for the YOLOv11 inferencer class, defining the structure and methods.
**engine.cpp:** Implementation of the YOLOv11 inferencer, including preprocessing, forward pass, and postprocessing steps.

## Example Usage
Here is a snippet from `main.cpp` demonstrating the usage:

```cpp
#include "engine.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

int main()
{
    std::wstring modelPath = L"yolo11n.onnx";
    const char* logid = "yolo_inference";
    const char* provider = "CPU"; // or "CUDA"

    YoloInferencer inferencer(modelPath, logid, provider);

    std::string imagePath = "test.jpg"; // Replace with your image path
    cv::Mat image = cv::imread(imagePath);

    if (image.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }

    std::vector<Detection> detections = inferencer.infer(image, 0.4, 0.5);

    for (const auto& detection : detections) {
        cv::rectangle(image, detection.box, cv::Scalar(0, 255, 0), 2);
        std::cout << "Detection: Class=" << detection.class_id << ", Confidence=" << detection.confidence
            << ", x=" << detection.box.x << ", y=" << detection.box.y
            << ", width=" << detection.box.width << ", height=" << detection.box.height << std::endl;
    }

    cv::imshow("output", image);
    cv::waitKey(0);

    return 0;
}
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with your changes.

## License
This project is licensed under the GPL3.0 License.

## Acknowledgments
This project borrows heavily from the original [yolov8-onnx-cpp repository](https://github.com/FourierMourier/yolov8-onnx-cpp).
This project is an update from my original project [YOLOv8-ONNXRuntime-CPP](https://github.com/K4HVH/YOLOv8-ONNXRuntime-CPP) to support YoloV10.
