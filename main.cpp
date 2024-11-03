#include "engine.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

// Basically everything has been stolen from https://github.com/FourierMourier/yolov8-onnx-cpp
// I've just reformatted and simplified everything for this project
// You should be able to run Infer in a loop at max speed, which was as issue in the original project
// Additionally, I removed all but Detection tasks, as this was whats important to me :)

// This version has been updated for YoloV10 off the original project

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