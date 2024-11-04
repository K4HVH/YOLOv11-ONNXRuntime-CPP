#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>
#include <regex>

struct Detection {
    cv::Rect box;
    int class_id;
    float confidence;
};

class YoloInferencer {
public:
    YoloInferencer(std::wstring& modelPath, const char* logid, const char* provider);
    ~YoloInferencer();
    std::vector<Detection> infer(cv::Mat& frame, float conf_threshold, float iou_threshold);

private:
    std::vector<Ort::Value> preprocess(cv::Mat& frame);
    std::vector<Ort::Value> forward(std::vector<Ort::Value>& inputTensors);
    std::vector<Detection> postprocess(std::vector<Ort::Value>& outputTensors, float conf_threshold, float iou_threshold);

    Ort::Env env_{ nullptr };
    Ort::Session session_{ nullptr };

    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    std::vector<const char*> inputNamesCStr_;
    std::vector<const char*> outputNamesCStr_;

    Ort::ModelMetadata model_metadata{ nullptr };
    std::unordered_map<std::string, std::string> metadata;

    std::vector<int> imgsz_;
    int stride_ = -1;
    int nc_ = -1;
    int ch_ = 3;

    std::unordered_map<int, std::string> names_;
    std::vector<int64_t> inputTensorShape_;
    std::string task_;

    //std::vector<float> inputTensorValues_;
    alignas(64) std::vector<float> inputTensorValues_;  // Ensure 64-byte alignment for AVX-512

    cv::Size cvSize_;
    cv::Size rawImgSize_;

    enum class SIMDSupport {
        UNINITIALIZED,
        NONE,
        AVX2,
        AVX512
    };
    SIMDSupport simd_support_ = SIMDSupport::UNINITIALIZED;

    SIMDSupport check_simd_support() {
        #if defined(_WIN32) || defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            int cpuInfo[4];

            // Check for AVX-512
            __cpuid(cpuInfo, 7);
            if ((cpuInfo[1] & (1 << 16)) != 0) {  // Check AVX-512F
                return SIMDSupport::AVX512;
            }

            // Check for AVX2
            __cpuid(cpuInfo, 1);
            if ((cpuInfo[2] & (1 << 28)) != 0) {  // Check AVX2
                return SIMDSupport::AVX2;
            }
        #endif
        return SIMDSupport::NONE;
    }

    // Helper functions, These were stolen and modified from https://github.com/FourierMourier/yolov8-onnx-cpp
    // Same with pretty much everything else

    std::vector<std::string> parseVectorString(const std::string& input) {
        std::regex number_pattern(R"(\d+)");

        std::vector<std::string> result;
        std::sregex_iterator it(input.begin(), input.end(), number_pattern);
        std::sregex_iterator end;

        while (it != end) {
            result.push_back(it->str());
            ++it;
        }

        return result;
    }

    std::vector<int> convertStringVectorToInts(const std::vector<std::string>& input) {
        std::vector<int> result;

        for (const std::string& str : input) {
            try {
                int value = std::stoi(str);
                result.push_back(value);
            }
            catch (const std::invalid_argument& e) {
                throw std::invalid_argument("Bad argument (cannot cast): value=" + str);
            }
            catch (const std::out_of_range& e) {
                throw std::out_of_range("Value out of range: " + str);
            }
        }

        return result;
    }

    std::unordered_map<int, std::string> parseNames(const std::string& input) {
        std::unordered_map<int, std::string> result;

        std::string cleanedInput = input;
        cleanedInput.erase(std::remove(cleanedInput.begin(), cleanedInput.end(), '{'), cleanedInput.end());
        cleanedInput.erase(std::remove(cleanedInput.begin(), cleanedInput.end(), '}'), cleanedInput.end());

        std::istringstream elementStream(cleanedInput);
        std::string element;
        while (std::getline(elementStream, element, ',')) {
            std::istringstream keyValueStream(element);
            std::string keyStr, value;
            if (std::getline(keyValueStream, keyStr, ':') && std::getline(keyValueStream, value)) {
                int key = std::stoi(keyStr);
                result[key] = value;
            }
        }

        return result;
    }

    void clip_boxes(cv::Rect& box, const cv::Size& shape) {
        box.x = std::max(0, std::min(box.x, shape.width));
        box.y = std::max(0, std::min(box.y, shape.height));
        box.width = std::max(0, std::min(box.width, shape.width - box.x));
        box.height = std::max(0, std::min(box.height, shape.height - box.y));
    }

    void clip_boxes(cv::Rect_<float>& box, const cv::Size& shape) {
        box.x = std::max(0.0f, std::min(box.x, static_cast<float>(shape.width)));
        box.y = std::max(0.0f, std::min(box.y, static_cast<float>(shape.height)));
        box.width = std::max(0.0f, std::min(box.width, static_cast<float>(shape.width - box.x)));
        box.height = std::max(0.0f, std::min(box.height, static_cast<float>(shape.height - box.y)));
    }


    void clip_boxes(std::vector<cv::Rect>& boxes, const cv::Size& shape) {
        for (cv::Rect& box : boxes) {
            clip_boxes(box, shape);
        }
    }

    void clip_boxes(std::vector<cv::Rect_<float>>& boxes, const cv::Size& shape) {
        for (cv::Rect_<float>& box : boxes) {
            clip_boxes(box, shape);
        }
    }

    cv::Rect_<float> scale_boxes(const cv::Size& img1_shape, cv::Rect_<float>& box, const cv::Size& img0_shape,
        std::pair<float, cv::Point2f> ratio_pad = std::make_pair(-1.0f, cv::Point2f(-1.0f, -1.0f)), bool padding = true) {

        float gain, pad_x, pad_y;

        if (ratio_pad.first < 0.0f) {
            gain = std::min(static_cast<float>(img1_shape.height) / static_cast<float>(img0_shape.height),
                static_cast<float>(img1_shape.width) / static_cast<float>(img0_shape.width));
            pad_x = roundf((img1_shape.width - img0_shape.width * gain) / 2.0f - 0.1f);
            pad_y = roundf((img1_shape.height - img0_shape.height * gain) / 2.0f - 0.1f);
        }
        else {
            gain = ratio_pad.first;
            pad_x = ratio_pad.second.x;
            pad_y = ratio_pad.second.y;
        }

        //cv::Rect scaledCoords(box);
        cv::Rect_<float> scaledCoords(box);

        if (padding) {
            scaledCoords.x -= pad_x;
            scaledCoords.y -= pad_y;
        }

        scaledCoords.x /= gain;
        scaledCoords.y /= gain;
        scaledCoords.width /= gain;
        scaledCoords.height /= gain;

        // Clip the box to the bounds of the image
        clip_boxes(scaledCoords, img0_shape);

        return scaledCoords;
    }
};