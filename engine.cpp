#include "engine.hpp"

YoloInferencer::YoloInferencer(std::wstring& modelPath, const char* logid, const char* provider)
    : env_(ORT_LOGGING_LEVEL_WARNING, logid) {

    // Set session options
    Ort::SessionOptions sessionOptions;
    if (strcmp(provider, "CUDA") == 0) {
        OrtCUDAProviderOptions cudaOption;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions);

    // Aquire input names
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    Ort::AllocatorWithDefaultOptions input_names_allocator;
    auto inputNodesNum = session_.GetInputCount();
    for (int i = 0; i < inputNodesNum; i++) {
        auto input_name = session_.GetInputNameAllocated(i, input_names_allocator);
        inputNodeNameAllocatedStrings.push_back(std::move(input_name));
        inputNames_.push_back(inputNodeNameAllocatedStrings.back().get());
    }

    // Convert input names to cstr
    for (const std::string& name : inputNames_) {
        inputNamesCStr_.push_back(name.c_str());
    }

    // Aquire output names
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    Ort::AllocatorWithDefaultOptions output_names_allocator;
    auto outputNodesNum = session_.GetOutputCount();
    for (int i = 0; i < outputNodesNum; i++)
    {
        auto output_name = session_.GetOutputNameAllocated(i, output_names_allocator);
        outputNodeNameAllocatedStrings.push_back(std::move(output_name));
        outputNames_.push_back(outputNodeNameAllocatedStrings.back().get());
    }

    // Convert output names to cstr
    for (const std::string& name : outputNames_) {
        outputNamesCStr_.push_back(name.c_str());
    }

    // Aquire model metadata
    model_metadata = session_.GetModelMetadata();

    Ort::AllocatorWithDefaultOptions metadata_allocator;
    std::vector<Ort::AllocatedStringPtr> metadataAllocatedKeys = model_metadata.GetCustomMetadataMapKeysAllocated(metadata_allocator);
    std::vector<std::string> metadata_keys;
    metadata_keys.reserve(metadataAllocatedKeys.size());

    for (const Ort::AllocatedStringPtr& allocatedString : metadataAllocatedKeys) {
        metadata_keys.emplace_back(allocatedString.get());
    }

    // Parse metadata
    for (const std::string& key : metadata_keys) {
        Ort::AllocatedStringPtr metadata_value = model_metadata.LookupCustomMetadataMapAllocated(key.c_str(), metadata_allocator);
        if (metadata_value != nullptr) {
            auto raw_metadata_value = metadata_value.get();
            metadata[key] = std::string(raw_metadata_value);
        }
    }

    for (const auto& item : metadata) {
		std::cout << item.first << ": " << item.second << std::endl;
	}

    // Find the input size of the model
    auto imgsz_item = metadata.find("imgsz");
    if (imgsz_item != metadata.end()) {
        // parse it and convert to int iterable
        std::vector<int> imgsz = convertStringVectorToInts(parseVectorString(imgsz_item->second));
        if (imgsz_.empty()) {
            imgsz_ = imgsz;
        }
    }
    else {
        std::cerr << "Warning: Cannot get imgsz value from metadata" << std::endl;
    }

    // For yolo this is normally 32 but get it anyway
    auto stride_item = metadata.find("stride");
    if (stride_item != metadata.end()) {
        // parse it and convert to int iterable
        int stride = std::stoi(stride_item->second);
        if (stride_ == -1) {
            stride_ = stride;
        }
    }
    else {
        std::cerr << "Warning: Cannot get stride value from metadata" << std::endl;
    }

    // For the names of the classes
    auto names_item = metadata.find("names");
    if (names_item != metadata.end()) {
        // parse it and convert to int iterable
        std::unordered_map<int, std::string> names = parseNames(names_item->second);
        std::cout << "***Names from metadata***" << std::endl;
        for (const auto& pair : names) {
            std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
        }
        // set it here:
        if (names_.empty()) {
            names_ = names;
        }
    }
    else {
        std::cerr << "Warning: Cannot get names value from metadata" << std::endl;
    }

    // Determine the task (We want detect)
    auto task_item = metadata.find("task");
    if (task_item != metadata.end()) {
        std::string task = std::string(task_item->second);

        if (task_.empty()) {
            task_ = task;
        }
    }
    else {
        std::cerr << "Warning: Cannot get task value from metadata" << std::endl;
    }

    // Aquire number of classes
    if (nc_ == -1 && names_.size() > 0) {
        nc_ = names_.size();
    }
    else {
        std::cerr << "Warning: Cannot get nc value from metadata (probably names wasn't set)" << std::endl;
    }

    // Setup the desired input shape
    if (!imgsz_.empty() && inputTensorShape_.empty())
    {
        inputTensorShape_ = { 1, ch_, imgsz_[0], imgsz_[1] };
    }

    // Setup the CV resizer
    if (!imgsz_.empty())
    {
        cvSize_ = cv::Size(imgsz_[1], imgsz_[0]);
    }
}

// Destructor for the class
YoloInferencer::~YoloInferencer() {
    // The Ort::Session and other Ort:: objects will automatically release resources upon destruction
    // due to their RAII design.
}

// This function does the preprocessing of the image and returns the tensor
std::vector<Ort::Value> YoloInferencer::preprocess(cv::Mat& frame) {
    // Check SIMD support once
    if (simd_support_ == SIMDSupport::UNINITIALIZED) {
        simd_support_ = check_simd_support();
    }

    if (simd_support_ == SIMDSupport::NONE) {
		std::cout << "Warning: SIMD support is not available on this platform." << std::endl;
	} else if (simd_support_ == SIMDSupport::AVX512) {
		std::cout << "AVX512 support is available on this platform." << std::endl;
	} else if (simd_support_ == SIMDSupport::AVX2) {
		std::cout << "AVX2 support is available on this platform." << std::endl;
	}

    rawImgSize_ = frame.size();

    // Calculate scaling
    float r = std::min(static_cast<float>(cvSize_.height) / static_cast<float>(rawImgSize_.height),
        static_cast<float>(cvSize_.width) / static_cast<float>(rawImgSize_.width));

    int new_width = static_cast<int>(std::round(static_cast<float>(rawImgSize_.width) * r));
    int new_height = static_cast<int>(std::round(static_cast<float>(rawImgSize_.height) * r));

    // Calculate padding once
    int pad_w = cvSize_.width - new_width;
    int pad_h = cvSize_.height - new_height;
    int top = pad_h / 2;
    int left = pad_w / 2;

    // Resize with optimized flags
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR + cv::INTER_LINEAR_EXACT);

    // Create padded image
    cv::Mat padded = cv::Mat(cvSize_.height, cvSize_.width, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(left, top, new_width, new_height)));

    // Prepare output tensor with 64-byte alignment for AVX-512
    size_t image_size = cvSize_.width * cvSize_.height;
    size_t total_size = image_size * 3;
    if (inputTensorValues_.size() < total_size) {
        inputTensorValues_.resize(total_size);
    }

    const uint8_t* src = padded.ptr<uint8_t>();
    float* dst = inputTensorValues_.data();

    if (simd_support_ == SIMDSupport::AVX512) {
        // AVX-512 implementation
        const __m512 scale = _mm512_set1_ps(1.0f / 255.0f);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < image_size; i += 16) {
            if (i + 16 <= image_size) {
                // Pre-gather RGB values for 16 pixels
                alignas(64) uint8_t r_bytes[16];
                alignas(64) uint8_t g_bytes[16];
                alignas(64) uint8_t b_bytes[16];

                // Gather with unrolled loop for better instruction pipelining
#pragma unroll
                for (int j = 0; j < 16; j++) {
                    const uint8_t* pixel = src + (i + j) * 3;
                    b_bytes[j] = pixel[0];
                    g_bytes[j] = pixel[1];
                    r_bytes[j] = pixel[2];
                }

                // Convert uint8 to int32 using AVX-512
                __m512i r_val = _mm512_cvtepu8_epi32(_mm_load_si128((__m128i*)r_bytes));
                __m512i g_val = _mm512_cvtepu8_epi32(_mm_load_si128((__m128i*)g_bytes));
                __m512i b_val = _mm512_cvtepu8_epi32(_mm_load_si128((__m128i*)b_bytes));

                // Convert to float and normalize using FMA
                __m512 r_float = _mm512_mul_ps(_mm512_cvtepi32_ps(r_val), scale);
                __m512 g_float = _mm512_mul_ps(_mm512_cvtepi32_ps(g_val), scale);
                __m512 b_float = _mm512_mul_ps(_mm512_cvtepi32_ps(b_val), scale);

                // Store aligned
                _mm512_store_ps(dst + i, r_float);
                _mm512_store_ps(dst + image_size + i, g_float);
                _mm512_store_ps(dst + 2 * image_size + i, b_float);
            }
            else {
                // Handle edge cases
                for (int j = i; j < image_size; j++) {
                    const uint8_t* pixel = src + j * 3;
                    dst[j] = pixel[2] / 255.0f;
                    dst[j + image_size] = pixel[1] / 255.0f;
                    dst[j + 2 * image_size] = pixel[0] / 255.0f;
                }
            }
        }
    }
    else if (simd_support_ == SIMDSupport::AVX2) {
        // Fallback to AVX2 for systems without AVX-512
        const __m256 scale = _mm256_set1_ps(1.0f / 255.0f);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < image_size; i += 32) {
            if (i + 32 <= image_size) {
                // Process in chunks of 32 pixels (4 AVX2 registers)
#pragma unroll
                for (int chunk = 0; chunk < 4; chunk++) {
                    int base_idx = i + chunk * 8;

                    alignas(32) uint8_t r_bytes[8];
                    alignas(32) uint8_t g_bytes[8];
                    alignas(32) uint8_t b_bytes[8];

#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const uint8_t* pixel = src + (base_idx + j) * 3;
                        b_bytes[j] = pixel[0];
                        g_bytes[j] = pixel[1];
                        r_bytes[j] = pixel[2];
                    }

                    __m256i r_val = _mm256_cvtepu8_epi32(_mm_load_si128((__m128i*)r_bytes));
                    __m256i g_val = _mm256_cvtepu8_epi32(_mm_load_si128((__m128i*)g_bytes));
                    __m256i b_val = _mm256_cvtepu8_epi32(_mm_load_si128((__m128i*)b_bytes));

                    __m256 r_float = _mm256_mul_ps(_mm256_cvtepi32_ps(r_val), scale);
                    __m256 g_float = _mm256_mul_ps(_mm256_cvtepi32_ps(g_val), scale);
                    __m256 b_float = _mm256_mul_ps(_mm256_cvtepi32_ps(b_val), scale);

                    _mm256_store_ps(dst + base_idx, r_float);
                    _mm256_store_ps(dst + image_size + base_idx, g_float);
                    _mm256_store_ps(dst + 2 * image_size + base_idx, b_float);
                }
            }
            else {
                for (int j = i; j < image_size; j++) {
                    const uint8_t* pixel = src + j * 3;
                    dst[j] = pixel[2] / 255.0f;
                    dst[j + image_size] = pixel[1] / 255.0f;
                    dst[j + 2 * image_size] = pixel[0] / 255.0f;
                }
            }
        }
    }
    else {
        // Fallback path for non-x86 or CPUs without AVX
#pragma omp parallel for schedule(static) collapse(2)
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < image_size; i++) {
                const uint8_t* pixel = src + i * 3;
                float* channel_ptr = dst + c * image_size;
                // RGB order: R=2, G=1, B=0
                channel_ptr[i] = pixel[2 - c] / 255.0f;
            }
        }
    }

    // Create tensor
    std::vector<Ort::Value> inputTensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues_.data(), total_size,
        inputTensorShape_.data(), inputTensorShape_.size()));

    return inputTensors;
}

// This function does the forward pass and returns the tensor
std::vector<Ort::Value> YoloInferencer::forward(std::vector<Ort::Value>& inputTensors) {
    return session_.Run(Ort::RunOptions{ nullptr }, inputNamesCStr_.data(), inputTensors.data(), inputNamesCStr_.size(), outputNamesCStr_.data(), outputNamesCStr_.size());
}

// This function does the postprocessing of the output and returns the detections
std::vector<Detection> YoloInferencer::postprocess(std::vector<Ort::Value>& outputTensors, float conf_threshold, float iou_threshold) {
    // Get raw pointer to output tensor data
    float* data = outputTensors[0].GetTensorMutableData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

    /*
    * Output tensor shape: [1, detection_attribute_size, num_detections]
    * where:
    * - First dimension (1) is the batch size
    * - Second dimension (detection_attribute_size) is 4 + num_classes
    *   - First 4 values are box coordinates (cx, cy, w, h)
    *   - Remaining values are class probabilities
    * - Third dimension (num_detections) is the number of detected objects
    *
    * Memory layout is column-major for dimension 2 and 3, meaning:
    * - Data for each attribute is stored contiguously for all detections
    * - To access attribute 'a' for detection 'i': data[i + a * num_detections]
    */

    int detection_attribute_size = outputShape[1];
    int num_detections = outputShape[2];
    int num_classes = detection_attribute_size - 4;

    // Pre-calculate array offsets for box coordinates
    const int x_offset = 0 * num_detections;
    const int y_offset = 1 * num_detections;
    const int w_offset = 2 * num_detections;
    const int h_offset = 3 * num_detections;

    // Reserve space for detections to avoid reallocations
    std::vector<Detection> detections;
    detections.reserve(num_detections);

    // Process each detection
    for (int i = 0; i < num_detections; ++i) {
        // Find highest confidence class for this detection
        float max_conf = 0.0f;
        int best_class = -1;

        // Starting offset for class probabilities
        const float* class_ptr = data + i + 4 * num_detections;

        // Find maximum confidence and corresponding class
        for (int c = 0; c < num_classes; c++) {
            float conf = class_ptr[c * num_detections];
            if (conf > max_conf) {
                max_conf = conf;
                best_class = c;
            }
        }

        // Filter low confidence detections
        if (max_conf <= conf_threshold) continue;

        // Get bounding box coordinates
        float cx = data[i + x_offset];
        float cy = data[i + y_offset];
        float w = data[i + w_offset];
        float h = data[i + h_offset];

        // Convert from center format (cx, cy, w, h) to corner format (x1, y1, w, h)
        float x1 = cx - w / 2;
        float y1 = cy - h / 2;

        // Create and store detection
        Detection detection;
        detection.class_id = best_class;
        detection.confidence = max_conf;

        // Create bounding box and scale to original image size
        cv::Rect_<float> bbox(x1, y1, w, h);
        detection.box = scale_boxes(cvSize_, bbox, rawImgSize_);
        detections.push_back(detection);
    }

    // Skip NMS if no detections
    if (detections.empty()) return detections;

    // Prepare data for NMS
    std::vector<cv::Rect> boxes;  // Using integer Rect for NMS
    std::vector<float> scores;
    boxes.reserve(detections.size());
    scores.reserve(detections.size());

    // Convert float rectangles to integer rectangles for NMS
    for (const auto& det : detections) {
        boxes.push_back(cv::Rect(
            static_cast<int>(std::round(det.box.x)),
            static_cast<int>(std::round(det.box.y)),
            static_cast<int>(std::round(det.box.width)),
            static_cast<int>(std::round(det.box.height))
        ));
        scores.push_back(det.confidence);
    }

    // Perform Non-Maximum Suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, indices);

    // Create final detection list
    std::vector<Detection> nms_detections;
    nms_detections.reserve(indices.size());
    for (int idx : indices) {
        nms_detections.push_back(detections[idx]);
    }

    return nms_detections;
}

// This function does the whole inference, and is publicly accessible, acting as a main
std::vector<Detection> YoloInferencer::infer(cv::Mat& frame, float conf_threshold, float iou_threshold) {
    auto total_start = std::chrono::high_resolution_clock::now();

    // Profile preprocessing
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    std::vector<Ort::Value> inputTensors = preprocess(frame);
    auto preprocess_end = std::chrono::high_resolution_clock::now();

    // Profile forward pass
    auto forward_start = std::chrono::high_resolution_clock::now();
    std::vector<Ort::Value> outputTensors = forward(inputTensors);
    auto forward_end = std::chrono::high_resolution_clock::now();

    // Profile postprocessing
    auto postprocess_start = std::chrono::high_resolution_clock::now();
    std::vector<Detection> detections = postprocess(outputTensors, conf_threshold, iou_threshold);
    auto postprocess_end = std::chrono::high_resolution_clock::now();

    auto total_end = std::chrono::high_resolution_clock::now();

    // Calculate durations in milliseconds
    float preprocess_time = std::chrono::duration<float, std::milli>(preprocess_end - preprocess_start).count();
    float forward_time = std::chrono::duration<float, std::milli>(forward_end - forward_start).count();
    float postprocess_time = std::chrono::duration<float, std::milli>(postprocess_end - postprocess_start).count();
    float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();

    // Log the timing information
    std::cout << "\nProfiling Results:" << std::endl;
    std::cout << "Preprocessing : " << preprocess_time << " ms" << std::endl;
    std::cout << "Forward Pass  : " << forward_time << " ms" << std::endl;
    std::cout << "Postprocessing: " << postprocess_time << " ms" << std::endl;
    std::cout << "Total Time    : " << total_time << " ms\n" << std::endl;

    return detections;
}