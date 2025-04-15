#include "AdianceFRVT.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <vector>
#include <cstring>
#include <cfloat>
#include <regex>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <filesystem>
namespace fs = std::filesystem;

// Force single-threaded operation for all OpenCV operations
struct InitOpenCV {
    InitOpenCV() {
        cv::setNumThreads(0);
    }
};
static InitOpenCV initOpenCV;

bool g_debug_output = false;

const int MAX_DEBUG_FACES = 30;   
bool g_save_visualization = true; 

namespace FRVT_1N
{
    // ==============================
    // RetinafaceWrapper
    // This class wraps the RetinaFace model that outputs aligned face, landmarks, etc.
    class RetinafaceWrapper
    {
    public:
        struct FaceInfo
        {
            float score;
            cv::Mat aligned_face;
            std::vector<cv::Point2f> landmarks;         // Original image landmarks
            std::vector<cv::Point2f> aligned_landmarks; // Landmarks mapped to aligned face
        };

        RetinafaceWrapper(const std::string &model_path)
            : env_(ORT_LOGGING_LEVEL_WARNING, "retinaface"), session_(nullptr)
        {
            cv::setNumThreads(0); // Disable OpenCV multithreading
            
            Ort::SessionOptions options;
            options.SetIntraOpNumThreads(1);
            options.SetInterOpNumThreads(1);
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            options.DisableMemPattern();
            options.SetExecutionMode(ORT_SEQUENTIAL);
            options.DisableCpuMemArena();

#ifdef _WIN32
            std::wstring w_model(model_path.begin(), model_path.end());
            session_ = Ort::Session(env_, w_model.c_str(), options);
#else
            session_ = Ort::Session(env_, model_path.c_str(), options);
#endif

            // Load input names
            for (size_t i = 0; i < session_.GetInputCount(); ++i)
            {
                auto name = session_.GetInputNameAllocated(i, allocator_);
                input_names_.push_back(name.get());
                input_name_ptrs_.push_back(std::move(name));
            }

            // Load output names
            for (size_t i = 0; i < session_.GetOutputCount(); ++i)
            {
                auto name = session_.GetOutputNameAllocated(i, allocator_);
                output_names_.push_back(name.get());
                output_name_ptrs_.push_back(std::move(name));
            }
        }

        // Reference landmarks for alignment
        const std::vector<cv::Point2f> referenceAlignment = {
            cv::Point2f(38.2946f, 51.6963f),
            cv::Point2f(73.5318f, 51.5014f),
            cv::Point2f(56.0252f, 71.7366f),
            cv::Point2f(41.5493f, 92.3655f),
            cv::Point2f(70.7299f, 92.2041f)};

        // Constants for RetinaFace
        const cv::Size inputSize{640, 640}; // Reduced input size for speed improvement
        const float confThresh = 0.5f;
        const float nmsThresh = 0.4f;
        const std::array<float, 2> variances = {0.1f, 0.2f};

        // Anchor cache for optimization
        struct SizeCompare {
            bool operator()(const cv::Size& a, const cv::Size& b) const {
                return a.width < b.width || (a.width == b.width && a.height < b.height);
            }
        };
        std::map<cv::Size, std::vector<std::array<float, 4>>, SizeCompare> anchorCache;

        // Resize image with aspect ratio preservation and padding
        cv::Mat resizeImage(const cv::Mat &frame, const cv::Size &targetShape, float &resizeFactor)
        {
            int orig_h = frame.rows;
            int orig_w = frame.cols;
            int target_w = targetShape.width;
            int target_h = targetShape.height;
            float im_ratio = static_cast<float>(orig_h) / static_cast<float>(orig_w);
            float model_ratio = static_cast<float>(target_h) / static_cast<float>(target_w);
            int new_w = 0, new_h = 0;
            if (im_ratio > model_ratio)
            {
                new_h = target_h;
                new_w = static_cast<int>(static_cast<float>(new_h) / im_ratio);
            }
            else
            {
                new_w = target_w;
                new_h = static_cast<int>(static_cast<float>(new_w) * im_ratio);
            }
            resizeFactor = static_cast<float>(new_h) / static_cast<float>(orig_h);
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(new_w, new_h));
            cv::Mat output = cv::Mat::zeros(targetShape, frame.type());
            resized.copyTo(output(cv::Rect(0, 0, new_w, new_h)));
            return output;
        }

        // Generate anchor boxes (priors) based on the input image size
        std::vector<std::array<float, 4>> generateAnchors(const cv::Size &imageSize)
        {
            // Check cache first
            auto it = anchorCache.find(imageSize);
            if (it != anchorCache.end()) {
                return it->second;
            }

            std::vector<std::array<float, 4>> anchors;
            std::vector<int> steps = {8, 16, 32};
            std::vector<std::vector<int>> minSizes = {{16, 32}, {64, 128}, {256, 512}};

            for (size_t k = 0; k < steps.size(); k++)
            {
                int step = steps[k];
                int fm_h = static_cast<int>(std::ceil(static_cast<float>(imageSize.height) / step));
                int fm_w = static_cast<int>(std::ceil(static_cast<float>(imageSize.width) / step));
                for (int i = 0; i < fm_h; i++)
                {
                    for (int j = 0; j < fm_w; j++)
                    {
                        for (int min_size : minSizes[k])
                        {
                            float s_kx = static_cast<float>(min_size) / imageSize.width;
                            float s_ky = static_cast<float>(min_size) / imageSize.height;
                            float cx = (j + 0.5f) * step / imageSize.width;
                            float cy = (i + 0.5f) * step / imageSize.height;
                            anchors.push_back({cx, cy, s_kx, s_ky});
                        }
                    }
                }
            }

            // Cache and return result
            anchorCache[imageSize] = anchors;
            return anchors;
        }

        // Decode bounding boxes from model predictions
        std::vector<std::array<float, 4>> decodeBoxes(const std::vector<float> &loc,
                                                      const std::vector<std::array<float, 4>> &priors)
        {
            std::vector<std::array<float, 4>> boxes;
            size_t num = priors.size();
            for (size_t i = 0; i < num; i++)
            {
                float loc0 = loc[i * 4];
                float loc1 = loc[i * 4 + 1];
                float loc2 = loc[i * 4 + 2];
                float loc3 = loc[i * 4 + 3];
                float prior_cx = priors[i][0];
                float prior_cy = priors[i][1];
                float prior_w = priors[i][2];
                float prior_h = priors[i][3];
                float cx = prior_cx + loc0 * variances[0] * prior_w;
                float cy = prior_cy + loc1 * variances[0] * prior_h;
                float w = prior_w * std::exp(loc2 * variances[1]);
                float h = prior_h * std::exp(loc3 * variances[1]);
                float xmin = cx - w / 2;
                float ymin = cy - h / 2;
                float xmax = cx + w / 2;
                float ymax = cy + h / 2;
                boxes.push_back({xmin, ymin, xmax, ymax});
            }
            return boxes;
        }

        // Decode landmarks from model predictions
        std::vector<std::array<float, 10>> decodeLandmarks(const std::vector<float> &landms,
                                                           const std::vector<std::array<float, 4>> &priors)
        {
            std::vector<std::array<float, 10>> landmarks;
            size_t num = priors.size();
            for (size_t i = 0; i < num; i++)
            {
                std::array<float, 10> landmark;
                for (int j = 0; j < 5; j++)
                {
                    float lx = landms[i * 10 + j * 2];
                    float ly = landms[i * 10 + j * 2 + 1];
                    float prior_cx = priors[i][0];
                    float prior_cy = priors[i][1];
                    float prior_w = priors[i][2];
                    float prior_h = priors[i][3];
                    landmark[j * 2] = prior_cx + lx * variances[0] * prior_w;
                    landmark[j * 2 + 1] = prior_cy + ly * variances[0] * prior_h;
                }
                landmarks.push_back(landmark);
            }
            return landmarks;
        }

        // Compute Intersection over Union (IoU) for two detections
        float IoU(const std::array<float, 5> &a, const std::array<float, 5> &b)
        {
            float x1 = std::max(a[0], b[0]);
            float y1 = std::max(a[1], b[1]);
            float x2 = std::min(a[2], b[2]);
            float y2 = std::min(a[3], b[3]);
            float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float area_a = (a[2] - a[0]) * (a[3] - a[1]);
            float area_b = (b[2] - b[0]) * (b[3] - b[1]);
            return inter_area / (area_a + area_b - inter_area);
        }

        // Non-Maximum Suppression (NMS)
        std::vector<int> nms(const std::vector<std::array<float, 5>> &dets, float threshold)
        {
            std::vector<int> indices(dets.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(), [&](int i1, int i2)
                      { return dets[i1][4] > dets[i2][4]; });
            std::vector<int> keep;
            while (!indices.empty())
            {
                int current = indices[0];
                keep.push_back(current);
                std::vector<int> new_indices;
                for (size_t i = 1; i < indices.size(); i++)
                {
                    int idx = indices[i];
                    if (IoU(dets[current], dets[idx]) <= threshold)
                    {
                        new_indices.push_back(idx);
                    }
                }
                indices = new_indices;
            }
            return keep;
        }

        // Face alignment function using the custom similarity transform
        cv::Mat faceAlignment(const cv::Mat &image, const std::vector<cv::Point2f> &landmarks,
                              std::vector<cv::Point2f> &alignedLandmarks, int imageSize = 128)
        {
            // Compute ratio and x-offset
            float ratio = (imageSize % 112 == 0) ? (static_cast<float>(imageSize) / 112.0f)
                                                 : (static_cast<float>(imageSize) / 128.0f);
            float diff_x = (imageSize % 112 == 0) ? 0.0f : (8.0f * ratio);

            // Create the destination template by scaling and shifting the reference landmarks
            std::vector<cv::Point2f> dst;
            for (const auto &pt : referenceAlignment)
            {
                dst.push_back(cv::Point2f(pt.x * ratio + diff_x, pt.y * ratio));
            }

            // Use OpenCV's built-in estimateAffinePartial2D
            cv::Mat transform = cv::estimateAffinePartial2D(landmarks, dst);

            // If transformation estimation fails, return original image
            if (transform.empty())
            {
                cv::Mat resized;
                cv::resize(image, resized, cv::Size(imageSize, imageSize));
                return resized;
            }

            // Transform the original landmarks to the aligned face space
            alignedLandmarks = dst; // Use reference landmarks as aligned landmarks

            // Apply the transformation
            cv::Mat aligned;
            cv::warpAffine(image, aligned, transform, cv::Size(imageSize, imageSize),
                           cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            return aligned;
        }

        // Preprocess image for ONNX input
        std::vector<float> preprocessForOnnx(const cv::Mat &image)
        {
            // Use preallocated buffers
            static thread_local cv::Mat floatImg;
            static thread_local std::vector<float> inputTensorValues;

            image.convertTo(floatImg, CV_32F);
            cv::subtract(floatImg, cv::Scalar(104, 117, 123), floatImg);

            // Resize vector once
            inputTensorValues.resize(image.total() * 3);

            // More efficient channel extraction
            const float* imgData = floatImg.ptr<float>();
            const int imgStep = floatImg.step1();
            const int imgArea = image.rows * image.cols;

            // Extract channels directly to input tensor
            for (int c = 0; c < 3; c++) {
                for (int i = 0; i < image.rows; i++) {
                    for (int j = 0; j < image.cols; j++) {
                        inputTensorValues[c * imgArea + i * image.cols + j] = 
                            imgData[i * imgStep + j * 3 + c];
                    }
                }
            }

            return inputTensorValues;
        }

        // Runs inference on the input image and returns detected FaceInfo(s)
        std::vector<FaceInfo> detect(const cv::Mat &image)
        {
            std::vector<FaceInfo> faces;
            if (image.empty())
                return faces;

            try
            {
                // Check image size and add protection
                if (image.cols <= 0 || image.rows <= 0 || image.cols > 4000 || image.rows > 4000)
                {
                    return faces;
                }

                // Limit memory usage by resizing extremely large images
                cv::Mat processImage = image;
                if (image.cols * image.rows > 1920 * 1080)
                { // If image is larger than Full HD
                    float scale = sqrt(1920.0 * 1080.0 / (image.cols * image.rows));
                    cv::resize(image, processImage, cv::Size(), scale, scale, cv::INTER_AREA);
                }

                float resizeFactor = 1.0f;
                cv::Mat resizedImage = resizeImage(processImage, inputSize, resizeFactor);

                // Preprocess for ONNX
                std::vector<float> inputTensorValues = preprocessForOnnx(resizedImage);
                std::vector<int64_t> inputDims = {1, 3, inputSize.height, inputSize.width};

                // Create input tensor
                auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                    mem_info, inputTensorValues.data(), inputTensorValues.size(),
                    inputDims.data(), inputDims.size());

                // Run inference
                std::vector<Ort::Value> outputs;
                try
                {
                    outputs = session_.Run(Ort::RunOptions{nullptr},
                                           input_names_.data(), &inputTensor, 1,
                                           output_names_.data(), output_names_.size());
                }
                catch (const Ort::Exception &e)
                {
                    return faces;
                }

                // Expected outputs: [0] loc, [1] conf, [2] landms
                if (outputs.size() < 3)
                    return faces;

                // Process output tensors
                float *locData = outputs[0].GetTensorMutableData<float>();
                auto locShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
                int num_priors = static_cast<int>(locShape[1]); // shape: [1, num_priors, 4]
                std::vector<float> loc(locData, locData + num_priors * 4);

                float *confData = outputs[1].GetTensorMutableData<float>();
                std::vector<float> conf(confData, confData + num_priors * 2);

                float *landmsData = outputs[2].GetTensorMutableData<float>();
                std::vector<float> landms(landmsData, landmsData + num_priors * 10);

                // Generate anchors
                std::vector<std::array<float, 4>> anchors = generateAnchors(inputSize);

                // Decode boxes and landmarks
                std::vector<std::array<float, 4>> boxes = decodeBoxes(loc, anchors);
                std::vector<std::array<float, 10>> landmarks_all = decodeLandmarks(landms, anchors);

                // Filter detections by confidence and prepare for NMS
                std::vector<std::array<float, 5>> detections; // [xmin, ymin, xmax, ymax, score]
                std::vector<std::array<float, 10>> filteredLandmarks;

                for (int i = 0; i < num_priors; i++)
                {
                    float score = conf[i * 2 + 1]; // face class score
                    if (score > confThresh)
                    {
                        // Scale box from normalized [0,1] to image coordinates
                        std::array<float, 4> box = boxes[i];
                        box[0] = box[0] * inputSize.width / resizeFactor;
                        box[1] = box[1] * inputSize.height / resizeFactor;
                        box[2] = box[2] * inputSize.width / resizeFactor;
                        box[3] = box[3] * inputSize.height / resizeFactor;
                        detections.push_back({box[0], box[1], box[2], box[3], score});

                        // Scale landmarks similarly
                        std::array<float, 10> lm = landmarks_all[i];
                        for (int j = 0; j < 5; j++)
                        {
                            lm[j * 2] = lm[j * 2] * inputSize.width / resizeFactor;
                            lm[j * 2 + 1] = lm[j * 2 + 1] * inputSize.height / resizeFactor;
                        }
                        filteredLandmarks.push_back(lm);
                    }
                }

                // Apply Non-Maximum Suppression
                std::vector<int> keep = nms(detections, nmsThresh);

                // Process detected faces
                for (size_t i = 0; i < keep.size(); i++)
                {
                    int idx = keep[i];
                    FaceInfo face;
                    face.score = detections[idx][4];

                    // Convert landmarks array to vector of points
                    std::array<float, 10> lm = filteredLandmarks[idx];
                    for (int j = 0; j < 5; j++)
                    {
                        face.landmarks.push_back(cv::Point2f(lm[j * 2], lm[j * 2 + 1]));
                    }

                    // Align the face using landmarks
                    std::vector<cv::Point2f> aligned_lm;
                    face.aligned_face = faceAlignment(image, face.landmarks, aligned_lm, 128);
                    face.aligned_landmarks = aligned_lm;

                    faces.push_back(face);
                }

                return faces;
            }
            catch (const Ort::Exception &e)
            {

                // Try a more conservative approach with smaller input
                try
                {
                    cv::Mat smallerImage;
                    cv::resize(image, smallerImage, cv::Size(320, 320));
                    float resizeFactor = 320.0f / std::max(image.cols, image.rows);

                    // ... proceed with detection on smaller image ...
                    // (This would be a simplified fallback implementation)

                    // For now, just return empty result instead of crashing
                    return faces;
                }
                catch (...)
                {
                    return faces;
                }
            }
            catch (const std::exception &e)
            {
                return faces;
            }
            catch (...)
            {
                return faces;
            }
        }

    private:
        Ort::Env env_;
        Ort::Session session_;
        Ort::AllocatorWithDefaultOptions allocator_;
        std::vector<const char *> input_names_;
        std::vector<const char *> output_names_;
        std::vector<Ort::AllocatedStringPtr> input_name_ptrs_;
        std::vector<Ort::AllocatedStringPtr> output_name_ptrs_;
    };

    // ==============================
    // Global Helper Functions

    namespace
    {

        void l2_normalize(std::vector<float> &feat)
        {
            float sum_sq = 0.0f;
            for (float v : feat)
                sum_sq += v * v;
            float norm = std::sqrt(sum_sq);
            if (norm > 0)
            {
                for (auto &v : feat)
                    v /= norm;
            }
        }

        std::vector<float> fuseFeaturesImpl(const std::vector<std::vector<float>> &feats)
        {
            if (feats.empty())
                return {};
            size_t feature_size = feats[0].size();
            std::vector<float> fused(feature_size, 0.0f);
            for (const auto &feat : feats)
            {
                for (size_t i = 0; i < feature_size; i++)
                    fused[i] += feat[i];
            }
            for (auto &v : fused)
                v /= feats.size();
            l2_normalize(fused);
            return fused;
        }

        float cosineSimilarityImpl(const std::vector<float> &a, const std::vector<float> &b)
        {
            if (a.empty() || b.empty() || a.size() != b.size())
            {
                return 0.0f;
            }

            float dot_product = 0.0f;
            float norm_a = 0.0f, norm_b = 0.0f;

            for (size_t i = 0; i < a.size(); i++)
            {
                dot_product += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];
            }

            if (norm_a <= 0.0f || norm_b <= 0.0f)
            {
                return 0.0f;
            }

            return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
        }
    } // end anonymous namespace

    cv::Mat AdianceFRVT::imageToMat(const FRVT::Image &image)
    {
        int channels = (image.depth == 24) ? 3 : 1;
        cv::Mat mat(image.height, image.width, CV_MAKETYPE(CV_8U, channels), image.data.get());
        return mat.clone();
    }

    std::vector<float> AdianceFRVT::extractFeatures(const cv::Mat &alignedFace, bool flipTest)
    {
        // Use static thread-local memory for better performance
        static thread_local cv::Mat resized(112, 112, CV_8UC3);
        static thread_local cv::Mat normalized(112, 112, CV_32FC3);
        static thread_local std::vector<float> inputTensor(3 * 112 * 112);
        
        // Resize to 112x112 if needed (match Python logic)
        if (alignedFace.rows != 112 || alignedFace.cols != 112) {
            cv::resize(alignedFace, resized, cv::Size(112, 112));
        } else {
            alignedFace.copyTo(resized);
        }
        
        // Convert BGR to RGB (match Python cv2.cvtColor)
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        
        // Match Python normalization exactly:
        // img = (img / 255.0 - mean) / std, where mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
        // This simplifies to: img = img/127.5 - 1
        resized.convertTo(normalized, CV_32F, 1.0/127.5, -1.0);
        
        // Reorganize to NCHW format (match Python transpose(2,0,1))
        float* inputData = inputTensor.data();
        const int planeSize = 112 * 112;
        
        for (int h = 0; h < 112; h++) {
            const float* rowPtr = normalized.ptr<float>(h);
            for (int w = 0; w < 112; w++) {
                for (int c = 0; c < 3; c++) {
                    inputData[c * planeSize + h * 112 + w] = rowPtr[w * 3 + c];
                }
            }
        }
        
        // Run inference
        std::vector<int64_t> inputDims = {1, 3, 112, 112};
        std::vector<float> features = adianceModel_->run(inputTensor, inputDims);
        
        // If flip test is enabled, add that logic
        if (flipTest) {
            // Create flipped version of the face
            cv::Mat flippedFace;
            cv::flip(alignedFace, flippedFace, 1); // 1 = horizontal flip
            
            // Extract features from flipped face
            cv::Mat flippedResized(112, 112, CV_8UC3);
            cv::Mat flippedNormalized(112, 112, CV_32FC3);
            std::vector<float> flippedInputTensor(3 * 112 * 112);
            
            cv::resize(flippedFace, flippedResized, cv::Size(112, 112));
            cv::cvtColor(flippedResized, flippedResized, cv::COLOR_BGR2RGB);
            flippedResized.convertTo(flippedNormalized, CV_32F, 1.0/127.5, -1.0);
            
            float* flippedInputData = flippedInputTensor.data();
            for (int h = 0; h < 112; h++) {
                const float* rowPtr = flippedNormalized.ptr<float>(h);
                for (int w = 0; w < 112; w++) {
                    for (int c = 0; c < 3; c++) {
                        flippedInputData[c * planeSize + h * 112 + w] = rowPtr[w * 3 + c];
                    }
                }
            }
            
            std::vector<float> flippedFeatures = adianceModel_->run(flippedInputTensor, inputDims);
            
            // Average the original features with the flipped features
            for (size_t i = 0; i < features.size(); i++) {
                features[i] = (features[i] + flippedFeatures[i]) / 2.0f;
            }
        }
        
        // L2 normalize (match Python embedding = embedding / np.linalg.norm(embedding))
        float norm = 0.0f;
        for (const auto& val : features) {
            norm += val * val;
        }
        
        if (norm > 0) {
            norm = 1.0f / std::sqrt(norm);
            for (auto& val : features) {
                val *= norm;
            }
        }
        
        return features;
    }

    std::vector<float> AdianceFRVT::fuseFeatures(const std::vector<std::vector<float>> &features)
    {
        return fuseFeaturesImpl(features);
    }

    // The output size for the model (128x128 instead of 112x112).
    const cv::Size AdianceFRVT::kOutputSize(128, 128);
    // Five-point destination template.
    const std::vector<cv::Point2f> AdianceFRVT::kDstLandmarks = {
        cv::Point2f(89.3f * 128.0f / 112.0f, 90.9f * 128.0f / 112.0f),
        cv::Point2f(137.5f * 128.0f / 112.0f, 91.0f * 128.0f / 112.0f),
        cv::Point2f(114.8f * 128.0f / 112.0f, 125.8f * 128.0f / 112.0f),
        cv::Point2f(96.0f * 128.0f / 112.0f, 154.1f * 128.0f / 112.0f),
        cv::Point2f(133.4f * 128.0f / 112.0f, 153.6f * 128.0f / 112.0f)};

    std::shared_ptr<Interface> Interface::getImplementation()
    {
        return std::make_shared<AdianceFRVT>("", "");
    }

    // ----- AdianceFRVT Constructor & Destructor -----
    AdianceFRVT::AdianceFRVT(const std::string &adianceModelPath, const std::string &mtcnnModelPath)
        : adianceModelPath_(adianceModelPath), mtcnnModelPath_(mtcnnModelPath)
    {
        cv::setNumThreads(0); // Disable OpenCV multithreading
        
        if (!adianceModelPath_.empty() && !mtcnnModelPath_.empty())
        {
            adianceModel_ = std::make_shared<OnnxModelWrapper>(adianceModelPath_);
        }
    }

    AdianceFRVT::~AdianceFRVT() {}

    ReturnStatus AdianceFRVT::initializeTemplateCreation(const std::string &configDir, TemplateRole role)
    {
        cv::setNumThreads(0); // Disable OpenCV multithreading
        
        configDir_ = configDir;
        std::string confPath = configDir + "/adiance.conf";
        std::ifstream fin(confPath);
        if (!fin)
            return ReturnStatus(ReturnCode::ConfigError, "Failed to open config file: " + confPath);
        std::string line;
        while (std::getline(fin, line))
        {
            if (line.empty() || line[0] == '#')
                continue;
            auto pos = line.find('=');
            if (pos == std::string::npos)
                continue;
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            if (key == "ADIANCE_MODEL_PATH")
                adianceModelPath_ = configDir + "/" + value; // Prepend configDir
            else if (key == "MTCNN_MODEL_PATH")
                mtcnnModelPath_ = configDir + "/" + value; // Prepend configDir
        }
        fin.close();

        if (adianceModelPath_.empty() || mtcnnModelPath_.empty())
            return ReturnStatus(ReturnCode::ConfigError,
                                "Config file missing required ADIANCE_MODEL_PATH or MTCNN_MODEL_PATH entries.");

        // Don't use absolute path conversion - use the paths directly
        adianceModel_ = std::make_shared<OnnxModelWrapper>(adianceModelPath_);
        return ReturnStatus(ReturnCode::Success);
    }

    // ----- Updated createFaceTemplate using RetinafaceWrapper exclusively -----
    ReturnStatus AdianceFRVT::createFaceTemplate(const std::vector<Image> &faces,
                                                 TemplateRole role,
                                                 std::vector<uint8_t> &templ,
                                                 std::vector<EyePair> &eyeCoordinates)
    {
        cv::setNumThreads(0); // Disable OpenCV multithreading
        
        std::vector<std::vector<float>> allFeatures;
        eyeCoordinates.clear();
        
        try
        {
            // Use the RetinaFace model to get a high-quality aligned face
            RetinafaceWrapper retinaface(mtcnnModelPath_);

            for (const auto &img : faces)
            {
                cv::Mat mat = imageToMat(img);

                // Face detection - without timing
                auto detectedFaces = retinaface.detect(mat);
                
                if (detectedFaces.empty()) {
                    continue;
                }
                
                // Get aligned face - without timing
                cv::Mat aligned = detectedFaces[0].aligned_face;
                std::vector<cv::Point2f> landmarks = detectedFaces[0].landmarks;

                // Make sure the aligned face is exactly 128x128
                if (aligned.size() != kOutputSize) {
                    cv::resize(aligned, aligned, kOutputSize);
                }

                // Feature extraction - without timing
                bool old_debug = g_debug_output;
                g_debug_output = false; // Temporarily disable debug output
                std::vector<float> feat = extractFeatures(aligned, false);
                g_debug_output = old_debug; // Restore debug setting

                allFeatures.push_back(feat);

                // Set eye coordinates from landmarks
                if (landmarks.size() >= 2) {
                    eyeCoordinates.push_back(EyePair(true, true,
                                                   landmarks[0].x, landmarks[0].y,
                                                   landmarks[1].x, landmarks[1].y));
                } else {
                    eyeCoordinates.push_back(EyePair(true, true, 30, 50, 80, 50));
                }
            }

            if (allFeatures.empty()) {
                return ReturnStatus(ReturnCode::FaceDetectionError, "No valid face features extracted");
            }

            // Feature fusion - without timing
            std::vector<float> fusedFeat = fuseFeatures(allFeatures);

            size_t byteCount = fusedFeat.size() * sizeof(float);
            templ.resize(byteCount);
            memcpy(templ.data(), fusedFeat.data(), byteCount);
            
            return ReturnStatus(ReturnCode::Success);
        }
        catch (const std::exception &e) {
            return ReturnStatus(ReturnCode::ConfigError,
                                std::string("Model error: ") + e.what());
        }
    }

    ReturnStatus AdianceFRVT::createFaceTemplate(const Image &image,
                                                 TemplateRole role,
                                                 std::vector<std::vector<uint8_t>> &templs,
                                                 std::vector<EyePair> &eyeCoordinates)
    {
        std::vector<Image> images = {image};
        std::vector<uint8_t> templ;
        ReturnStatus status = createFaceTemplate(images, role, templ, eyeCoordinates);
        if (status.code == ReturnCode::Success)
            templs.push_back(templ);
        return status;
    }

    ReturnStatus AdianceFRVT::createIrisTemplate(const std::vector<Image> &irises,
                                                 TemplateRole role,
                                                 std::vector<uint8_t> &templ,
                                                 std::vector<IrisAnnulus> &irisLocations)
    {
        return ReturnStatus(ReturnCode::NotImplemented, "Iris template creation not implemented");
    }

    ReturnStatus AdianceFRVT::createFaceAndIrisTemplate(const std::vector<Image> &facesIrises,
                                                        TemplateRole role,
                                                        std::vector<uint8_t> &templ)
    {
        return ReturnStatus(ReturnCode::NotImplemented, "Face+Iris template creation not implemented");
    }

    ReturnStatus AdianceFRVT::finalizeEnrollment(const std::string &configDir,
                                                 const std::string &enrollmentDir,
                                                 const std::string &edbName,
                                                 const std::string &edbManifestName,
                                                 GalleryType galleryType)
    {
        std::ifstream checkEdb(edbName);
        std::ifstream checkManifest(edbManifestName);
        if (!checkEdb || !checkManifest)
        {
            return ReturnStatus(ReturnCode::ConfigError, "Target enrollment files are missing or can't be accessed.");
        }
        return ReturnStatus(ReturnCode::Success);
    }

    ReturnStatus AdianceFRVT::initializeIdentification(const std::string &configDir,
                                                       const std::string &enrollmentDir)
    {
        return loadEnrollmentGallery(enrollmentDir, gallery_);
    }

    ReturnStatus AdianceFRVT::identifyTemplate(const std::vector<uint8_t> &idTemplate,
                                               const uint32_t candidateListLength,
                                               std::vector<Candidate> &candidateList)
    {
        cv::setNumThreads(0); // Disable OpenCV multithreading
        
        if (idTemplate.empty())
            return ReturnStatus(ReturnCode::TemplateCreationError, "Empty probe template");

        size_t numFloats = idTemplate.size() / sizeof(float);
        const float *probeData = reinterpret_cast<const float *>(idTemplate.data());
        std::vector<float> probeTemplate(probeData, probeData + numFloats);

        // Normalize probe template
        l2_normalize(probeTemplate);

        std::vector<std::pair<std::string, float>> scores;

        // Calculate similarity scores for each gallery entry
        for (const auto &entry : gallery_)
        {
            float sim = cosineSimilarityImpl(probeTemplate, entry.second);
            scores.push_back({entry.first, sim});
        }

        // Sort by similarity score (descending)
        std::sort(scores.begin(), scores.end(), [](const auto &a, const auto &b)
                  { return a.second > b.second; });

        // Create candidate list
        candidateList.clear();
        for (size_t i = 0; i < std::min((size_t)candidateListLength, scores.size()); i++)
        {
            // Make sure we're using the properly formatted S-ID
            std::string candidateId = scores[i].first;
            candidateList.push_back(Candidate(true, candidateId, scores[i].second));
        }

        return ReturnStatus(ReturnCode::Success);
    }

    ReturnStatus AdianceFRVT::loadEnrollmentGallery(const std::string &enrollmentDir,
                                                    std::map<std::string, std::vector<float>> &gallery)
    {
        std::string manifestPath = enrollmentDir + "/manifest";
        std::string edbPath = enrollmentDir + "/edb";

        std::ifstream manifestFile(manifestPath);
        std::ifstream edbFile(edbPath, std::ios::binary);

        if (!manifestFile)
        {

            return ReturnStatus(ReturnCode::ConfigError, "Failed to open enrollment manifest");
        }

        if (!edbFile)
        {

            return ReturnStatus(ReturnCode::ConfigError, "Failed to open enrollment database");
        }

        gallery.clear();
        std::string line;
        int templatesLoaded = 0;

        // Read the entire manifest line for debugging

        std::string debugLine;
        // Reset file position to start
        manifestFile.clear();
        manifestFile.seekg(0, std::ios::beg);

        while (std::getline(manifestFile, line))
        {
            std::istringstream iss(line);
            std::string idStr;
            size_t templateSize, offset;

            if (iss >> idStr >> templateSize >> offset)
            {
                // CRITICAL FIX: Use the EXACT Template ID from manifest
                // with NO conversion whatsoever
                std::string templateId = idStr; // Use as-is

                // Read template from edb file at specified offset
                std::vector<char> buffer(templateSize);
                edbFile.seekg(offset, std::ios::beg);
                edbFile.read(buffer.data(), templateSize);

                if (edbFile.gcount() == static_cast<std::streamsize>(templateSize))
                {
                    // Process template and add to gallery
                    size_t numFloats = templateSize / sizeof(float);
                    const float *fptr = reinterpret_cast<const float *>(buffer.data());
                    std::vector<float> feat(fptr, fptr + numFloats);
                    l2_normalize(feat);
                    gallery[templateId] = feat;
                    templatesLoaded++;
                }
                else
                {
                }
            }
            else
            {
            }
        }

        if (gallery.empty())
        {
            return ReturnStatus(ReturnCode::ConfigError, "No templates were loaded into gallery");
        }

        return ReturnStatus(ReturnCode::Success);
    }

    float AdianceFRVT::cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b)
    {
        return cosineSimilarityImpl(a, b);
    }

    std::shared_ptr<Interface> AdianceFRVT::getImplementation(const std::string &adianceModelPath,
                                                              const std::string &mtcnnModelPath)
    {
        return std::make_shared<AdianceFRVT>(adianceModelPath, mtcnnModelPath);
    }

} // namespace FRVT_1N