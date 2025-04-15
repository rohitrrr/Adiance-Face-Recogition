#ifndef ADIANCE_FRVT_H_
#define ADIANCE_FRVT_H_

#include <memory>
#include <map>
#include <vector>
#include <string>
#include "frvt1N.h"

#include <opencv2/core.hpp>      // For cv::Point2f, cv::Size, etc.
#include <onnxruntime_cxx_api.h> // Required for the ONNX model wrapper

using FRVT::EyePair;
using FRVT::Image;
using FRVT::IrisAnnulus;
using FRVT::ReturnCode;
using FRVT::ReturnStatus;
using FRVT::TemplateRole;
using FRVT_1N::Candidate;
using FRVT_1N::GalleryType;

namespace cv
{
    class Mat;
}

namespace FRVT_1N
{
    // Forward declaration for RetinafaceWrapper.
    class RetinafaceWrapper;

    /// A re‐implementation of the FRVT 1:N interface that uses TransFace for feature extraction.
    /// This version uses an ONNX Runtime wrapper for both the TransFace and the MTCNN models.
    /// The MTCNN model is used to detect 5 facial landmarks (after preprocessing to 48×48) which are then used
    /// to align the face to a fixed reference (kDstLandmarks) with output size kOutputSize.
    class AdianceFRVT : public Interface
    {
    public:
        AdianceFRVT(const std::string &adianceModelPath, const std::string &mtcnnModelPath);
        virtual ~AdianceFRVT();

        // Enrollment functions
        virtual ReturnStatus initializeTemplateCreation(const std::string &configDir,
                                                        TemplateRole role) override;
        virtual ReturnStatus createFaceTemplate(const std::vector<Image> &faces,
                                                TemplateRole role,
                                                std::vector<uint8_t> &templ,
                                                std::vector<EyePair> &eyeCoordinates) override;
        virtual ReturnStatus createFaceTemplate(const Image &image,
                                                TemplateRole role,
                                                std::vector<std::vector<uint8_t>> &templs,
                                                std::vector<EyePair> &eyeCoordinates) override;
        virtual ReturnStatus createIrisTemplate(const std::vector<Image> &irises,
                                                TemplateRole role,
                                                std::vector<uint8_t> &templ,
                                                std::vector<IrisAnnulus> &irisLocations) override;
        virtual ReturnStatus createFaceAndIrisTemplate(const std::vector<Image> &facesIrises,
                                                       TemplateRole role,
                                                       std::vector<uint8_t> &templ) override;
        virtual ReturnStatus finalizeEnrollment(const std::string &configDir,
                                                const std::string &enrollmentDir,
                                                const std::string &edbName,
                                                const std::string &edbManifestName,
                                                GalleryType galleryType) override;
        // Identification (search) functions
        virtual ReturnStatus initializeIdentification(const std::string &configDir,
                                                      const std::string &enrollmentDir) override;
        virtual ReturnStatus identifyTemplate(const std::vector<uint8_t> &idTemplate,
                                              const uint32_t candidateListLength,
                                              std::vector<Candidate> &candidateList) override;

        // Factory method: returns an instance of the implementation.
        static std::shared_ptr<Interface> getImplementation(const std::string &adianceModelPath,
                                                            const std::string &mtcnnModelPath);

        // Public definition of the nested OnnxModelWrapper class.
        class OnnxModelWrapper
        {
        public:
            OnnxModelWrapper(const std::string &model_path)
            {
                env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Adiance");
                Ort::SessionOptions options;
                options.SetIntraOpNumThreads(1);
                options.SetInterOpNumThreads(1);
                options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
                options.DisableMemPattern();
                options.SetExecutionMode(ORT_SEQUENTIAL);
                options.DisableCpuMemArena();
                session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), options);
                Ort::AllocatorWithDefaultOptions allocator;
                auto input_name_alloc = session_->GetInputNameAllocated(0, allocator);
                input_name_ = std::string(input_name_alloc.get());
                size_t numOutputs = session_->GetOutputCount();
                for (size_t i = 0; i < numOutputs; i++)
                {
                    auto output_name_alloc = session_->GetOutputNameAllocated(i, allocator);
                    output_names_.push_back(std::string(output_name_alloc.get()));
                }
            }

            std::vector<float> run(const std::vector<float> &input_tensor_values,
                                   const std::vector<int64_t> &input_dims,
                                   int output_index = 0)
            {
                Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                    OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                    memory_info, const_cast<float *>(input_tensor_values.data()),
                    input_tensor_values.size(), input_dims.data(), input_dims.size());
                const char *inputNames[] = {input_name_.c_str()};
                const char *outputName = output_names_.at(output_index).c_str();
                auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                                    inputNames, &input_tensor, 1, &outputName, 1);
                float *output_data = output_tensors.front().GetTensorMutableData<float>();
                size_t output_size = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
                return std::vector<float>(output_data, output_data + output_size);
            }

        private:
            std::unique_ptr<Ort::Env> env_;
            std::unique_ptr<Ort::Session> session_;
            std::string input_name_;
            std::vector<std::string> output_names_;
        };

    private:
        // Helper functions
        cv::Mat imageToMat(const Image &image);
        std::vector<float> extractFeatures(const cv::Mat &alignedFace, bool flipTest = false);
        std::vector<float> fuseFeatures(const std::vector<std::vector<float>> &features);
        ReturnStatus loadEnrollmentGallery(const std::string &enrollmentDir,
                                           std::map<std::string, std::vector<float>> &gallery);
        float cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b);

        // Model file paths.
        std::string adianceModelPath_;
        std::string mtcnnModelPath_;

        // ONNX model wrappers.
        std::shared_ptr<OnnxModelWrapper> adianceModel_;
        // Detection model: load RetinafaceWrapper once in initialization.
        std::shared_ptr<RetinafaceWrapper> retinaface_;

        // Enrollment gallery: subject ID to fused feature vector.
        std::map<std::string, std::vector<float>> gallery_;

        // Configuration directory.
        std::string configDir_;

        // Constants for alignment.
        static const cv::Size kOutputSize;                   // e.g. 112×112
        static const std::vector<cv::Point2f> kDstLandmarks; // Fixed reference landmarks.

        mutable std::vector<float> preallocatedInputBuffer_;
    };

} // namespace FRVT_1N

#endif // ADIANCE_FRVT_H_
