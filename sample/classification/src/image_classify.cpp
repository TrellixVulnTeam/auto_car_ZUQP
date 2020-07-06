#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <unistd.h>
#include <mutex>
#include <csignal>
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "io/paddle_inference_api.h"
#include "json.hpp"

using namespace std;
using namespace cv;
using namespace paddle_mobile;
using json = nlohmann::json;

std::vector<string> labels;
std::unique_ptr<paddle_mobile::PaddlePredictor> g_predictor;

void read_labels(json& j) {
    auto label_path = j["labels"];
    if (label_path == nullptr) {
        return;
    }
    std::ifstream file(label_path);
    if (file.is_open()) {
        std::string line;
        while (getline(file, line)) {
            labels.push_back(line);
        }
        file.close();
    }
}

void init(json& j) {
    PaddleMobileConfig config;
    std::string model_dir = j["model"];
    config.precision = PaddleMobileConfig::FP32;
    config.device = PaddleMobileConfig::kFPGA;
    config.prog_file = model_dir + "/model";
    config.param_file = model_dir + "/params";
    config.thread_num = 4;
    g_predictor = CreatePaddlePredictor<PaddleMobileConfig,
                    PaddleEngineKind::kPaddleMobile>(config);
}

void read_image(json& value, float* data) {
    auto image = value["image"];
    Mat img = imread(image);
    std::string format = value["format"];
    std::transform(format.begin(), format.end(),format.begin(), ::toupper);

    int width = value["input_width"];
    int height = value["input_width"];
    std::vector<float> mean = value["mean"];
    float scale = value["scale"];

    Mat img2;
    resize(img, img2, Size(width, height));
    
    Mat sample_float;
    img2.convertTo(sample_float, CV_32FC3);

    int index = 0;
    for (int row = 0; row < sample_float.rows; ++row) {
        float* ptr = (float*)sample_float.ptr(row);
        for (int col = 0; col < sample_float.cols; col++) {
            float* uc_pixel = ptr;
            float b = uc_pixel[0];
            float g = uc_pixel[1];
            float r = uc_pixel[2];

            if (format == "RGB") {
                data[index] = (r - mean[0]) * scale ;
                data[index + 1] = (g - mean[1]) * scale ;
                data[index + 2] = (b - mean[2]) * scale ;
            } else {
                data[index] = (b - mean[0]) * scale ;
                data[index + 1] = (g - mean[1]) * scale ;
                data[index + 2] = (r - mean[2]) * scale ;
            }
            ptr += 3;
            index += 3;
        }
    }
}

void predict(json& value) {
    
    int width = value["input_width"];
    int height = value["input_height"];
    float* input = new float[3 * width * height];
    read_image(value, input);

    PaddleTensor tensor;
    tensor.shape = std::vector<int>({1, 3, width, height});
    tensor.data = PaddleBuf(input, sizeof(input));
    tensor.dtype = PaddleDType::FLOAT32;
    std::vector<PaddleTensor> paddle_tensor_feeds(1, tensor);

    PaddleTensor tensor_out;
    tensor_out.shape = std::vector<int>({});
    tensor_out.data = PaddleBuf();
    tensor_out.dtype = PaddleDType::FLOAT32;
    std::vector<PaddleTensor> outputs(1, tensor_out);

    g_predictor->Run(paddle_tensor_feeds, &outputs);
    float *data = static_cast<float *>(outputs[0].data.data());
    int size = outputs[0].shape[1];

    int max_index = 0;
    float score = 0;
    for (int i = 0; i < size; i++) {
        float value = data[i];
        if (value > score) {
            max_index = i;
            score = value;
        }
    }
    if (labels.size() > 0) {
        std::cout << "label:" << labels[max_index] << std::endl;    
    }
    std::cout << "index:" << max_index << std::endl;
    std::cout << "score:" << score << std::endl;

    delete[] input;
}

int main(int argc, char* argv[]){
    std::string path;
    if (argc > 1) {
        path = argv[1];
    } else {
        path = "../configs/config.json";
    }
    
    json j;
    std::ifstream is(path);
    is >> j;
    init(j);
    read_labels(j);
    predict(j);
    return 0;
}

