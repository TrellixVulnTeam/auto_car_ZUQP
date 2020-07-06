#include <fstream>
#include <iostream>
#include <vector>
#include <thread>
#include <unistd.h>
#include <mutex>
#include <csignal>
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "camera.hpp"
#include "blocking_queue.h"
#include "io/paddle_inference_api.h"
#include "json.hpp"

using namespace std;
using namespace cv;
using namespace paddle_mobile;
using json = nlohmann::json;

const string name = "detection";

static Camera g_cap;
static BlockingQueue<Mat> g_image_queue;
static cv::Mat g_display_frame;
static std::mutex g_mtx;

static int width = 300;
static int height = 300;
static float THRESHOLD = 0.5f;

float* data = nullptr;

std::vector<float> mean_data;
float scale = 0;
static std::string image_format = "BGR";

//SSD解析后的结果
struct SSDResult{
    int type;
    float score;
    int x;
    int y;
    int width;
    int height;
};

std::unique_ptr<paddle_mobile::PaddlePredictor> g_predictor;
std::vector<SSDResult> g_results;

void init(json& value) {
    PaddleMobileConfig config;
    std::string model_dir = value["model"];
    config.precision = PaddleMobileConfig::FP32;
    config.device = PaddleMobileConfig::kFPGA;

    config.prog_file = model_dir + "/model";
    config.param_file = model_dir + "/params";
    config.thread_num = 4;
    g_predictor =
      CreatePaddlePredictor<PaddleMobileConfig,
                            PaddleEngineKind::kPaddleMobile>(config);

    width = value["input_width"];
    height = value["input_width"];
    std::vector<float> mean = value["mean"];
    for (int i = 0; i < mean.size(); ++i) {
        mean_data.push_back(mean[i]);
    }

    scale = value["scale"];
    image_format = value["format"];
    THRESHOLD = value["threshold"];
    data = new float[3 * width * height];
}

int predict() {
    while (true) {
        cv::Mat mat = g_image_queue.Take();
        g_mtx.lock();
        // 预处理。
        cv::Mat preprocessMat;
        mat.convertTo(preprocessMat, CV_32FC3);
        int index = 0;
        for (int row = 0; row < preprocessMat.rows; ++row) {
            float* ptr = (float*)preprocessMat.ptr(row);
            for (int col = 0; col < preprocessMat.cols; col++) {
                float* uc_pixel = ptr;
                float b = uc_pixel[0];
                float g = uc_pixel[1];
                float r = uc_pixel[2];
                // 减均值
                if (image_format == "RGB") {
                    data[index] = (r - mean_data[0]) * scale ;
                    data[index + 1] = (g - mean_data[1]) * scale ;
                    data[index + 2] = (b - mean_data[2]) * scale ;
                } else {
                    data[index] = (b - mean_data[0]) * scale ;
                    data[index + 1] = (g - mean_data[1]) * scale ;
                    data[index + 2] = (r - mean_data[2]) * scale ;
                }
                ptr += 3;
                index += 3;
            }
        }

        PaddleTensor tensor;
        tensor.shape = std::vector<int>({1, 3, width, height});
        tensor.data = PaddleBuf(data, 3 * width * height * sizeof(float));
        tensor.dtype = PaddleDType::FLOAT32;
        std::vector<PaddleTensor> paddle_tensor_feeds(1, tensor);

        PaddleTensor tensor_out;
        tensor_out.shape = std::vector<int>({});
        tensor_out.data = PaddleBuf();
        tensor_out.dtype = PaddleDType::FLOAT32;
        std::vector<PaddleTensor> outputs(1, tensor_out);

        g_predictor->Run(paddle_tensor_feeds, &outputs);
        float *result_data = static_cast<float *>(outputs[0].data.data());
        int size = outputs[0].shape[0];

        g_results.clear();
        int display_width = g_display_frame.cols;
        int display_height = g_display_frame.rows;

        for (int i = 0; i < size; i++) {
            float* data = result_data + i * 6;
            float score = data[1];
            if (score < THRESHOLD) {
                continue;
            }
            SSDResult r;
            r.type = (int)data[0];
            r.score = score;
            r.x = data[2] * display_width;
            r.y = data[3] * display_height;
            r.width = data[4] * display_width - r.x;
            r.height = data[5] * display_height - r.y;
            g_results.push_back(r);
        }   
        g_mtx.unlock();
    }
    return 0;
}

//相机回调，image为最新的一帧图片
int image_callback(cv::Mat& image) {
    if (g_image_queue.Size() >= 1) {
        return 0; 
    }
    cv::Mat mat;
    g_mtx.lock();
    g_display_frame = image;
    cv::resize(image, mat, Size(width, height), INTER_AREA);
    g_mtx.unlock();
    g_image_queue.Put(mat);
    return 0;    
}

void capture() {
    CameraConfig config;
    config.dev_name = "/dev/video2";
    config.width = 1280;
    config.height = 720;
    g_cap.setConfig(config);
    g_cap.start(image_callback);
    g_cap.loop();
}

void signal_handler( int signum ) {
    g_cap.release();
    exit(signum);  
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
    signal(SIGINT, signal_handler); 

    // 为相机开启一个新的线程
    std::thread task_capture(capture);
    // 预测线程
    std::thread task_predict(predict);
    
    cv::namedWindow(name);
    moveWindow(name, 20, 20);
    usleep(2000); // 等待窗口初始化
    while (true) {
        if (g_display_frame.cols == 0) {
            continue;
        }
        g_mtx.lock();
        for (int i = 0; i < g_results.size(); ++i) {
            SSDResult r = g_results[i];
            cv::Rect rect(r.x, r.y,r.width,r.height);
            cv::rectangle(g_display_frame, rect, Scalar(0,0,224), 2);
        }
        imshow(name, g_display_frame);
        g_mtx.unlock();
        cv::waitKey(10);
    }
    task_capture.join();
    task_predict.join();
    return 0;
}

