#ifndef RKYOLOV5S_H
#define RKYOLOV5S_H

// 补充计时所需头文件
#include <sys/time.h>
#include <mutex>
#include <string>
#include <vector>

#include "rknn_api.h"

#include "opencv2/core/core.hpp"

// 保留原有静态函数声明
static void dump_tensor_attr(rknn_tensor_attr *attr);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);

class rkYolov5s
{
private:
    // ========== 原有私有成员 ==========
    int ret;
    std::mutex mtx;
    std::string model_path;
    unsigned char *model_data;

    rknn_context ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    rknn_input inputs[1];

    int channel, width, height;
    int img_width, img_height;

    float nms_threshold, box_conf_threshold;

    // ========== 新增：计时相关私有成员 ==========
    // 单帧各阶段耗时（毫秒）
    double preprocess_time = 0.0;   // 前处理耗时
    double infer_time = 0.0;        // NPU推理耗时
    double postprocess_time = 0.0;  // 后处理耗时

    // 工具函数：获取当前时间（毫秒）
    inline double get_current_time_ms() {
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    }

    // 重置单帧计时（内部使用）
    inline void reset_time_stats() {
        preprocess_time = 0.0;
        infer_time = 0.0;
        postprocess_time = 0.0;
    }

public:
    // ========== 原有公有接口 ==========
    rkYolov5s(const std::string &model_path);
    int init(rknn_context *ctx_in, bool isChild);
    rknn_context *get_pctx();
    cv::Mat infer(cv::Mat &ori_img);
    ~rkYolov5s();

    // ========== 新增：获取分段耗时的公有接口 ==========
    // 获取前处理耗时（毫秒）
    double get_preprocess_time() const { return preprocess_time; }
    // 获取NPU推理耗时（毫秒）
    double get_infer_time() const { return infer_time; }
    // 获取后处理耗时（毫秒）
    double get_postprocess_time() const { return postprocess_time; }
};

#endif
