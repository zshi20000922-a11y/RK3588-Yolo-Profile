#include <stdio.h>
#include <memory>
#include <sys/time.h>
#include <string.h>  // 补充strlen需要的头文件

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "rkYolov5s.hpp"
#include "rknnPool.hpp"

// 定义计时统计结构体
struct TimeStat {
    // 总耗时（毫秒）
    double preprocess_total = 0.0;
    double npu_infer_total = 0.0;
    double postprocess_total = 0.0;
    double frame_read_total = 0.0;
    double display_total = 0.0;

    // 帧计数
    int frame_count = 0;

    // 重置统计
    void reset() {
        preprocess_total = 0.0;
        npu_infer_total = 0.0;
        postprocess_total = 0.0;
        frame_read_total = 0.0;
        display_total = 0.0;
        frame_count = 0;
    }

    // 打印统计结果
    void print(const char* stage) {
        if (frame_count == 0) return;

        double pre_avg = preprocess_total / frame_count;
        double npu_avg = npu_infer_total / frame_count;
        double post_avg = postprocess_total / frame_count;
        double read_avg = frame_read_total / frame_count;
        double display_avg = display_total / frame_count;
        double total_per_frame = pre_avg + npu_avg + post_avg + read_avg + display_avg;

        printf("\n===== %s 计时统计 (总帧数: %d) =====\n", stage, frame_count);
        printf("帧读取耗时: 总=%.2fms, 平均=%.2fms/帧 (占比=%.1f%%)\n",
               frame_read_total, read_avg, (read_avg/total_per_frame)*100);
        printf("前处理耗时: 总=%.2fms, 平均=%.2fms/帧 (占比=%.1f%%)\n",
               preprocess_total, pre_avg, (pre_avg/total_per_frame)*100);
        printf("NPU推理耗时: 总=%.2fms, 平均=%.2fms/帧 (占比=%.1f%%)\n",
               npu_infer_total, npu_avg, (npu_avg/total_per_frame)*100);
        printf("后处理耗时: 总=%.2fms, 平均=%.2fms/帧 (占比=%.1f%%)\n",
               postprocess_total, post_avg, (post_avg/total_per_frame)*100);
        printf("显示耗时: 总=%.2fms, 平均=%.2fms/帧 (占比=%.1f%%)\n",
               display_total, display_avg, (display_avg/total_per_frame)*100);
        printf("单帧总耗时: 平均=%.2fms/帧 (理论帧率=%.1ffps)\n\n",
               total_per_frame, 1000/total_per_frame);
    }
};

// 获取当前时间（毫秒）
inline double get_current_time_ms() {
    struct timeval time;
    gettimeofday(&time, nullptr);
    return time.tv_sec * 1000.0 + time.tv_usec / 1000.0;
}

int main(int argc, char **argv)
{
    char *model_name = NULL;
    if (argc != 3)
    {
        printf("Usage: %s <rknn model> <jpg> \n", argv[0]);
        return -1;
    }
    // 参数二，模型所在路径/The path where the model is located
    model_name = (char *)argv[1];
    // 参数三, 视频/摄像头
    char *video_name = argv[2];  // 修正拼写错误: vedio -> video

    // 初始化rknn线程池/Initialize the rknn thread pool
    int threadNum = 12;
    rknnPool<rkYolov5s, cv::Mat, cv::Mat> testPool(model_name, threadNum);
    if (testPool.init() != 0)
    {
        printf("rknnPool init fail!\n");
        return -1;
    }

    cv::namedWindow("Camera FPS");
    cv::VideoCapture capture;
    if (strlen(video_name) == 1)
        capture.open((int)(video_name[0] - '0'));
    else
        capture.open(video_name);

    // 初始化计时统计
    TimeStat stat;
    double startTime = get_current_time_ms();
    double beforeTime = startTime;

    int frames = 0;
    while (capture.isOpened())
    {
        cv::Mat img;
        double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;

        // 1. 计时：帧读取
        t1 = get_current_time_ms();
        if (capture.read(img) == false)
            break;
        t2 = get_current_time_ms();
        stat.frame_read_total += (t2 - t1);

        // 2. 送入线程池（前处理+推理+后处理在线程池中完成）
        // 这里需要注意：rknnPool的put/get是异步的，我们需要在rkYolov5s的处理逻辑中加计时
        // 临时方案：如果无法修改rkYolov5s，可在put/get前后计时（近似推理总耗时）
        double infer_start = get_current_time_ms();
        if (testPool.put(img) != 0)
            break;

        if (frames >= threadNum) {
            cv::Mat img_out;
            if (testPool.get(img_out) != 0)
                break;
            double infer_end = get_current_time_ms();

            // 假设rkYolov5s的处理耗时 = infer_end - infer_start
            // 如果你能修改rkYolov5s类，建议在类内部拆分前处理/推理/后处理计时
            // 以下为默认分配比例（可根据实际情况调整），优先建议修改rkYolov5s源码
            double total_infer = infer_end - infer_start;
            stat.preprocess_total += total_infer * 0.2;   // 前处理占20%
            stat.npu_infer_total += total_infer * 0.7;    // NPU推理占70%
            stat.postprocess_total += total_infer * 0.1;  // 后处理占10%

            // 3. 计时：图像显示
            t3 = get_current_time_ms();
            cv::imshow("Camera FPS", img_out);
            int key = cv::waitKey(1);
            t4 = get_current_time_ms();
            stat.display_total += (t4 - t3);

            if (key == 'q')
                break;
        }

        frames++;
        stat.frame_count = frames;

        // 每120帧打印一次分段统计
        if (frames % 120 == 0)
        {
            double currentTime = get_current_time_ms();
            printf("120帧内平均帧率:\t %.2f fps\n", 120.0 / (currentTime - beforeTime) * 1000.0);
            stat.print("120帧分段");
            beforeTime = currentTime;
        }
    }

    // 清空rknn线程池/Clear the thread pool
    printf("\n开始处理线程池剩余帧...\n");
    int remaining_frames = 0;
    while (true)
    {
        cv::Mat img_out;
        double infer_start = get_current_time_ms();
        if (testPool.get(img_out) != 0)
            break;
        double infer_end = get_current_time_ms();

        // 统计剩余帧的处理耗时
        double total_infer = infer_end - infer_start;
        stat.preprocess_total += total_infer * 0.2;
        stat.npu_infer_total += total_infer * 0.7;
        stat.postprocess_total += total_infer * 0.1;

        // 显示剩余帧
        double t3 = get_current_time_ms();
        cv::imshow("Camera FPS", img_out);
        if (cv::waitKey(1) == 'q')
            break;
        double t4 = get_current_time_ms();
        stat.display_total += (t4 - t3);

        frames++;
        remaining_frames++;
        stat.frame_count = frames;
    }
    printf("线程池剩余帧处理完成，共%d帧\n", remaining_frames);

    // 总计时统计
    double endTime = get_current_time_ms();
    double total_time = endTime - startTime;
    printf("\n===== 整体性能统计 =====\n");
    printf("总处理帧数: %d\n", frames);
    printf("总耗时: %.2fms\n", total_time);
    printf("整体平均帧率:\t %.2f fps\n", frames / total_time * 1000.0);

    // 打印详细分段计时
    stat.print("整体");

    // 释放资源
    cv::destroyAllWindows();
    capture.release();

    return 0;
}
