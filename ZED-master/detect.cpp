// // Minimal, single-file example: YOLO + ZED lift-to-3D for basketball center
// // deps: OpenCV (core,imgproc,highgui,dnn), PCL (common,visualization), your ZED wrapper (Zed.hpp)

// #include <opencv2/opencv.hpp>
// #include <opencv2/dnn.hpp>
// #include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <signal.h>
// #include <algorithm>
// #include <numeric>
// #include <cmath>
// #include <vector>
// #include <string>
// #include <iostream>

// #include "Zed.hpp"

// using namespace cv;
// //yolo相关结构体
// struct Detection2D
// {
//     Rect box;
//     int class_id;
//     float score;
// };
// //3d中心结构体
// struct Basketball3D
// {
//     Detection2D det;
//     Vec3f center;
//     bool valid = false;
//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr crop;
// };
// //内联函数
// static inline bool finite3(float x, float y, float z) { return std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && z > 0.f; }
// static inline float median1(std::vector<float> &v)
// {
//     if (v.empty())
//         return NAN;
//     size_t m = v.size() / 2;
//     std::nth_element(v.begin(), v.begin() + m, v.end());
//     return v[m];
// }
// /**
//  * @brief 
//  * 
//  * @param pts 
//  * @return Vec3f 
//  */
// static Vec3f medianXYZ(const std::vector<Vec3f> &pts)
// {
//     std::vector<float> X, Y, Z;
//     X.reserve(pts.size());
//     Y.reserve(pts.size());
//     Z.reserve(pts.size());
//     for (auto &p : pts)
//     {
//         X.push_back(p[0]);
//         Y.push_back(p[1]);
//         Z.push_back(p[2]);
//     }
//     return {median1(X), median1(Y), median1(Z)};
// }

// class YOLO
// {
// public:
//     bool load(const std::string &onnx, int input = 640, float conf = 0.35f, float nms = 0.45f)
//     {
//         in_ = input;
//         conf_ = conf;
//         nms_ = nms;
//         net_ = dnn::readNetFromONNX(onnx);
//         net_.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
//         net_.setPreferableTarget(dnn::DNN_TARGET_CPU);
//         return !net_.empty();
//     }
//     std::vector<Detection2D> infer(const Mat &bgr, int num_classes)
//     {
//         std::vector<Detection2D> out;
//         if (bgr.empty())
//             return out;
//         Mat blob = dnn::blobFromImage(bgr, 1 / 255.f, Size(in_, in_), Scalar(), true, false);
//         net_.setInput(blob);
//         std::vector<Mat> outs;
//         net_.forward(outs);
//         if (outs.empty())
//             return out;
//         float rx = (float)bgr.cols / in_, ry = (float)bgr.rows / in_;
//         Mat det = outs[0];
//         det = det.reshape(1, det.total() / (5 + num_classes));
//         std::vector<Rect> boxes;
//         std::vector<float> scores;
//         std::vector<int> ids;
//         for (int i = 0; i < det.rows; ++i)
//         {
//             float cx = det.at<float>(i, 0), cy = det.at<float>(i, 1), w = det.at<float>(i, 2), h = det.at<float>(i, 3), obj = det.at<float>(i, 4);
//             int bid = -1;
//             float bsc = 0;
//             for (int c = 5; c < det.cols; ++c)
//             {
//                 float s = det.at<float>(i, c);
//                 if (s > bsc)
//                 {
//                     bsc = s;
//                     bid = c - 5;
//                 }
//             }
//             float sc = obj * bsc;
//             if (sc < conf_)
//                 continue;
//             int x = (int)((cx - w / 2) * rx), y = (int)((cy - h / 2) * ry), bw = (int)(w * rx), bh = (int)(h * ry);
//             Rect r(x, y, bw, bh);
//             r &= Rect(0, 0, bgr.cols, bgr.rows);
//             if (r.area() <= 0)
//                 continue;
//             boxes.push_back(r);
//             scores.push_back(sc);
//             ids.push_back(bid);
//         }
//         std::vector<int> keep;
//         dnn::NMSBoxes(boxes, scores, conf_, nms_, keep);
//         for (int k : keep)
//             out.push_back({boxes[k], ids[k], scores[k]});
//         return out;
//     }

// private:
//     dnn::Net net_;
//     int in_ = 640;
//     float conf_ = 0.35f, nms_ = 0.45f;
// };

// static Basketball3D lift_ball_3d(const Detection2D &det, const Mat &img,
//                                  std::function<void(int, int, float[3])> coord_fn, int stride = 2, float inner_ratio = 0.45f, bool build_cloud = true)
// {
//     Basketball3D r;
//     r.det = det;
//     r.crop.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
//     int x0 = det.box.x, y0 = det.box.y, w = det.box.width, h = det.box.height;
//     int cx = x0 + w / 2, cy = y0 + h / 2;
//     float rad = 0.5f * std::min(w, h) * inner_ratio;
//     std::vector<Vec3f> pts;
//     pts.reserve((w / stride + 1) * (h / stride + 1));
//     for (int v = y0; v < y0 + h; v += stride)
//     {
//         for (int u = x0; u < x0 + w; u += stride)
//         {
//             float dx = u - cx, dy = v - cy;
//             if (dx * dx + dy * dy > rad * rad)
//                 continue;
//             float p[3] = {0, 0, 0};
//             coord_fn(u, v, p);
//             if (!finite3(p[0], p[1], p[2]))
//                 continue;
//             pts.emplace_back(p[0], p[1], p[2]);
//             if (build_cloud)
//             {
//                 pcl::PointXYZRGB q;
//                 q.x = p[0];
//                 q.y = p[1];
//                 q.z = p[2];
//                 Vec3b bgr = img.at<Vec3b>(v, u);
//                 q.b = bgr[0];
//                 q.g = bgr[1];
//                 q.r = bgr[2];
//                 r.crop->points.push_back(q);
//             }
//         }
//     }
//     if (!pts.empty())
//     {
//         Vec3f med = medianXYZ(pts); // optional inlier refine
//         std::vector<Vec3f> inl;
//         inl.reserve(pts.size());
//         float th = 0.20f;
//         for (auto &p : pts)
//         {
//             if (norm(p - med) < th)
//                 inl.push_back(p);
//         }
//         r.center = inl.empty() ? med : medianXYZ(inl);
//         r.valid = std::isfinite(r.center[0]) && std::isfinite(r.center[1]) && std::isfinite(r.center[2]);
//     }
//     if (build_cloud)
//     {
//         r.crop->width = (uint32_t)r.crop->points.size();
//         r.crop->height = 1;
//         r.crop->is_dense = false;
//     }
//     return r;
// }

// static volatile bool stop_flag = false;
// static void onSig(int) { stop_flag = true; }

// int main()
// {
//     signal(SIGINT, onSig);
//     // ---- config ----
//     const std::string onnx_path = "/home/lyk/TensorRT-YOLO/models/output1/best.onnx"; // TODO: set
//         const int NUM_CLASSES = 1;                             // TODO: set
//     const int BASKETBALL_ID = 0;                               // TODO: set

//     // ZED
//     ZED Zed;
//     Zed.init();
//     Zed.setCamera();

//     // Viewer
//     auto viewer = boost::make_shared<pcl::visualization::PCLVisualizer>("Viewer");
//     viewer->addCoordinateSystem(1.0);

//     // YOLO
//     YOLO yolo;
//     if (!yolo.load(onnx_path, 640, 0.35f, 0.45f))
//     {
//         std::cerr << "YOLO load failed\n";
//         return 1;
//     }
//     //
//     while (!stop_flag)
//     {
//         pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
//         Mat img;
//         Zed.getCloud(cloud, img);
//         if (img.empty())
//             continue;
//         Zed.update_sl_point_cloud();

//         auto dets = yolo.infer(img, NUM_CLASSES);
//         // pick best basketball
//         Detection2D best;
//         bool has = false;
//         float best_sc = 0.f;
//         for (auto &d : dets)
//         {
//             if (d.class_id != BASKETBALL_ID)
//                 continue;
//             if (!has || d.score > best_sc)
//             {
//                 best = d;
//                 best_sc = d.score;
//                 has = true;
//             }
//         }

//         if (has)
//         {
//             auto coord_fn = [&](int u, int v, float out3[3])
//             { Zed.coordinate_from_sl_point_cloud(Point(u, v), out3); };
//             auto ball = lift_ball_3d(best, img, coord_fn, 2, 0.45f, true);
            
//             rectangle(img, best.box, Scalar(0, 165, 255), 2);
//             if (ball.valid)
//             {
//                 char text[128];
//                 snprintf(text, sizeof(text), "(%.2f,%.2f,%.2f)m conf=%.2f", ball.center[0], ball.center[1], ball.center[2], best.score);
//                 putText(img, text, {best.box.x, std::max(0, best.box.y - 6)}, FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 165, 255), 2);
//                 if (viewer->contains("ball"))
//                     viewer->removePointCloud("ball");
//                 viewer->addPointCloud(ball.crop, "ball");
//                 viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "ball");
//             }
//             else
//             {
//                 putText(img, "ball", {best.box.x, std::max(0, best.box.y - 6)}, FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
//             }
//         }

//         imshow("img", img);
//         waitKey(1);
//         if (viewer->contains("cloud"))
//             viewer->removePointCloud("cloud");
//         viewer->addPointCloud(cloud, "cloud");
//         viewer->spinOnce(1);
//     }

//     Zed.close();
//     return 0;
// }
/**
 * @file detect.cpp
 * @author lyk (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2025-11-08
 * 
 * @copyright Copyright (c) 2025
 * 
 *
 * @file detect.cpp
 * @brief ZED + TensorRT-YOLO 实时篮球检测 + 质心三维坐标
 */
#include <iostream>
#include <cmath>
#include <memory>
#include <pcl/visualization/pcl_visualizer.h> // 新增
#include <pcl/visualization/cloud_viewer.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "Zed.hpp"     
#include "trtyolo.hpp" 

int main()
{
    try
    {
        // ZED 初始化 
        std::cout << "[INFO] Init ZED..." << std::endl;
        ZED Zed;
        Zed.init();
        Zed.setCamera();
        std::cout << "[INFO] ZED init done." << std::endl;

        //  TRT YOLO 初始化
        const std::string engine_path = "/home/lyk/TensorRT-YOLO/examples/detect/models/ball.engine";
        std::cout << "[INFO] Loading TensorRT-YOLO engine: " << engine_path << std::endl;

        trtyolo::InferOption option;
        option.enableSwapRB(); // BGR -> RGB

        // 如果你的 YOLO 训练仅做 /255，这里就保持 0/1 不做 mean/std
        option.setNormalizeParams({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});

        std::cout << "[INFO] Create YOLO model..." << std::endl;
        trtyolo::DetectModel model(engine_path, option);

        cv::namedWindow("ZED + TRT-YOLO", cv::WINDOW_NORMAL);

        pcl::visualization::PCLVisualizer::Ptr v(new pcl::visualization::PCLVisualizer);
        int p1, p2;
        v->createViewPort(0.0, 0.0, 0.5, 1.0, p1);
        v->createViewPort(0.5, 0.0, 1.0, 1.0, p2);

        while (true)
        {
            std::cout << "[INFO] Grab ZED frame..." << std::endl;

            // --------------- 获取 ZED 图像 + 点云 ---------------
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            cv::Mat img;
            Zed.getCloud(cloud, img);

            if (img.empty())
            {
                std::cerr << "[WARN] ZED 图像为空，跳过这一帧" << std::endl;
                continue;
            }

            if (!cloud || cloud->empty())
            {
                std::cerr << "[WARN] 点云为空，跳过这一帧" << std::endl;
                continue;
            }

            const int img_w = img.cols;
            const int img_h = img.rows;

            // =================== 关键新增：更新 ZED 深度缓存 ===================
            // 必须在本帧使用 coordinate_from_sl_point_cloud 之前调用
            Zed.update_sl_point_cloud();
            // ===================================================================

            // （可选 debug）看一下图像中心的深度是否正常
            {
                int cx_c = img_w / 2;
                int cy_c = img_h / 2;
                float w_center[3] = {0.f, 0.f, 0.f};
                Zed.coordinate_from_sl_point_cloud(cv::Point(cx_c, cy_c), w_center);
                float d_center = std::sqrt(
                    w_center[0] * w_center[0] +
                    w_center[1] * w_center[1] +
                    w_center[2] * w_center[2]);
                std::cout << "[DEBUG] center depth 3D=("
                          << w_center[0] << ", " << w_center[1] << ", " << w_center[2]
                          << "), dist=" << d_center << " m" << std::endl;
            }

            // --------------- YOLO 推理 ---------------
            
            trtyolo::Image timg(img.data, img.cols, img.rows);
            trtyolo::DetectRes det_res = model.predict(timg);
            std::cout << "[INFO] YOLO predict done. num=" << det_res.num << std::endl;

            if (det_res.num == 0)
            {
                std::cout << "[INFO] 本帧未检测到目标" << std::endl;
                cv::imshow("ZED + TRT-YOLO", img);
                int key0 = cv::waitKey(1);
                if (key0 == 27)
                    break;
                continue;
            }

            int best_idx = -1;
            float best_score = 0.0f;

            if (det_res.boxes.size() < det_res.num ||
                det_res.scores.size() < det_res.num ||
                det_res.classes.size() < det_res.num)
            {
                std::cerr << "[ERROR] DetectRes 数组长度与 num 不一致，可能是 trtyolo 内部问题" << std::endl;
                break;
            }

            for (size_t i = 0; i < det_res.num; ++i)
            {
                float score = det_res.scores[i];
                if (score > best_score)
                {
                    best_score = score;
                    best_idx = static_cast<int>(i);
                }
            }

            if (best_idx >= 0)
            {
                const auto &box = det_res.boxes[best_idx];
                int cls = det_res.classes[best_idx];

                std::cout << "[DEBUG] box: l=" << box.left << " t=" << box.top
                          << " r=" << box.right << " b=" << box.bottom
                          << " score=" << best_score << " cls=" << cls << std::endl;

                int left = static_cast<int>(std::round(box.left));
                int top = static_cast<int>(std::round(box.top));
                int right = static_cast<int>(std::round(box.right));
                int bottom = static_cast<int>(std::round(box.bottom));

                // 边界裁剪
                left = std::max(0, std::min(left, img_w - 1));
                right = std::max(0, std::min(right, img_w - 1));
                top = std::max(0, std::min(top, img_h - 1));
                bottom = std::max(0, std::min(bottom, img_h - 1));

                if (right <= left || bottom <= top)
                {
                    std::cerr << "[WARN] 裁剪后 bbox 无效，跳过这一帧" << std::endl;
                    cv::imshow("ZED + TRT-YOLO", img);
                    int key1 = cv::waitKey(1);
                    if (key1 == 27)
                        break;
                    continue;
                }

                cv::Rect rect(
                    cv::Point(left, top),
                    cv::Point(right, bottom));

                int cx = rect.x + rect.width / 2;
                int cy = rect.y + rect.height / 2;

                // 再次保证中心点在图像范围内
                cx = std::max(0, std::min(cx, img_w - 1));
                cy = std::max(0, std::min(cy, img_h - 1));

                // --------------- 从 ZED 点云获取 3D 坐标 ---------------
                float world3[3] = {0.f, 0.f, 0.f};
                try
                {
                    Zed.coordinate_from_sl_point_cloud(cv::Point(cx, cy), world3);
                }
                catch (const std::exception &e)
                {
                    std::cerr << "[ERROR] coordinate_from_sl_point_cloud 异常: "
                              << e.what() << std::endl;
                    continue;
                }


                float X = world3[0];
                float Y = world3[1];
                float Z = world3[2];
                float dist = std::sqrt(X * X + Y * Y + Z * Z);

                std::cout << "[INFO] Ball center pixel: (" << cx << ", " << cy << ")  "
                          << "3D: (" << X << ", " << Y << ", " << Z << ")  "
                          << "distance = " << dist << " m"
                          << std::endl;

                // --------------- 在图像上可视化 ---------------
                cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);
                cv::circle(img, cv::Point(cx, cy), 4, cv::Scalar(0, 0, 255), -1);

                std::string text = "ball " + cv::format("%.2f m", dist);
                cv::putText(img, text, cv::Point(rect.x, std::max(0, rect.y - 5)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7,
                            cv::Scalar(0, 255, 0), 2);



                v->addPointCloud(cloud, "cloud", p1);
                v->addSphere(pcl::PointXYZ(X, Y, Z), 0.4, "sphere", p1);
                v->spinOnce();
                v->removeAllPointClouds(p1);
                v->removeAllShapes(p1);
            }
            // //pcl显示
            // if (!viewer->updatePointCloud<pcl::PointXYZRGB>(cloud, cloud_id))
            // {
            //     viewer->addPointCloud<pcl::PointXYZRGB>(cloud, cloud_id);
            //     viewer->setPointCloudRenderingProperties(
            //         pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloud_id);
            // }
            // viewer->spinOnce(3); // 刷新点云窗口



            // --------------- 显示 ---------------
            cv::imshow("ZED + TRT-YOLO", img);
            int key = cv::waitKey(1);
            if (key == 27)
                break;
        }

        Zed.close();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Error] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}
