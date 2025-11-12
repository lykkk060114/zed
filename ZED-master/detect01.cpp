#include <iostream>
#include <cmath>
#include <memory>

#include <opencv2/opencv.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "Zed.hpp"     
#include "trtyolo.hpp" 

// -------------------- RANSAC 球拟合函数 --------------------
// 多加一个 inlier_cloud，用于可视化拟合到球上的点
bool fitSphereRANSAC(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_roi,
    Eigen::Vector4f &sphere_center, float &sphere_radius,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &inlier_cloud)
{
    if (!cloud_roi || cloud_roi->empty())
    {
        std::cerr << "[RANSAC] ROI 点云为空，无法拟合球" << std::endl;
        return false;
    }

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_SPHERE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(2000);
    seg.setDistanceThreshold(0.01f); // 根据噪声情况可以调
    seg.setInputCloud(cloud_roi);

    pcl::ModelCoefficients coeffs;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    seg.segment(*inliers, coeffs);

    if (inliers->indices.empty())
    {
        std::cerr << "[RANSAC] 拟合球失败,没有 inliers" << std::endl;
        return false;
    }

    if (coeffs.values.size() < 4)
    {
        std::cerr << "[RANSAC] 球参数不足,size=" << coeffs.values.size() << std::endl;
        return false;
    }

    // coeffs: [cx, cy, cz, r]
    sphere_center[0] = coeffs.values[0];
    sphere_center[1] = coeffs.values[1];
    sphere_center[2] = coeffs.values[2];
    sphere_center[3] = 1.0f;
    sphere_radius = coeffs.values[3];

    // 提取 inlier 点云，用于右侧可视化
    inlier_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    inlier_cloud->reserve(inliers->indices.size());
    for (int idx : inliers->indices)
    {
        inlier_cloud->push_back((*cloud_roi)[idx]);
    }
    std::cout << "[RANSAC] Sphere center = (" << sphere_center[0] << ", "
              << sphere_center[1] << ", " << sphere_center[2]
              << "), r = " << sphere_radius
              << ", inliers = " << inliers->indices.size() << std::endl;
    return true;
}

int main()
{
    try
    {
        // ---------------- ZED 初始化  
        std::cout << "[INFO] Init ZED..." << std::endl;
        ZED Zed;
        Zed.init();
        Zed.setCamera();
        std::cout << "[INFO] ZED init done." << std::endl;

        // ---------------- TRT YOLO 初始化 ----------------
        const std::string engine_path = "/home/lyk/TensorRT-YOLO/examples/detect/models/ball.engine";
        std::cout << "[INFO] Loading TensorRT-YOLO engine: " << engine_path << std::endl;

        trtyolo::InferOption option;
        option.enableSwapRB(); // BGR -> RGB
        option.setNormalizeParams({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});

        std::cout << "[INFO] Create YOLO model..." << std::endl;
        trtyolo::DetectModel model(engine_path, option);
        std::cout << "[INFO] TensorRT-YOLO model created." << std::endl;

        // ---------------- OpenCV 窗口 ----------------
        cv::namedWindow("ZED + TRT-YOLO", cv::WINDOW_NORMAL);

        // ---------------- PCL 可视化窗口：左右两个视口 ----------------
        auto viewer = boost::make_shared<pcl::visualization::PCLVisualizer>("ZED Cloud Viewer");
        viewer->setBackgroundColor(0.0, 0.0, 0.0);

        // 创建两个 viewports：左 0~0.5，右 0.5~1.0
        int v_left = 0;
        int v_right = 1;
        viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v_left);
        viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v_right);

        // 左右都放一个坐标系，方便看
        viewer->addCoordinateSystem(0.5, "world_left", v_left);
        viewer->addCoordinateSystem(0.5, "world_right", v_right);

        
        viewer->initCameraParameters();
        viewer->setCameraPosition(
            0, 0, -5,  // 相机位置
            0, 0, 0,   // 观察目标
            0, -1, 0); // 上方向

        // 主循环
        while (!viewer->wasStopped())
        {
            // --------------- 获取 ZED 图像 + 点云 ---------------
            std::cout << "[INFO] Grab ZED frame..." << std::endl;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_color(new pcl::PointCloud<pcl::PointXYZRGB>);
            cv::Mat img;
            Zed.getCloud(cloud_color, img);

            if (img.empty())
            {
                std::cerr << "[WARN] ZED 图像为空，跳过这一帧" << std::endl;
                viewer->spinOnce(1);
                continue;
            }

            if (!cloud_color || cloud_color->empty())
            {
                std::cerr << "[WARN] 点云为空，跳过这一帧" << std::endl;
                viewer->spinOnce(1);
                continue;
            }

            // ★ 每帧更新深度缓存，供 coordinate_from_sl_point_cloud 使用
            Zed.update_sl_point_cloud();

            const int img_w = img.cols;
            const int img_h = img.rows;

            // 左视图：显示原始彩色点云
            if (!viewer->updatePointCloud<pcl::PointXYZRGB>(cloud_color, "cloud_left"))
            {
                viewer->addPointCloud<pcl::PointXYZRGB>(cloud_color, "cloud_left", v_left);
                viewer->setPointCloudRenderingProperties(
                    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_left");
            }

            // --------------- YOLO 推理 ---------------
            std::cout << "[INFO] Run YOLO predict..." << std::endl;
            trtyolo::Image timg(img.data, img.cols, img.rows);
            trtyolo::DetectRes det_res = model.predict(timg);
            std::cout << "[INFO] YOLO predict done. num=" << det_res.num << std::endl;

            // 默认右侧没有数据时，清掉旧的“球点云”
            if (viewer->contains("cloud_ball"))
            {
                viewer->removePointCloud("cloud_ball");
            }

            if (det_res.num == 0)
            {
                std::cout << "[INFO] 本帧未检测到目标" << std::endl;
                // 刷新可视化
                viewer->spinOnce(1);
                cv::imshow("ZED + TRT-YOLO", img);
                if (cv::waitKey(1) == 27)
                    break;
                continue;
            }

            if (det_res.boxes.size() < det_res.num ||
                det_res.scores.size() < det_res.num ||
                det_res.classes.size() < det_res.num)
            {
                std::cerr << "[ERROR] DetectRes 数组长度与 num 不一致" << std::endl;
                break;
            }

            // 选置信度最高的 bbox
            int best_idx = -1;
            float best_score = 0.0f;
            for (size_t i = 0; i < det_res.num; ++i)
            {
                float score = det_res.scores[i];
                if (score > best_score)
                {
                    best_score = score;
                    best_idx = static_cast<int>(i);
                }
            }

            if (best_idx < 0)
            {
                std::cerr << "[WARN] 未找到有效 bbox" << std::endl;
                viewer->spinOnce(1);
                cv::imshow("ZED + TRT-YOLO", img);
                if (cv::waitKey(1) == 27)
                    break;
                continue;
            }

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
                std::cerr << "[WARN] 裁剪后 bbox 无效" << std::endl;
                viewer->spinOnce(1);
                cv::imshow("ZED + TRT-YOLO", img);
                if (cv::waitKey(1) == 27)
                    break;
                continue;
            }

            cv::Rect rect(cv::Point(left, top), cv::Point(right, bottom));

            // --------------- 利用 bbox 提取局部点云（ROI）---------------
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_roi(new pcl::PointCloud<pcl::PointXYZ>);
            cloud_roi->reserve((rect.width + 1) * (rect.height + 1));

            int debug_cnt = 0;
            for (int v = rect.y; v <= rect.y + rect.height; ++v)
            {
                for (int u = rect.x; u <= rect.x + rect.width; ++u)
                {
                    float w[3] = {0.f, 0.f, 0.f};
                    Zed.coordinate_from_sl_point_cloud(cv::Point(u, v), w);

                    // 只过滤 NaN/Inf
                    if (!std::isfinite(w[0]) || !std::isfinite(w[1]) || !std::isfinite(w[2]))
                        continue;

                    // ❌ 不再过滤 w[2] <= 0，否则你的点会全部被干掉
                    // if (w[2] <= 0.0f) continue;

                    float dist = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
                    if (dist < 0.05f || dist > 5.0f) // 按实际场景调
                        continue;

                    if (debug_cnt < 5)
                    {
                        std::cout << "[DEBUG] ROI sample (" << u << "," << v << ") -> ("
                                  << w[0] << "," << w[1] << "," << w[2]
                                  << "), dist=" << dist << std::endl;
                        ++debug_cnt;
                    }

                    pcl::PointXYZ p;
                    p.x = w[0];
                    p.y = w[1];
                    p.z = w[2];
                    cloud_roi->push_back(p);
                }
            }

            std::cout << "[INFO] ROI cloud size = " << cloud_roi->size() << std::endl;

            // --------------- RANSAC 拟合球 + 右视图显示 inlier 点云 ---------------
            Eigen::Vector4f sphere_center;
            float sphere_radius = 0.0f;
            pcl::PointCloud<pcl::PointXYZ>::Ptr ball_cloud; // inlier 点云

            bool ok = fitSphereRANSAC(cloud_roi, sphere_center, sphere_radius, ball_cloud);

            if (!ok || !ball_cloud || ball_cloud->empty())
            {
                std::cerr << "[WARN] 球拟合失败，退回到原来 bbox 中心点单点深度" << std::endl;

                int cx = rect.x + rect.width / 2;
                int cy = rect.y + rect.height / 2;
                cx = std::max(0, std::min(cx, img_w - 1));
                cy = std::max(0, std::min(cy, img_h - 1));

                float world3[3] = {0.f, 0.f, 0.f};
                Zed.coordinate_from_sl_point_cloud(cv::Point(cx, cy), world3);
                float dist = std::sqrt(world3[0] * world3[0] +
                                       world3[1] * world3[1] +
                                       world3[2] * world3[2]);

                std::cout << "[INFO] Fallback center pixel: (" << cx << ", " << cy << ") 3D: ("
                          << world3[0] << ", " << world3[1] << ", " << world3[2]
                          << ") dist=" << dist << " m" << std::endl;

                cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);
                cv::circle(img, cv::Point(cx, cy), 4, cv::Scalar(0, 0, 255), -1);
                std::string text = "ball " + cv::format("%.2f m", dist);
                cv::putText(img, text, cv::Point(rect.x, std::max(0, rect.y - 5)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            }
            else
            {
                float X = sphere_center[0];
                float Y = sphere_center[1];
                float Zc = sphere_center[2];
                float dist = std::sqrt(X * X + Y * Y + Zc * Zc);

                int cx = rect.x + rect.width / 2;
                int cy = rect.y + rect.height / 2;
                cx = std::max(0, std::min(cx, img_w - 1));
                cy = std::max(0, std::min(cy, img_h - 1));

                std::cout << "[INFO] Ball RANSAC center 3D: (" << X << ", " << Y << ", " << Zc
                          << ") dist=" << dist << " m" << std::endl;

                cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);
                cv::circle(img, cv::Point(cx, cy), 4, cv::Scalar(0, 0, 255), -1);
                std::string text = "ball " + cv::format("%.2f m", dist);
                cv::putText(img, text, cv::Point(rect.x, std::max(0, rect.y - 5)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

                // 右视图显示拟合到球上的 inlier 点云
                viewer->addPointCloud<pcl::PointXYZ>(ball_cloud, "cloud_ball", v_right);
                viewer->setPointCloudRenderingProperties(
                    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud_ball");
            }

            // --------------- 刷新 3D & 2D ---------------
            viewer->spinOnce(1);
            cv::imshow("ZED + TRT-YOLO", img);
            int key = cv::waitKey(1);
            if (key == 27) // ESC
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
