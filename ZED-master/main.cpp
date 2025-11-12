#include <iostream>

#include <opencv2/opencv.hpp>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "Zed.hpp"

using namespace std;
using namespace cv;
using namespace pcl;

// ctrl c中断
#include <signal.h>
bool ctrl_c_pressed = false;
void ctrlc(int)
{
    ctrl_c_pressed = true;
}

int main()
{
    // Ctrl C 中断
    signal(SIGINT, ctrlc);

    ZED Zed;
    Zed.init();
    Zed.setCamera();

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Viewer_PCL"));

    // 设置默认的坐标系
    viewer->addCoordinateSystem(1.0);
    // 设置固定的元素。红色是X轴，绿色是Y轴，蓝色是Z
    viewer->addLine(pcl::PointXYZ(0, 0, 0), pcl::PointXYZ(10, 0, 0), "x");
    viewer->addLine(pcl::PointXYZ(0, 0, 0), pcl::PointXYZ(0, 5, 0), "y");
    viewer->addLine(pcl::PointXYZ(0, 0, 0), pcl::PointXYZ(0, 0, 2), "z");

    while (1)
    {
        if (ctrl_c_pressed == true)
            break;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        cv::Mat img;
        Zed.getCloud(cloud, img);

        timeval tt1, tt2;
        gettimeofday(&tt1, NULL);
        {
            Zed.update_sl_point_cloud();
            for (int i = -10; i <= 10; i++)
            {
                for (int j = -10; j <= 10; j++)
                {
                    float world3[3];
                    Zed.coordinate_from_sl_point_cloud(Point(img.size().width / 2 + i, img.size().height / 2 + j), world3);
                    // printf("%f %f %f\n", world3[0], world3[1], world3[2]);
                }
            }
        }
        gettimeofday(&tt2, NULL);
        printf("time: %f\n", (tt2.tv_sec - tt1.tv_sec) * 1000.0 + (tt2.tv_usec - tt1.tv_usec) / 1000.0);

        float distance = Zed.calcDistance(Point(img.size().width / 2, img.size().height / 2));
        printf("距离为%f\n", distance);
        imshow("img", img);
        waitKey(3);

        viewer->addPointCloud(cloud, "cloud");
        viewer->spinOnce(3);
        viewer->removePointCloud("cloud");
    }

    Zed.close();
}



// #include <iostream>

// #include <opencv2/opencv.hpp>

// #include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/visualization/cloud_viewer.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>

// #include "Zed.hpp"

// #include <algorithm> // [新增] median 计算需要
// #include <vector>    // [新增]
// #include <cmath>     // [新增] std::isfinite

// using namespace std;
// using namespace cv;
// using namespace pcl;

// // ctrl c中断
// #include <signal.h>
// bool ctrl_c_pressed = false;
// void ctrlc(int)
// {
//     ctrl_c_pressed = true;
// }

// // [新增] 简单中位数工具函数（鲁棒）
// template <typename T>
// T median_value(std::vector<T> v)
// {
//     if (v.empty())
//         return std::numeric_limits<T>::quiet_NaN();
//     std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
//     return v[v.size() / 2];
// }

// int main()
// {
//     // Ctrl C 中断
//     signal(SIGINT, ctrlc);

//     ZED Zed;
//     Zed.init();
//     Zed.setCamera();

//     boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Viewer_PCL"));

//     // 设置默认的坐标系
//     viewer->addCoordinateSystem(1.0);
//     // 设置固定的元素。红色是X轴，绿色是Y轴，蓝色是Z
//     viewer->addLine(pcl::PointXYZ(0, 0, 0), pcl::PointXYZ(10, 0, 0), "x");
//     viewer->addLine(pcl::PointXYZ(0, 0, 0), pcl::PointXYZ(0, 5, 0), "y");
//     viewer->addLine(pcl::PointXYZ(0, 0, 0), pcl::PointXYZ(0, 0, 2), "z");

//     while (1)
//     {
//         if (ctrl_c_pressed == true)
//             break;

//         pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
//         cv::Mat img;
//         Zed.getCloud(cloud, img);

//         if (img.empty())
//         {
//             std::cerr << "图像为空，跳过该帧" << std::endl;
//             continue;
//         }

//         // ===== 计算耗时起点 =====
//         timeval tt1, tt2;
//         gettimeofday(&tt1, NULL);

//         // 更新与该帧对应的 ZED 点云/深度缓存
//         Zed.update_sl_point_cloud();

//         // 以图像中心附近 21x21 小窗做鲁棒距离估计
//         int cx = img.cols / 2;
//         int cy = img.rows / 2;
//         std::vector<float> dists;
//         dists.reserve(441);

//         for (int j = -10; j <= 10; ++j)
//         {
//             for (int i = -10; i <= 10; ++i)
//             {
//                 int x = cx + i;
//                 int y = cy + j;
//                 if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
//                     continue;
//                 float world3[3] = {0, 0, 0};
//                 Zed.coordinate_from_sl_point_cloud(cv::Point(x, y), world3);
//                 if (std::isfinite(world3[0]) && std::isfinite(world3[1]) && std::isfinite(world3[2]))
//                 {
//                     float d = std::sqrt(world3[0] * world3[0] + world3[1] * world3[1] + world3[2] * world3[2]);
//                     if (std::isfinite(d) && d > 0.f)
//                         dists.push_back(d);
//                 }
//             }
//         }
//         float robust_center = median_value(dists);

//         // 同时用原生 calcDistance 在图像中心取一个对比值
//         float d_calc = Zed.calcDistance(cv::Point(cx, cy));

//         gettimeofday(&tt2, NULL);
//         printf("time: %.3f ms\n", (tt2.tv_sec - tt1.tv_sec) * 1000.0 + (tt2.tv_usec - tt1.tv_usec) / 1000.0);

//         // 打印：优先展示鲁棒中位数，其次展示 calcDistance
//         if (std::isfinite(robust_center) && robust_center > 0.f)
//         {
//             printf("中心(21x21)鲁棒距离: %.3f m\n", robust_center);
//         }
//         else
//         {
//             printf("中心(21x21)鲁棒距离无效\n");
//         }

//         if (!std::isfinite(d_calc) || d_calc <= 0.f)
//         {
//             printf("calcDistance 无效(返回 %s)\n",
//                    std::isnan(d_calc) ? "NaN" : (std::isinf(d_calc) ? (d_calc > 0 ? "+Inf" : "-Inf") : "非正"));
//         }
//         else
//         {
//             printf("calcDistance 距离: %.3f m\n", d_calc);
//         }

//         imshow("img", img);
//         waitKey(3);

//         viewer->addPointCloud(cloud, "cloud");
//         viewer->spinOnce(3);
//         viewer->removePointCloud("cloud");
//     }

//     Zed.close();
// }
