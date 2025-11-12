#include <iostream>
#include <cmath>
#include <memory>
//opencv相关
#include <opencv2/opencv.hpp>
//pcl相关
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "Zed.hpp"    
#include "trtyolo.hpp" // TensorRT-YOLO 头文件

bool fitSphereRANSAC(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_roi,Eigen::Vector4f &sphere_center, float &sphere_radius, pcl::PointCloud<pcl::PointXYZ> &inlier_cloud)
{
    /**
     * @brief 错误检查
     * 
     */
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
    seg.setDistanceThreshold(0.01f);
    seg.setInputCloud(cloud_roi);

    pcl::ModelCoefficients coeffs;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    seg.segment(*inliers, coeffs);
    if (inliers->)
}
