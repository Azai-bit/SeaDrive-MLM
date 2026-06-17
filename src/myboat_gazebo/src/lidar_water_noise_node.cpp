#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <tf/transform_listener.h>

#include <cstdint>
#include <random>
#include <ctime>

ros::Publisher g_pub_cloud;
tf::TransformListener* g_tf_listener = nullptr;

bool   g_enable_water_noise      = true;
double g_water_level_world       = 0.0;   // 水面在 world 坐标系下的 z
double g_water_lidar_max_range   = 5.0;   // 仅在传感器 5m 内添加水面噪声
double g_water_lidar_density     = 0.7;   // 前方半圆区域每平方米噪声点数量
double g_water_lidar_z_std       = 0.05;  // 水面点的垂直方向噪声（米）
double g_water_lidar_fov_deg     = 180.0; // 仅在前方若干度范围内添加

// RViz point cloud colors (store as int for ROS param compatibility)
// - orig_*: color of the original input points
// - water_*: color of the synthetic water-noise points
int g_orig_color_r = 100;
int g_orig_color_g = 100;
int g_orig_color_b = 100;
int g_water_color_r = 255;
int g_water_color_g = 0;
int g_water_color_b = 0;

std::default_random_engine       g_rng(static_cast<unsigned>(std::time(nullptr)));
std::uniform_real_distribution<double> g_uni01(0.0, 1.0);
std::normal_distribution<double>       g_gauss01(0.0, 1.0);

void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg,
                   const std::string& output_frame)
{
  // 直接复制输入点云
  pcl::PointCloud<pcl::PointXYZ> cloud_in;
  pcl::fromROSMsg(*msg, cloud_in);

  // Output as RGB point cloud so RViz can colorize automatically.
  pcl::PointCloud<pcl::PointXYZRGB> cloud_out_rgb;
  cloud_out_rgb.points.reserve(cloud_in.points.size());
  for (const auto& p : cloud_in.points) {
    pcl::PointXYZRGB pt;
    pt.x = p.x;
    pt.y = p.y;
    pt.z = p.z;
    pt.r = static_cast<uint8_t>(g_orig_color_r);
    pt.g = static_cast<uint8_t>(g_orig_color_g);
    pt.b = static_cast<uint8_t>(g_orig_color_b);
    cloud_out_rgb.points.push_back(pt);
  }

  if (g_enable_water_noise && g_tf_listener) {
    tf::StampedTransform T_wl;  // world -> lidar
    try {
      // 使用最新变换，避免时间同步问题
      g_tf_listener->lookupTransform("world", msg->header.frame_id,
                                     ros::Time(0), T_wl);
    } catch (tf::TransformException& ex) {
      ROS_WARN_THROTTLE(2.0,
                        "[lidar_water_noise] TF lookup failed: %s, pass-through only.",
                        ex.what());
      // 只发布原始点云（带颜色）
      sensor_msgs::PointCloud2 out_msg;
      cloud_out_rgb.width = cloud_out_rgb.points.size();
      cloud_out_rgb.height = 1;
      cloud_out_rgb.is_dense = false;
      pcl::toROSMsg(cloud_out_rgb, out_msg);
      out_msg.header = msg->header;
      g_pub_cloud.publish(out_msg);
      return;
    }

    tf::Vector3 lidar_pos_w = T_wl.getOrigin();

    // 估算前方半圆区域面积
    double R = g_water_lidar_max_range;
    double area_front_semi_circle = 0.5 * M_PI * R * R;
    int noise_num = static_cast<int>(area_front_semi_circle * g_water_lidar_density);
    if (noise_num < 0) noise_num = 0;

    // LIDAR x 轴在 world 下的方向作为“前方”方向
    tf::Matrix3x3 R_wl(T_wl.getRotation());
    tf::Vector3 x_axis_w = R_wl * tf::Vector3(1.0, 0.0, 0.0);
    double yaw = std::atan2(x_axis_w.y(), x_axis_w.x());

    // world -> lidar 变换
    tf::Transform T_lw = T_wl.inverse();

    double half_fov_rad = 0.5 * g_water_lidar_fov_deg * M_PI / 180.0;

    for (int k = 0; k < noise_num; ++k) {
      // 在 [0, R] 上按面积均匀采样半径
      double r = R * std::sqrt(g_uni01(g_rng));
      // 在 [-FOV/2, FOV/2] 上采样偏航角
      double alpha = -half_fov_rad + g_uni01(g_rng) * (2.0 * half_fov_rad);

      double dir_yaw = yaw + alpha;
      double dx = std::cos(dir_yaw);
      double dy = std::sin(dir_yaw);

      // 水面上的世界坐标
      double xw = lidar_pos_w.x() + r * dx;
      double yw = lidar_pos_w.y() + r * dy;
      double zw = g_water_level_world + g_gauss01(g_rng) * g_water_lidar_z_std;

      tf::Vector3 Pw(xw, yw, zw);
      // 转回 LIDAR 坐标系
      tf::Vector3 Pl = T_lw * Pw;

      // 再做一次距离检查，避免超过 R
      double dist = Pl.length();
      if (dist > R) continue;

      pcl::PointXYZ pt;
      pt.x = static_cast<float>(Pl.x());
      pt.y = static_cast<float>(Pl.y());
      pt.z = static_cast<float>(Pl.z());
      pcl::PointXYZRGB pt_rgb;
      pt_rgb.x = pt.x;
      pt_rgb.y = pt.y;
      pt_rgb.z = pt.z;
      pt_rgb.r = static_cast<uint8_t>(g_water_color_r);
      pt_rgb.g = static_cast<uint8_t>(g_water_color_g);
      pt_rgb.b = static_cast<uint8_t>(g_water_color_b);
      cloud_out_rgb.points.push_back(pt_rgb);
    }
  }

  cloud_out_rgb.width  = cloud_out_rgb.points.size();
  cloud_out_rgb.height = 1;
  cloud_out_rgb.is_dense = false;

  sensor_msgs::PointCloud2 out_msg;
  pcl::toROSMsg(cloud_out_rgb, out_msg);
  out_msg.header = msg->header;
  g_pub_cloud.publish(out_msg);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "lidar_water_noise_node");
  ros::NodeHandle nh("~");

  std::string input_topic;
  std::string output_topic;

  nh.param<std::string>("input_topic", input_topic,
                        std::string("/myboat/sensors/lidar_wamv/points"));
  nh.param<std::string>("output_topic", output_topic,
                        std::string("/myboat/sensors/lidar_wamv/points_with_water"));

  nh.param("enable_water_noise", g_enable_water_noise, true);
  nh.param("water_level", g_water_level_world, 0.0);
  nh.param("water_lidar_max_range", g_water_lidar_max_range, 5.0);
  nh.param("water_lidar_density", g_water_lidar_density, 0.7);
  nh.param("water_lidar_z_std", g_water_lidar_z_std, 0.05);
  nh.param("water_lidar_fov_deg", g_water_lidar_fov_deg, 180.0);

  nh.param("orig_color_r", g_orig_color_r, g_orig_color_r);
  nh.param("orig_color_g", g_orig_color_g, g_orig_color_g);
  nh.param("orig_color_b", g_orig_color_b, g_orig_color_b);
  nh.param("water_color_r", g_water_color_r, g_water_color_r);
  nh.param("water_color_g", g_water_color_g, g_water_color_g);
  nh.param("water_color_b", g_water_color_b, g_water_color_b);

  g_tf_listener = new tf::TransformListener();

  g_pub_cloud = nh.advertise<sensor_msgs::PointCloud2>(output_topic, 1);

  // 使用 boost::bind 传递额外参数（此处暂未使用 output_frame，但保留接口）
  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>(
      input_topic, 1,
      boost::bind(&cloudCallback, _1, std::string("")));

  ros::spin();

  delete g_tf_listener;
  return 0;
}

