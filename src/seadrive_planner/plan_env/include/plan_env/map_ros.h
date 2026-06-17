#ifndef _MAP_ROS_H
#define _MAP_ROS_H

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf/transform_listener.h>

#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>

#include <memory>
#include <random>

using std::shared_ptr;
using std::normal_distribution;
using std::default_random_engine;

namespace fast_planner {
class SDFMap;

class MapROS {
public:
  MapROS();
  ~MapROS();
  void setMap(SDFMap* map);
  void init();

private:
  void depthPoseCallback(const sensor_msgs::ImageConstPtr& img,
                         const geometry_msgs::PoseStampedConstPtr& pose);
  void cloudPoseCallback(const sensor_msgs::PointCloud2ConstPtr& msg,
                         const geometry_msgs::PoseStampedConstPtr& pose);
  void updateESDFCallback(const ros::TimerEvent& /*event*/);
  void visCallback(const ros::TimerEvent& /*event*/);

  void publishMapAll();
  void publishMapLocal();
  void publishESDF();
  void publishExploredGrid();
  void publishUpdateRange();
  void publishUnknown();
  void publishDepth();

  void proessDepthImage();

  SDFMap* map_;
  // may use ExactTime?
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PoseStamped>
      SyncPolicyImagePose;
  typedef shared_ptr<message_filters::Synchronizer<SyncPolicyImagePose>> SynchronizerImagePose;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2,
                                                          geometry_msgs::PoseStamped>
      SyncPolicyCloudPose;
  typedef shared_ptr<message_filters::Synchronizer<SyncPolicyCloudPose>> SynchronizerCloudPose;

  ros::NodeHandle node_;
  shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> depth_sub_;
  shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> cloud_sub_;
  shared_ptr<message_filters::Subscriber<geometry_msgs::PoseStamped>> pose_sub_;
  SynchronizerImagePose sync_image_pose_;
  SynchronizerCloudPose sync_cloud_pose_;
  shared_ptr<tf::TransformListener> tf_listener_;

  ros::Publisher map_local_pub_, map_local_inflate_pub_, esdf_pub_, map_all_pub_, unknown_pub_,
      explored_grid_pub_, update_range_pub_, depth_pub_;
  ros::Timer esdf_timer_, vis_timer_;

  // params, depth projection
  double cx_, cy_, fx_, fy_;
  double depth_filter_maxdist_, depth_filter_mindist_;
  int depth_filter_margin_;
  double k_depth_scaling_factor_;
  int skip_pixel_;
  string frame_id_;
  // msg publication
  double esdf_slice_height_;
  double occ_z_min_;  // occupancy_local 可视化的最低高度阈值
  double visualization_truncate_height_, visualization_truncate_low_;
  double coverage_z_min_, coverage_z_max_;  // coverage 统计使用的 z 体素带
  bool show_esdf_time_, show_occ_time_;
  bool show_all_map_;

  // Hard filter to remove water-surface noise points before they are fused
  // into the SDF map (to avoid water clutter showing up in explored_grid).
  bool water_filter_enable_;
  double water_filter_z_min_;
  double water_filter_z_max_;
  // Hard filter in sensor frame: points closer than this distance are dropped.
  // It helps suppress near-field reflection artifacts from camera/lidar housing.
  double cloud_filter_mindist_;

  // explored grid publication (2D projection of 3D SDFMap)
  // Additional yaw correction applied to sensor->world camera_q_ before feeding into SDFMap.
  // This is useful when the lidar azimuth definition differs by a constant yaw offset.
  double camera_yaw_offset_;

  // If true, apply quaternion conjugate (inverse rotation) before feeding into SDFMap.
  // This swaps the expected rotation direction convention (sensor->world vs world->sensor).
  bool camera_q_conjugate_;

  double explored_grid_z_;
  double explored_grid_z_half_thickness_;
  double explored_grid_obs_z_min_;
  double explored_grid_obs_z_max_;
  double explored_grid_obs_occ_log_soft_thresh_;
  double explored_grid_pub_dt_;
  ros::Time last_explored_grid_pub_;
  bool explored_grid_debug_;
  double explored_grid_debug_throttle_;

  // PointCloud debug (to verify cloud->map correlation)
  bool cloud_debug_;
  double cloud_debug_throttle_;

  // data
  // flags of map state
  bool local_updated_, esdf_need_update_;
  // input
  Eigen::Vector3d camera_pos_;
  Eigen::Quaterniond camera_q_;
  unique_ptr<cv::Mat> depth_image_;
  vector<Eigen::Vector3d> proj_points_;
  int proj_points_cnt;
  double fuse_time_, esdf_time_, max_fuse_time_, max_esdf_time_;
  int fuse_num_, esdf_num_;
  pcl::PointCloud<pcl::PointXYZ> point_cloud_;

  normal_distribution<double> rand_noise_;
  default_random_engine eng_;

  ros::Time map_start_time_;

  std::string log_filename;

  friend SDFMap;
};
}

#endif