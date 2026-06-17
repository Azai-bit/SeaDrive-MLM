#include <plan_env/sdf_map.h>
#include <plan_env/map_ros.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <visualization_msgs/Marker.h>
#include <nav_msgs/OccupancyGrid.h>
#include <ros/package.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <ctime>
namespace fast_planner {
MapROS::MapROS() {
}

MapROS::~MapROS() {
}

void MapROS::setMap(SDFMap* map) {
  this->map_ = map;
}

void MapROS::init() {
  node_.param("map_ros/fx", fx_, -1.0);
  node_.param("map_ros/fy", fy_, -1.0);
  node_.param("map_ros/cx", cx_, -1.0);
  node_.param("map_ros/cy", cy_, -1.0);
  node_.param("map_ros/depth_filter_maxdist", depth_filter_maxdist_, -1.0);
  node_.param("map_ros/depth_filter_mindist", depth_filter_mindist_, -1.0);
  node_.param("map_ros/depth_filter_margin", depth_filter_margin_, -1);
  node_.param("map_ros/k_depth_scaling_factor", k_depth_scaling_factor_, -1.0);
  node_.param("map_ros/skip_pixel", skip_pixel_, -1);

  node_.param("map_ros/esdf_slice_height", esdf_slice_height_, -0.1);
  node_.param("map_ros/visualization_truncate_height", visualization_truncate_height_, -0.1);
  node_.param("map_ros/visualization_truncate_low", visualization_truncate_low_, -0.1);
  node_.param("map_ros/show_occ_time", show_occ_time_, false);
  node_.param("map_ros/show_esdf_time", show_esdf_time_, false);
  node_.param("map_ros/show_all_map", show_all_map_, false);
  node_.param("map_ros/occ_z_min", occ_z_min_, -1e9);  // occupancy_local 的最低可视化高度阈值
  node_.param("map_ros/frame_id", frame_id_, string("world"));

  // Water-surface hard filtering:
  // If enabled, points whose world-z is within [water_filter_z_min, water_filter_z_max]
  // will be discarded before fusing into the SDF map.
  node_.param("map_ros/water_filter_enable", water_filter_enable_, false);
  node_.param("map_ros/water_filter_z_min", water_filter_z_min_, -0.2);
  node_.param("map_ros/water_filter_z_max", water_filter_z_max_, 0.2);
  node_.param("map_ros/cloud_filter_mindist", cloud_filter_mindist_, 2.0);

  // Lidar azimuth convention correction (constant yaw).
  // SDFMap active-ray model assumes azimuth 0 aligns with sensor +X.
  // If your lidar definition is shifted by a constant yaw, this parameter fixes it.
  node_.param("map_ros/camera_yaw_offset", camera_yaw_offset_, 0.0);

  // Rotation direction convention fix.
  // If your integration expects world->sensor but you provide sensor->world (or vice versa),
  // enabling this will conjugate the camera quaternion before feeding into SDFMap.
  node_.param("map_ros/camera_q_conjugate", camera_q_conjugate_, false);

  // 2D explored grid (OccupancyGrid) publication
  // Project the 3D SDFMap into an x-y occupancy grid around this fixed z height.
  node_.param("map_ros/explored_grid_z", explored_grid_z_, 0.5);
  node_.param("map_ros/explored_grid_z_half_thickness", explored_grid_z_half_thickness_, 0.2);
  node_.param("map_ros/explored_grid_obs_z_min", explored_grid_obs_z_min_, occ_z_min_);
  node_.param("map_ros/explored_grid_obs_z_max", explored_grid_obs_z_max_,
              visualization_truncate_height_);
  node_.param("map_ros/explored_grid_obs_occ_log_soft_thresh", explored_grid_obs_occ_log_soft_thresh_, 0.3);
  node_.param("map_ros/explored_grid_pub_dt", explored_grid_pub_dt_, 0.5);
  node_.param("map_ros/explored_grid_debug", explored_grid_debug_, false);
  node_.param("map_ros/explored_grid_debug_throttle", explored_grid_debug_throttle_, 2.0);
  last_explored_grid_pub_ = ros::Time(0);

  node_.param("map_ros/cloud_debug", cloud_debug_, false);
  node_.param("map_ros/cloud_debug_throttle", cloud_debug_throttle_, 2.0);

  proj_points_.resize(640 * 480 / (skip_pixel_ * skip_pixel_));
  point_cloud_.points.resize(640 * 480 / (skip_pixel_ * skip_pixel_));
  // proj_points_.reserve(640 * 480 / map_->mp_->skip_pixel_ / map_->mp_->skip_pixel_);
  proj_points_cnt = 0;

  local_updated_ = false;
  esdf_need_update_ = false;
  fuse_time_ = 0.0;
  esdf_time_ = 0.0;
  max_fuse_time_ = 0.0;
  max_esdf_time_ = 0.0;
  fuse_num_ = 0;
  esdf_num_ = 0;
  depth_image_.reset(new cv::Mat);

  rand_noise_ = normal_distribution<double>(0, 0.1);
  random_device rd;
  eng_ = default_random_engine(rd());

  esdf_timer_ = node_.createTimer(ros::Duration(0.05), &MapROS::updateESDFCallback, this);
  vis_timer_ = node_.createTimer(ros::Duration(0.05), &MapROS::visCallback, this);

  map_all_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/occupancy_all", 10);
  map_local_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/occupancy_local", 10);
  map_local_inflate_pub_ =
      node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/occupancy_local_inflate", 10);
  unknown_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/unknown", 10);
  explored_grid_pub_ =
      node_.advertise<nav_msgs::OccupancyGrid>("/sdf_map/explored_grid", 1, true);
  esdf_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/esdf", 10);
  update_range_pub_ = node_.advertise<visualization_msgs::Marker>("/sdf_map/update_range", 10);
  depth_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/depth_cloud", 10);

  depth_sub_.reset(new message_filters::Subscriber<sensor_msgs::Image>(node_, "/map_ros/depth", 50));
  cloud_sub_.reset(
      new message_filters::Subscriber<sensor_msgs::PointCloud2>(node_, "/map_ros/cloud", 50));
  pose_sub_.reset(
      new message_filters::Subscriber<geometry_msgs::PoseStamped>(node_, "/map_ros/pose", 25));
  tf_listener_.reset(new tf::TransformListener());

  sync_image_pose_.reset(new message_filters::Synchronizer<MapROS::SyncPolicyImagePose>(
      MapROS::SyncPolicyImagePose(100), *depth_sub_, *pose_sub_));
  sync_image_pose_->registerCallback(boost::bind(&MapROS::depthPoseCallback, this, _1, _2));
  sync_cloud_pose_.reset(new message_filters::Synchronizer<MapROS::SyncPolicyCloudPose>(
      MapROS::SyncPolicyCloudPose(100), *cloud_sub_, *pose_sub_));
  sync_cloud_pose_->registerCallback(boost::bind(&MapROS::cloudPoseCallback, this, _1, _2));

  map_start_time_ = ros::Time::now();
  
  std::time_t now = std::time(nullptr);
  std::tm* t = std::localtime(&now);
  char buf[64];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", t);
  std::string log_dir = ros::package::getPath("exploration_manager") + "/resource";
  if (log_dir.empty()) {
    log_dir = "/tmp";
  }
  log_filename = log_dir + "/reserve_sensor_known_volume_coverage_time_" + std::string(buf) + ".csv";

  // Initialize CSV header once per run.
  std::ofstream csv(log_filename, std::ios::out);
  if (csv.is_open()) {
    csv << "time,known_volume,total_volume,coverage" << std::endl;
  }
}

void MapROS::visCallback(const ros::TimerEvent& e) {
  publishMapLocal();
  if (show_all_map_) {
    // Limit the frequency of all map
    static double tpass = 0.0;
    tpass += (e.current_real - e.last_real).toSec();
    if (tpass > 0.1) {
      publishMapAll();
      tpass = 0.0;
    }
  }
  publishUnknown();
  publishESDF();
  publishExploredGrid();

  // publishUpdateRange();
  // publishDepth();
}

void MapROS::publishExploredGrid() {
  const ros::Time now = ros::Time::now();
  if (!last_explored_grid_pub_.isZero()) {
    const double dt = (now - last_explored_grid_pub_).toSec();
    if (dt < explored_grid_pub_dt_) return;
  }
  last_explored_grid_pub_ = now;

  if (!map_ || !map_->mp_ || !map_->md_) return;

  nav_msgs::OccupancyGrid grid;
  grid.header.stamp = now;
  grid.header.frame_id = frame_id_;

  const int width = map_->mp_->map_voxel_num_(0);
  const int height = map_->mp_->map_voxel_num_(1);
  const double res = map_->mp_->resolution_;
  grid.info.resolution = res;
  grid.info.width = width;
  grid.info.height = height;
  grid.info.origin.position.x = map_->mp_->map_origin_(0);
  grid.info.origin.position.y = map_->mp_->map_origin_(1);
  grid.info.origin.position.z = 0.0;
  grid.info.origin.orientation.x = 0.0;
  grid.info.origin.orientation.y = 0.0;
  grid.info.origin.orientation.z = 0.0;
  grid.info.origin.orientation.w = 1.0;

  grid.data.assign(width * height, static_cast<int8_t>(-1));

  // Determine z slice (voxel indices)
  const double z_center = explored_grid_z_;
  const double z_min_world = z_center - explored_grid_z_half_thickness_;
  const double z_max_world = z_center + explored_grid_z_half_thickness_;
  int z_min_idx = static_cast<int>(
      floor((z_min_world - map_->mp_->map_origin_(2)) * map_->mp_->resolution_inv_));
  int z_max_idx = static_cast<int>(
      floor((z_max_world - map_->mp_->map_origin_(2)) * map_->mp_->resolution_inv_));
  z_min_idx = std::max(0, std::min(z_min_idx, map_->mp_->map_voxel_num_(2) - 1));
  z_max_idx = std::max(0, std::min(z_max_idx, map_->mp_->map_voxel_num_(2) - 1));
  if (z_max_idx < z_min_idx) std::swap(z_min_idx, z_max_idx);

  // Occupied projection uses a wider z band than explored slice, to better
  // recover obstacle footprint consistent with 3D point cloud occupancy.
  int obs_z_min_idx = static_cast<int>(
      floor((explored_grid_obs_z_min_ - map_->mp_->map_origin_(2)) * map_->mp_->resolution_inv_));
  int obs_z_max_idx = static_cast<int>(
      floor((explored_grid_obs_z_max_ - map_->mp_->map_origin_(2)) * map_->mp_->resolution_inv_));
  obs_z_min_idx = std::max(0, std::min(obs_z_min_idx, map_->mp_->map_voxel_num_(2) - 1));
  obs_z_max_idx = std::max(0, std::min(obs_z_max_idx, map_->mp_->map_voxel_num_(2) - 1));
  if (obs_z_max_idx < obs_z_min_idx) std::swap(obs_z_min_idx, obs_z_max_idx);

  const double log_known_thresh = map_->mp_->clamp_min_log_ - 1e-3;  // explored if occ > this

  int cnt_occ_log = 0;
  int cnt_occ_inflate = 0;
  int cnt_occ_inflate_only = 0;
  int cnt_explored = 0;
  int cnt_unknown = 0;
  double max_occ_log_seen = -1e9;
  double min_occ_log_seen = 1e9;

  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      bool explored = false;
      bool occupied = false;
      bool occupied_inflate = false;
      double max_occ_log_thin = -1e18;
      for (int z = z_min_idx; z <= z_max_idx; ++z) {
        const int adr = map_->toAddress(x, y, z);
        const double occ_log = map_->md_->occupancy_buffer_[adr];
        max_occ_log_seen = std::max(max_occ_log_seen, occ_log);
        min_occ_log_seen = std::min(min_occ_log_seen, occ_log);
        max_occ_log_thin = std::max(max_occ_log_thin, occ_log);
        if (occ_log > map_->mp_->min_occupancy_log_) occupied = true;
        if (occ_log > log_known_thresh) explored = true;
      }

      // Also track inflated occupancy (often matches visualized obstacle footprint better)
      for (int z = obs_z_min_idx; z <= obs_z_max_idx; ++z) {
        const int adr = map_->toAddress(x, y, z);
        if (map_->md_->occupancy_buffer_inflate_[adr] == 1) {
          occupied_inflate = true;
          break;
        }
      }

      const int idx2d = y * width + x;
      // Conservative obstacle rule:
      // - hard occupied evidence from thin slice
      // - inflate can trigger obstacle only if thin-slice occ_log is above a soft threshold
      const bool inflate_accept =
          occupied_inflate && explored && (max_occ_log_thin >= explored_grid_obs_occ_log_soft_thresh_);
      const bool occupied_any = occupied || inflate_accept;
      const bool obstacle_from_inflate_only = (!occupied) && inflate_accept;
      grid.data[idx2d] = occupied_any ? 100 : (explored ? 0 : static_cast<int8_t>(-1));

      if (grid.data[idx2d] == 100) cnt_occ_log++;
      else if (grid.data[idx2d] == 0) cnt_explored++;
      else cnt_unknown++;
      if (occupied_inflate) cnt_occ_inflate++;
      if (obstacle_from_inflate_only) cnt_occ_inflate_only++;
    }
  }

  explored_grid_pub_.publish(grid);

  if (explored_grid_debug_) {
    ROS_WARN_THROTTLE(explored_grid_debug_throttle_,
                      "[explored_grid] size=%dx%d res=%.3f origin=(%.2f,%.2f,%.2f) "
                      "explored_z=%.2f slice=[%d,%d] obs_z=[%.2f,%.2f] idx=[%d,%d] "
                      "thresh_occ_log=%.3f thresh_known_log=%.3f soft_occ_log=%.3f "
                      "cells: occ(any)=%d occ(inflate)=%d occ(inflate_only)=%d explored=%d unknown=%d "
                      "occ_log_seen[min,max]=[%.3f,%.3f]",
                      width, height, res,
                      map_->mp_->map_origin_(0), map_->mp_->map_origin_(1), map_->mp_->map_origin_(2),
                      explored_grid_z_, z_min_idx, z_max_idx,
                      explored_grid_obs_z_min_, explored_grid_obs_z_max_, obs_z_min_idx, obs_z_max_idx,
                      map_->mp_->min_occupancy_log_, log_known_thresh,
                      explored_grid_obs_occ_log_soft_thresh_,
                      cnt_occ_log, cnt_occ_inflate, cnt_occ_inflate_only, cnt_explored, cnt_unknown,
                      min_occ_log_seen, max_occ_log_seen);
  }
}

void MapROS::updateESDFCallback(const ros::TimerEvent& /*event*/) {
  if (!esdf_need_update_) return;
  auto t1 = ros::Time::now();

  map_->updateESDF3d();
  esdf_need_update_ = false;

  auto t2 = ros::Time::now();
  esdf_time_ += (t2 - t1).toSec();
  max_esdf_time_ = max(max_esdf_time_, (t2 - t1).toSec());
  esdf_num_++;
  if (show_esdf_time_)
    ROS_WARN("ESDF t: cur: %lf, avg: %lf, max: %lf", (t2 - t1).toSec(), esdf_time_ / esdf_num_,
             max_esdf_time_);
}

void MapROS::depthPoseCallback(const sensor_msgs::ImageConstPtr& img,
                               const geometry_msgs::PoseStampedConstPtr& pose) {
  camera_pos_(0) = pose->pose.position.x;
  camera_pos_(1) = pose->pose.position.y;
  camera_pos_(2) = pose->pose.position.z;
  if (!map_->isInMap(camera_pos_))  // exceed mapped region
    return;

  camera_q_ = Eigen::Quaterniond(pose->pose.orientation.w, pose->pose.orientation.x,
                                 pose->pose.orientation.y, pose->pose.orientation.z);
  if (std::abs(camera_yaw_offset_) > 1e-9) {
    // Apply constant yaw correction on sensor->world camera quaternion.
    // SDFMap active-ray model uses azimuth with 0 aligned to sensor +X.
    Eigen::Quaterniond q_yaw(Eigen::AngleAxisd(camera_yaw_offset_, Eigen::Vector3d::UnitZ()));
    // Left-multiply to apply rotation around world Z (matches observed "around Z" mismatch).
    camera_q_ = q_yaw * camera_q_;
  }
  if (camera_q_conjugate_) {
    camera_q_ = camera_q_.conjugate();
  }
  // 仿真中该 topic 实际上传的是 RGB 图像而非深度图，这里只在编码为真正的深度格式时才处理；
  // 否则直接跳过，避免在 proessDepthImage 中按 16UC1 访问 RGB 数据导致段错误。
  if (img->encoding != sensor_msgs::image_encodings::TYPE_16UC1 &&
      img->encoding != sensor_msgs::image_encodings::TYPE_32FC1) {
    ROS_WARN_THROTTLE(5.0,
                      "[MapROS] Received non-depth image '%s' on /map_ros/depth, skip this frame.",
                      img->encoding.c_str());
    return;
  }

  cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img, img->encoding);
  if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
    (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, k_depth_scaling_factor_);
  cv_ptr->image.copyTo(*depth_image_);

  auto t1 = ros::Time::now();

  // generate point cloud, update map
  proessDepthImage();
  map_->inputPointCloud(point_cloud_, proj_points_cnt, camera_pos_, camera_q_);
  if (local_updated_) {
    map_->clearAndInflateLocalMap();
    esdf_need_update_ = true;
    local_updated_ = false;
  }

  auto t2 = ros::Time::now();
  fuse_time_ += (t2 - t1).toSec();
  max_fuse_time_ = max(max_fuse_time_, (t2 - t1).toSec());
  fuse_num_ += 1;
  if (show_occ_time_)
    ROS_WARN("Fusion t: cur: %lf, avg: %lf, max: %lf", (t2 - t1).toSec(), fuse_time_ / fuse_num_,
             max_fuse_time_);
}

void MapROS::cloudPoseCallback(const sensor_msgs::PointCloud2ConstPtr& msg,
                               const geometry_msgs::PoseStampedConstPtr& pose) {
  camera_pos_(0) = pose->pose.position.x;
  camera_pos_(1) = pose->pose.position.y;
  camera_pos_(2) = pose->pose.position.z;
  camera_q_ = Eigen::Quaterniond(pose->pose.orientation.w, pose->pose.orientation.x,
                                 pose->pose.orientation.y, pose->pose.orientation.z);

  // 对点云必须使用传感器自身 frame 的 TF。/map_ros/pose 当前来自船体 odom，
  // 只表示 base_link 位姿，若直接拿它去变换 lidar 点云，会忽略传感器安装偏移。
  // 先按点云时间戳查 TF，避免运动时用“最新 TF”带来的时序错配。
  tf::StampedTransform sensor_tf;
  bool has_sensor_tf = false;
  if (tf_listener_) {
    try {
      tf_listener_->lookupTransform(
          frame_id_, msg->header.frame_id, msg->header.stamp, sensor_tf);
      has_sensor_tf = true;
    } catch (tf::TransformException& ex) {
      try {
        tf_listener_->lookupTransform(frame_id_, msg->header.frame_id, ros::Time(0), sensor_tf);
        has_sensor_tf = true;
      } catch (tf::TransformException&) {
        ROS_WARN_THROTTLE(
            2.0,
            "[MapROS] TF world <- %s unavailable, fall back to /map_ros/pose: %s",
            msg->header.frame_id.c_str(), ex.what());
      }
    }

    if (has_sensor_tf) {
      // lookupTransform(target=frame_id_, source=msg->header.frame_id) returns
      // a transform that maps sensor-frame coordinates into world-frame coordinates.
      camera_pos_(0) = sensor_tf.getOrigin().x();
      camera_pos_(1) = sensor_tf.getOrigin().y();
      camera_pos_(2) = sensor_tf.getOrigin().z();

      // Use tf's own rotation quaternion to avoid potential row/column
      // index convention mismatch when converting Matrix3x3 -> Eigen.
      const tf::Quaternion q_tf = sensor_tf.getRotation();
      camera_q_ = Eigen::Quaterniond(q_tf.w(), q_tf.x(), q_tf.y(), q_tf.z());
    }
  }

  if (std::abs(camera_yaw_offset_) > 1e-9) {
    Eigen::Quaterniond q_yaw(Eigen::AngleAxisd(camera_yaw_offset_, Eigen::Vector3d::UnitZ()));
    camera_q_ = q_yaw * camera_q_;
  }
  if (camera_q_conjugate_) {
    camera_q_ = camera_q_.conjugate();
  }

  // 使用传感器 frame 的实际 TF 将点云从传感器坐标系变换到 world 坐标系。
  pcl::PointCloud<pcl::PointXYZ> cloud_sensor;
  pcl::fromROSMsg(*msg, cloud_sensor);
  int num = cloud_sensor.points.size();

  // 若相机/雷达自身已经在地图外，则本帧直接丢弃，避免无效融合
  if (!map_->isInMap(camera_pos_)) return;

  Eigen::Matrix3d R = camera_q_.toRotationMatrix();
  pcl::PointCloud<pcl::PointXYZ> cloud_world;
  cloud_world.points.reserve(num);

  // Debug: check whether incoming point cloud (after TF) has points
  // within the same z bands used by explored_grid projection.
  const double z_center = explored_grid_z_;
  const double z_slice_min = z_center - explored_grid_z_half_thickness_;
  const double z_slice_max = z_center + explored_grid_z_half_thickness_;
  const double z_obs_min = explored_grid_obs_z_min_;
  const double z_obs_max = explored_grid_obs_z_max_;
  int cnt_in_slice_z = 0;
  int cnt_in_obs_z = 0;
  int cnt_filtered_water = 0;
  int cnt_filtered_near = 0;
  double z_min_seen = 1e18;
  double z_max_seen = -1e18;

  for (int i = 0; i < num; ++i) {
    const auto& pt = cloud_sensor.points[i];
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
    // Drop near-field points in sensor frame to suppress reflection artifacts.
    const double r2_sensor = static_cast<double>(pt.x) * pt.x +
                             static_cast<double>(pt.y) * pt.y +
                             static_cast<double>(pt.z) * pt.z;
    if (cloud_filter_mindist_ > 0.0 && r2_sensor < cloud_filter_mindist_ * cloud_filter_mindist_) {
      cnt_filtered_near++;
      continue;
    }

    Eigen::Vector3d p_w;
    if (has_sensor_tf) {
      tf::Vector3 ps(pt.x, pt.y, pt.z);
      tf::Vector3 pw = sensor_tf * ps;  // sensor -> world
      p_w = Eigen::Vector3d(pw.x(), pw.y(), pw.z());
    } else {
      Eigen::Vector3d p_s(pt.x, pt.y, pt.z);
      p_w = R * p_s + camera_pos_;
    }

    // Hard filter out water-surface noise clutter.
    if (water_filter_enable_ && p_w.z() >= water_filter_z_min_ && p_w.z() <= water_filter_z_max_) {
      cnt_filtered_water++;
      continue;
    }
    pcl::PointXYZ pw;
    pw.x = static_cast<float>(p_w.x());
    pw.y = static_cast<float>(p_w.y());
    pw.z = static_cast<float>(p_w.z());
    cloud_world.points.push_back(pw);

    z_min_seen = std::min(z_min_seen, static_cast<double>(pw.z));
    z_max_seen = std::max(z_max_seen, static_cast<double>(pw.z));
    if (pw.z >= z_slice_min && pw.z <= z_slice_max) cnt_in_slice_z++;
    if (pw.z >= z_obs_min && pw.z <= z_obs_max) cnt_in_obs_z++;
  }

  if (cloud_debug_) {
    const int cnt_after_filter = static_cast<int>(cloud_world.points.size());
    if (cnt_after_filter > 0) {
      ROS_WARN_THROTTLE(cloud_debug_throttle_,
                        "[pointcloud->map] in=%d after_filter=%d "
                        "filtered_water=%d filtered_near=%d "
                        "z[min,max]=[%.3f,%.3f] in_slice_z=%d in_obs_z=%d "
                        "(slice=[%.2f,%.2f] obs=[%.2f,%.2f])",
                        num, cnt_after_filter, cnt_filtered_water, cnt_filtered_near,
                        z_min_seen, z_max_seen,
                        cnt_in_slice_z, cnt_in_obs_z, z_slice_min, z_slice_max,
                        z_obs_min, z_obs_max);
    }
  }

  if (!cloud_world.points.empty()) {
    map_->inputPointCloud(cloud_world, cloud_world.points.size(), camera_pos_, camera_q_);
  }

  if (local_updated_) {
    map_->clearAndInflateLocalMap();
    esdf_need_update_ = true;
    local_updated_ = false;
  }
}

void MapROS::proessDepthImage() {
  proj_points_cnt = 0;

  uint16_t* row_ptr;
  int cols = depth_image_->cols;
  int rows = depth_image_->rows;
  double depth;
  Eigen::Matrix3d camera_r = camera_q_.toRotationMatrix();
  Eigen::Vector3d pt_cur, pt_world;
  const double inv_factor = 1.0 / k_depth_scaling_factor_;

  for (int v = depth_filter_margin_; v < rows - depth_filter_margin_; v += skip_pixel_) {
    row_ptr = depth_image_->ptr<uint16_t>(v) + depth_filter_margin_;
    for (int u = depth_filter_margin_; u < cols - depth_filter_margin_; u += skip_pixel_) {
      depth = (*row_ptr) * inv_factor;
      row_ptr = row_ptr + skip_pixel_;

      // // filter depth
      // if (depth > 0.01)
      //   depth += rand_noise_(eng_);

      // TODO: simplify the logic here
      if (*row_ptr == 0 || depth > depth_filter_maxdist_)
        depth = depth_filter_maxdist_;
      else if (depth < depth_filter_mindist_)
        continue;

      pt_cur(0) = (u - cx_) * depth / fx_;
      pt_cur(1) = (v - cy_) * depth / fy_;
      pt_cur(2) = depth;
      pt_world = camera_r * pt_cur + camera_pos_;
      auto& pt = point_cloud_.points[proj_points_cnt++];
      pt.x = pt_world[0];
      pt.y = pt_world[1];
      pt.z = pt_world[2];
    }
  }

  publishDepth();
}

void MapROS::publishMapAll() {
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;

  const double log_known_thresh = map_->mp_->clamp_min_log_ - 1e-3;  // 判断“已探索”的阈值

  // 使用 box_min_/box_max_ 作为索引范围前，先通过 boundIndex 裁剪到合法体素索引范围内，
  // 避免在缩小地图尺寸、但仍使用较大 box 时越界访问 occupancy_buffer_。
  Eigen::Vector3i min_idx = map_->mp_->box_min_;
  Eigen::Vector3i max_idx = map_->mp_->box_max_ - Eigen::Vector3i::Ones();
  map_->boundIndex(min_idx);
  map_->boundIndex(max_idx);

  for (int x = min_idx(0); x <= max_idx(0); ++x)
    for (int y = min_idx(1); y <= max_idx(1); ++y)
      for (int z = min_idx(2); z <= max_idx(2); ++z) {
        if (map_->md_->occupancy_buffer_[map_->toAddress(x, y, z)] < map_->mp_->min_occupancy_log_)
          continue;

        Eigen::Vector3d pos;
        map_->indexToPos(Eigen::Vector3i(x, y, z), pos);
        if (pos(2) > visualization_truncate_height_ || pos(2) < visualization_truncate_low_) continue;

        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = pos(2);
        cloud.push_back(pt);
      }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  map_all_pub_.publish(cloud_msg);

  // Output time and known volume（体积统计依然使用“已探索”判据）
  double time_now = (ros::Time::now() - map_start_time_).toSec();
  double known_volumn = 0;
  const double voxel_volume = map_->mp_->resolution_ * map_->mp_->resolution_ * map_->mp_->resolution_;

  for (int x = min_idx(0); x <= max_idx(0); ++x)
    for (int y = min_idx(1); y <= max_idx(1); ++y)
      for (int z = min_idx(2); z <= max_idx(2); ++z) {
        if (map_->md_->occupancy_buffer_[map_->toAddress(x, y, z)] > log_known_thresh)
          known_volumn += voxel_volume;
      }

  const double total_volume =
      static_cast<double>(max_idx(0) - min_idx(0) + 1) *
      static_cast<double>(max_idx(1) - min_idx(1) + 1) *
      static_cast<double>(max_idx(2) - min_idx(2) + 1) * voxel_volume;
  const double coverage = (total_volume > 1e-9) ? (known_volumn / total_volume) : 0.0;

  ofstream file(log_filename, ios::app);
  if (file.is_open()) {
    file << time_now << "," << known_volumn << "," << total_volume << "," << coverage
         << std::endl;
  }
}

void MapROS::publishMapLocal() {
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::PointCloud<pcl::PointXYZ> cloud2;
  Eigen::Vector3i min_cut = map_->md_->local_bound_min_;
  Eigen::Vector3i max_cut = map_->md_->local_bound_max_;
  map_->boundIndex(min_cut);
  map_->boundIndex(max_cut);

  // for (int z = min_cut(2); z <= max_cut(2); ++z)
  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = map_->mp_->box_min_(2); z < map_->mp_->box_max_(2); ++z) {
        if (map_->md_->occupancy_buffer_[map_->toAddress(x, y, z)] > map_->mp_->min_occupancy_log_) {
          // Occupied cells
          Eigen::Vector3d pos;
          map_->indexToPos(Eigen::Vector3i(x, y, z), pos);
          if (pos(2) > visualization_truncate_height_) continue;
          if (pos(2) < visualization_truncate_low_) continue;
          // 仅用于可视化：低于 occ_z_min_ 的占据体素不画出来（过滤掉水面噪点）
          if (pos(2) < occ_z_min_) continue;

          pt.x = pos(0);
          pt.y = pos(1);
          pt.z = pos(2);
          cloud.push_back(pt);
        }
        // else if (map_->md_->occupancy_buffer_inflate_[map_->toAddress(x, y, z)] == 1)
        // {
        //   // Inflated occupied cells
        //   Eigen::Vector3d pos;
        //   map_->indexToPos(Eigen::Vector3i(x, y, z), pos);
        //   if (pos(2) > visualization_truncate_height_)
        //     continue;
        //   if (pos(2) < visualization_truncate_low_)
        //     continue;

        //   pt.x = pos(0);
        //   pt.y = pos(1);
        //   pt.z = pos(2);
        //   cloud2.push_back(pt);
        // }
      }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  cloud2.width = cloud2.points.size();
  cloud2.height = 1;
  cloud2.is_dense = true;
  cloud2.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;

  pcl::toROSMsg(cloud, cloud_msg);
  map_local_pub_.publish(cloud_msg);
  pcl::toROSMsg(cloud2, cloud_msg);
  map_local_inflate_pub_.publish(cloud_msg);
}

void MapROS::publishUnknown() {
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  Eigen::Vector3i min_cut = map_->md_->local_bound_min_;
  Eigen::Vector3i max_cut = map_->md_->local_bound_max_;
  map_->boundIndex(max_cut);
  map_->boundIndex(min_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z) {
        if (map_->md_->occupancy_buffer_[map_->toAddress(x, y, z)] < map_->mp_->clamp_min_log_ - 1e-3) {
          Eigen::Vector3d pos;
          map_->indexToPos(Eigen::Vector3i(x, y, z), pos);
          if (pos(2) > visualization_truncate_height_) continue;
          if (pos(2) < visualization_truncate_low_) continue;
          pt.x = pos(0);
          pt.y = pos(1);
          pt.z = pos(2);
          cloud.push_back(pt);
        }
      }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  unknown_pub_.publish(cloud_msg);
}

void MapROS::publishDepth() {
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  for (int i = 0; i < proj_points_cnt; ++i) {
    cloud.push_back(point_cloud_.points[i]);
  }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  depth_pub_.publish(cloud_msg);
}

void MapROS::publishUpdateRange() {
  Eigen::Vector3d esdf_min_pos, esdf_max_pos, cube_pos, cube_scale;
  visualization_msgs::Marker mk;
  map_->indexToPos(map_->md_->local_bound_min_, esdf_min_pos);
  map_->indexToPos(map_->md_->local_bound_max_, esdf_max_pos);

  cube_pos = 0.5 * (esdf_min_pos + esdf_max_pos);
  cube_scale = esdf_max_pos - esdf_min_pos;
  mk.header.frame_id = frame_id_;
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::CUBE;
  mk.action = visualization_msgs::Marker::ADD;
  mk.id = 0;
  mk.pose.position.x = cube_pos(0);
  mk.pose.position.y = cube_pos(1);
  mk.pose.position.z = cube_pos(2);
  mk.scale.x = cube_scale(0);
  mk.scale.y = cube_scale(1);
  mk.scale.z = cube_scale(2);
  mk.color.a = 0.3;
  mk.color.r = 1.0;
  mk.color.g = 0.0;
  mk.color.b = 0.0;
  mk.pose.orientation.w = 1.0;
  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;

  update_range_pub_.publish(mk);
}

void MapROS::publishESDF() {
  double dist;
  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::PointXYZI pt;

  const double min_dist = 0.0;
  const double max_dist = 3.0;

  Eigen::Vector3i min_cut = map_->md_->local_bound_min_ - Eigen::Vector3i(map_->mp_->local_map_margin_,
                                                                          map_->mp_->local_map_margin_,
                                                                          map_->mp_->local_map_margin_);
  Eigen::Vector3i max_cut = map_->md_->local_bound_max_ + Eigen::Vector3i(map_->mp_->local_map_margin_,
                                                                          map_->mp_->local_map_margin_,
                                                                          map_->mp_->local_map_margin_);
  map_->boundIndex(min_cut);
  map_->boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y) {
      Eigen::Vector3d pos;
      map_->indexToPos(Eigen::Vector3i(x, y, 1), pos);
      pos(2) = esdf_slice_height_;
      dist = map_->getDistance(pos);
      dist = min(dist, max_dist);
      dist = max(dist, min_dist);
      pt.x = pos(0);
      pt.y = pos(1);
      pt.z = -0.2;
      pt.intensity = (dist - min_dist) / (max_dist - min_dist);
      cloud.push_back(pt);
    }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);

  esdf_pub_.publish(cloud_msg);

  // ROS_INFO("pub esdf");
}
}