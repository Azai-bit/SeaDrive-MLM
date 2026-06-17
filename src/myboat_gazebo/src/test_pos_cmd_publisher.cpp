#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <tf/transform_datatypes.h>

class TestPosCmdPublisher {
public:
  TestPosCmdPublisher() : has_odom_(false) {
    ros::NodeHandle pnh("~");

    pnh.param<std::string>("odom_topic", odom_topic_, std::string("/myboat/odom"));
    pnh.param<std::string>("pos_cmd_topic", pos_cmd_topic_, std::string("/planning/pos_cmd"));
    pnh.param("forward_distance", forward_distance_, 5.0);
    pnh.param("publish_rate", publish_rate_, 5.0);

    ros::NodeHandle nh;
    odom_sub_ = nh.subscribe(odom_topic_, 10, &TestPosCmdPublisher::odomCallback, this);
    pos_cmd_pub_ = nh.advertise<quadrotor_msgs::PositionCommand>(pos_cmd_topic_, 10);

    timer_ = nh.createTimer(ros::Duration(1.0 / publish_rate_),
                            &TestPosCmdPublisher::timerCallback, this);

    ROS_WARN("[TestPosCmdPublisher] Debug node started.");
    ROS_WARN("  odom_topic: %s", odom_topic_.c_str());
    ROS_WARN("  pos_cmd_topic: %s", pos_cmd_topic_.c_str());
    ROS_WARN("  forward_distance: %.2f m, rate: %.2f Hz", forward_distance_, publish_rate_);
  }

private:
  void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    last_odom_ = *msg;
    has_odom_ = true;
  }

  void timerCallback(const ros::TimerEvent&) {
    if (!has_odom_) {
      ROS_WARN_THROTTLE(2.0, "[TestPosCmdPublisher] Waiting for odom on %s...",
                        odom_topic_.c_str());
      return;
    }

    // 从当前姿态提取 yaw
    double roll, pitch, yaw;
    tf::Quaternion q(last_odom_.pose.pose.orientation.x,
                     last_odom_.pose.pose.orientation.y,
                     last_odom_.pose.pose.orientation.z,
                     last_odom_.pose.pose.orientation.w);
    tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

    // 目标点：沿当前朝向 forward_distance_ 米
    quadrotor_msgs::PositionCommand cmd;
    cmd.header.stamp = ros::Time::now();
    cmd.header.frame_id = "world";

    cmd.position.x =
        last_odom_.pose.pose.position.x + forward_distance_ * std::cos(yaw);
    cmd.position.y =
        last_odom_.pose.pose.position.y + forward_distance_ * std::sin(yaw);
    cmd.position.z = last_odom_.pose.pose.position.z;  // 保持高度不变

    cmd.yaw = yaw;  // 朝向目标前进

    // 其余字段置零（当前消息类型中没有 jerk 字段）
    cmd.velocity.x = cmd.velocity.y = cmd.velocity.z = 0.0;
    cmd.acceleration.x = cmd.acceleration.y = cmd.acceleration.z = 0.0;

    pos_cmd_pub_.publish(cmd);

    ROS_INFO_THROTTLE(1.0,
                      "[TestPosCmdPublisher] Publish debug PositionCommand to (%.2f, %.2f, %.2f)",
                      cmd.position.x, cmd.position.y, cmd.position.z);
  }

  ros::Subscriber odom_sub_;
  ros::Publisher pos_cmd_pub_;
  ros::Timer timer_;

  std::string odom_topic_;
  std::string pos_cmd_topic_;
  double forward_distance_;
  double publish_rate_;

  nav_msgs::Odometry last_odom_;
  bool has_odom_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "test_pos_cmd_publisher");

  TestPosCmdPublisher tester;
  ros::spin();

  return 0;
}

