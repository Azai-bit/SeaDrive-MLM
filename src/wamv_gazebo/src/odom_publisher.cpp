/*
 * Simple Gazebo odometry publisher for Myboat / WAM-V style vehicles.
 *
 * Subscribes to /gazebo/model_states, extracts the pose & twist of a given
 * model, and republishes them as a standard nav_msgs/Odometry message.
 *
 * This lets higher‑level planners (e.g. exploration_manager) consume odom
 * from a Gazebo world without depending on a custom dynamics plugin.
 */

#include <algorithm>
#include <string>

#include <gazebo_msgs/ModelStates.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>

class OdomPublisher
{
public:
  OdomPublisher()
    : nh_("~")
    , model_index_(-1)
  {
    nh_.param<std::string>("model_name", model_name_, std::string("myboat"));
    nh_.param<std::string>("odom_topic", odom_topic_, std::string("/myboat/odom"));
    nh_.param<std::string>("world_frame", world_frame_, std::string("world"));
    nh_.param<std::string>("base_frame", base_frame_, std::string("myboat/base_link"));

    odom_pub_ = nh_.advertise<nav_msgs::Odometry>(odom_topic_, 10);
    states_sub_ = nh_.subscribe("/gazebo/model_states", 1,
                                &OdomPublisher::statesCallback, this);

    ROS_INFO_STREAM("[odom_publisher] model_name=" << model_name_
                    << ", odom_topic=" << odom_topic_
                    << ", world_frame=" << world_frame_
                    << ", base_frame=" << base_frame_);
  }

private:
  void statesCallback(const gazebo_msgs::ModelStates::ConstPtr &msg)
  {
    if (msg->name.empty())
    {
      ROS_WARN_THROTTLE(5.0, "[odom_publisher] /gazebo/model_states has no models yet.");
      return;
    }

    // Resolve model index once, then reuse.
    if (model_index_ < 0)
    {
      std::vector<std::string>::const_iterator it =
          std::find(msg->name.begin(), msg->name.end(), model_name_);

      if (it == msg->name.end())
      {
        ROS_WARN_THROTTLE(5.0,
                          "[odom_publisher] Model '%s' not found in /gazebo/model_states.",
                          model_name_.c_str());
        return;
      }

      model_index_ = static_cast<int>(std::distance(msg->name.begin(), it));

      ROS_INFO("[odom_publisher] Found model '%s' at index %d.",
               model_name_.c_str(), model_index_);
    }

    if (model_index_ < 0 ||
        model_index_ >= static_cast<int>(msg->pose.size()) ||
        model_index_ >= static_cast<int>(msg->twist.size()))
    {
      ROS_WARN_THROTTLE(5.0,
                        "[odom_publisher] Invalid model index %d for model '%s'.",
                        model_index_, model_name_.c_str());
      return;
    }

    const geometry_msgs::Pose &pose = msg->pose[model_index_];
    const geometry_msgs::Twist &twist = msg->twist[model_index_];

    nav_msgs::Odometry odom;
    // gazebo_msgs/ModelStates 在一些 ROS 版本中没有 header，直接用当前时间
    odom.header.stamp = ros::Time::now();
    odom.header.frame_id = world_frame_;
    odom.child_frame_id = base_frame_;

    odom.pose.pose = pose;
    odom.twist.twist = twist;

    odom_pub_.publish(odom);
  }

  ros::NodeHandle nh_;

  std::string model_name_;
  std::string odom_topic_;
  std::string world_frame_;
  std::string base_frame_;

  int model_index_;

  ros::Publisher odom_pub_;
  ros::Subscriber states_sub_;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "odom_publisher");

  OdomPublisher node;
  ros::spin();

  return 0;
}

