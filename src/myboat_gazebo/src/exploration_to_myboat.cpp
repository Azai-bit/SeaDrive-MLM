#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <tf/transform_datatypes.h>
#include <cmath>
#include <algorithm>

class ExplorationToMyBoat {
private:
    ros::NodeHandle nh_;   // 全局命名空间句柄（用于话题）
    ros::NodeHandle pnh_;  // 私有命名空间句柄（用于 ~param）
    ros::Subscriber pos_cmd_sub_;         // 订阅规划位置指令（PositionCommand）
    ros::Subscriber odom_sub_;            // 订阅里程计
    ros::Publisher cmd_vel_pub_;          // 发布速度命令
    
    double max_linear_vel_;               // 最大线速度
    double max_angular_vel_;              // 最大角速度
    double kp_linear_;                    // 线速度P控制器增益
    double kp_angular_;                   // 角速度P控制器增益
    double distance_threshold_;           // 距离阈值（到达/停止判定）
    double yaw_align_threshold_;          // 允许在前进前的最大偏航误差
    bool   use_planned_yaw_;              // 是否优先使用规划给出的 yaw
    double last_distance_;                // 上一次的距离误差，用于检测是否在收敛
    double ff_vel_gain_;                  // 规划速度前馈增益
    double cmd_linear_acc_limit_;         // 线速度指令加速度限制 (m/s^2)
    double cmd_angular_acc_limit_;        // 角速度指令加速度限制 (rad/s^2)
    double min_moving_speed_;             // 大转角时仍保持的小前进速度

    double last_cmd_linear_;
    double last_cmd_angular_;
    ros::Time last_cmd_time_;

    // For yaw-rate estimation and cascaded yaw control
    double last_yaw_;
    ros::Time last_yaw_time_;
    bool have_last_yaw_;

    double omega_meas_filt_alpha_;
    double last_omega_meas_;
    double last_omega_err_;

    double kd_yaw_;      // Outer-loop D term (yaw error rate approx)
    double kp_omega_;    // Inner-loop P term (omega error)
    double kd_omega_;   // Inner-loop D term (omega error derivative)

    // When yaw is about to overshoot, brake it by reducing linear speed so
    // that twist2thrust mapping can generate reverse thrust on one side.
    bool brake_enable_;
    double brake_linear_scale_;
    double brake_omega_thresh_;
    
    geometry_msgs::Pose current_pose_;
    bool has_odom_;

public:
    ExplorationToMyBoat()
                : nh_(), pnh_("~"), has_odom_(false), last_distance_(-1.0),
                    last_cmd_linear_(0.0), last_cmd_angular_(0.0) {
        // 获取参数（优先从私有命名空间 ~ 读取，对应 launch 文件中的 <param>）
        pnh_.param("max_linear_vel", max_linear_vel_, 0.8);
        pnh_.param("max_angular_vel", max_angular_vel_, 0.5);
        pnh_.param("kp_linear", kp_linear_, 1.2);
        pnh_.param("kp_angular", kp_angular_, 1.5);
        pnh_.param("distance_threshold", distance_threshold_, 0.1);
        pnh_.param("yaw_align_threshold", yaw_align_threshold_, 0.5);  // 约 30 度
        pnh_.param("use_planned_yaw", use_planned_yaw_, true);
        pnh_.param("ff_vel_gain", ff_vel_gain_, 0.7);
        pnh_.param("cmd_linear_acc_limit", cmd_linear_acc_limit_, 0.6);
        pnh_.param("cmd_angular_acc_limit", cmd_angular_acc_limit_, 1.2);
        pnh_.param("min_moving_speed", min_moving_speed_, 0.08);

        // Cascaded yaw PD (yaw -> omega -> cmd angular rate)
        pnh_.param("kd_yaw", kd_yaw_, 0.2);
        pnh_.param("kp_omega", kp_omega_, 1.0);
        pnh_.param("kd_omega", kd_omega_, 0.05);
        pnh_.param("omega_meas_filt_alpha", omega_meas_filt_alpha_, 0.7);

        // Brake configuration: reduce linear speed when yaw-rate overshoot is detected.
        pnh_.param("brake_enable", brake_enable_, false);
        pnh_.param("brake_linear_scale", brake_linear_scale_, 0.2);
        pnh_.param("brake_omega_thresh", brake_omega_thresh_, 0.4);

        last_yaw_ = 0.0;
        last_yaw_time_ = ros::Time::now();
        have_last_yaw_ = false;
        last_omega_meas_ = 0.0;
        last_omega_err_ = 0.0;
        
        // 订阅规划位置指令（traj_server 输出的 quadrotor_msgs/PositionCommand）
        pos_cmd_sub_ = nh_.subscribe("/planning/pos_cmd", 10, 
                                     &ExplorationToMyBoat::posCmdCallback, this);
        
        // 订阅里程计用于获取当前位置
        odom_sub_ = nh_.subscribe("/myboat/odom", 10,
                                  &ExplorationToMyBoat::odomCallback, this);
        
        // 发布速度命令到 myboat
        cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
        
        ROS_INFO("[ExplorationToMyBoat] Initialized successfully");
        ROS_INFO("  max_linear_vel: %.2f m/s", max_linear_vel_);
        ROS_INFO("  max_angular_vel: %.2f rad/s", max_angular_vel_);
        ROS_INFO("  kp_linear: %.2f", kp_linear_);
        ROS_INFO("  kp_angular: %.2f", kp_angular_);
        ROS_INFO("  distance_threshold: %.3f m", distance_threshold_);
        ROS_INFO("  yaw_align_threshold: %.3f rad", yaw_align_threshold_);
        ROS_INFO("  use_planned_yaw: %s", use_planned_yaw_ ? "true" : "false");
        ROS_INFO("  ff_vel_gain: %.2f", ff_vel_gain_);
        ROS_INFO("  cmd_linear_acc_limit: %.2f m/s^2", cmd_linear_acc_limit_);
        ROS_INFO("  cmd_angular_acc_limit: %.2f rad/s^2", cmd_angular_acc_limit_);
        ROS_INFO("  min_moving_speed: %.2f m/s", min_moving_speed_);
        ROS_INFO("  kd_yaw: %.2f", kd_yaw_);
        ROS_INFO("  kp_omega: %.2f", kp_omega_);
        ROS_INFO("  kd_omega: %.2f", kd_omega_);
        ROS_INFO("  omega_meas_filt_alpha: %.2f", omega_meas_filt_alpha_);
        ROS_INFO("  brake_enable: %s", brake_enable_ ? "true" : "false");
        ROS_INFO("  brake_linear_scale: %.2f", brake_linear_scale_);
        ROS_INFO("  brake_omega_thresh: %.2f rad/s", brake_omega_thresh_);

        last_cmd_time_ = ros::Time::now();
    }
    
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        current_pose_ = msg->pose.pose;
        has_odom_ = true;
    }
    
    void posCmdCallback(const quadrotor_msgs::PositionCommand::ConstPtr& msg) {
        if (!has_odom_) {
            ROS_WARN_ONCE("[ExplorationToMyBoat] Waiting for odometry data...");
            return;
        }
        
        // 计算相对于当前位置的目标偏差（使用 PositionCommand 中的 position 字段）
        double dx = msg->position.x - current_pose_.position.x;
        double dy = msg->position.y - current_pose_.position.y;
        double distance = std::sqrt(dx * dx + dy * dy);
        
        // 目标偏航角：优先使用规划器给出的 yaw，其次使用指向目标点的方向
        double current_yaw = getYawFromQuaternion(current_pose_.orientation);
        double target_yaw;
        if (use_planned_yaw_) {
            target_yaw = msg->yaw;
        } else {
            target_yaw = std::atan2(dy, dx);
        }
        
        // 计算偏航角误差
        double yaw_error = target_yaw - current_yaw;
        
        // 正规化角度到 [-pi, pi]
        while (yaw_error > M_PI) yaw_error -= 2 * M_PI;
        while (yaw_error < -M_PI) yaw_error += 2 * M_PI;

        // 生成速度命令
        geometry_msgs::Twist cmd;

        // === Cascaded PD yaw control for differential-drive boat ===
        // Measure yaw-rate by differentiating yaw (odom has no angular.z).
        const ros::Time now = ros::Time::now();
        double dt_yaw = (now - last_yaw_time_).toSec();
        if (dt_yaw <= 1e-4) dt_yaw = 0.0;

        // Estimate omega_meas (rad/s) from wrapped yaw delta
        double omega_meas_raw = last_omega_meas_;
        if (!have_last_yaw_ || dt_yaw <= 0.0) {
          omega_meas_raw = 0.0;
        } else {
          double dyaw = current_yaw - last_yaw_;
          while (dyaw > M_PI) dyaw -= 2 * M_PI;
          while (dyaw < -M_PI) dyaw += 2 * M_PI;
          omega_meas_raw = dyaw / dt_yaw;
        }
        // Low-pass filter omega_meas to reduce noise in inner D term
        double omega_meas = omega_meas_filt_alpha_ * last_omega_meas_ + (1.0 - omega_meas_filt_alpha_) * omega_meas_raw;

        // Outer loop: yaw_error -> desired yaw rate
        const double yaw_dot_ref = use_planned_yaw_ ? msg->yaw_dot : 0.0;
        const double omega_des = yaw_dot_ref + kp_angular_ * yaw_error + kd_yaw_ * (yaw_dot_ref - omega_meas);

        // Inner loop: track omega_des with cascaded PD on omega error
        const double omega_err = omega_des - omega_meas;
        double omega_err_dot = 0.0;
        if (dt_yaw > 0.0) {
          omega_err_dot = (omega_err - last_omega_err_) / dt_yaw;
        }
        double omega_cmd_raw = omega_des + kp_omega_ * omega_err + kd_omega_ * omega_err_dot;
        double omega_cmd = std::min(max_angular_vel_, std::max(-max_angular_vel_, omega_cmd_raw));

        cmd.angular.z = omega_cmd;

        // Brake trigger:
        // when we need to reduce omega (omega_err < 0) but the measured omega is already large,
        // scale down linear speed so that twist2thrust (left=lin-ang, right=lin+ang) can produce
        // reverse thrust on one side, generating a stronger yaw braking moment.
        const bool brake_now =
            brake_enable_ && (omega_err < 0.0) && (std::fabs(omega_meas) > brake_omega_thresh_);

        // Save yaw-rate history
        last_yaw_ = current_yaw;
        last_yaw_time_ = now;
        have_last_yaw_ = true;
        last_omega_meas_ = omega_meas;
        last_omega_err_ = omega_err;

        // 船体非完整约束：只允许沿船头方向前进，绝不输出等效横移
        // 目标方向（世界系）与船头方向（世界系）的点积 = 前向分量，仅当前向分量 > 0 时才给前进速度
        double dir_x = distance > 1e-6 ? dx / distance : 0.0;
        double dir_y = distance > 1e-6 ? dy / distance : 0.0;
        double forward_x = std::cos(current_yaw);
        double forward_y = std::sin(current_yaw);
        double forward_component = dir_x * forward_x + dir_y * forward_y;

        double forward_weight = std::max(0.0, std::cos(yaw_error));
        double ff_forward = 0.0;
        if (distance > 1e-3) {
            ff_forward = (msg->velocity.x * dx + msg->velocity.y * dy) / distance;
            ff_forward = std::max(0.0, ff_forward);
        }

        if (distance > distance_threshold_ && forward_component > 0.0) {
            // P + 速度前馈；转角较大时自动降速，但不完全停住，避免“停-转-冲”抖动
            double target_linear = kp_linear_ * distance * forward_component;
            target_linear += ff_vel_gain_ * ff_forward * forward_component;

            double heading_scale = std::min(1.0, std::max(0.15, forward_weight));
            target_linear *= heading_scale;

            if (std::fabs(yaw_error) > yaw_align_threshold_) {
                target_linear = std::max(min_moving_speed_, target_linear * 0.5);
            }
            cmd.linear.x = std::min(max_linear_vel_, std::max(0.0, target_linear));
        } else {
            cmd.linear.x = 0.0;
        }

        if (brake_now) {
            cmd.linear.x = std::max(0.0, cmd.linear.x * brake_linear_scale_);
        }

        // If we've reached the commanded position, don't keep rotating in place.
        // This is important for FINISH/stop mode to ensure the boat actually stops moving.
        if (distance <= distance_threshold_) {
            cmd.angular.z = 0.0;
        }

        cmd.linear.y = 0.0;
        cmd.linear.z = 0.0;
        cmd.angular.x = 0.0;
        cmd.angular.y = 0.0;
        
        ROS_DEBUG("[ExplorationToMyBoat] Distance: %.3f, Yaw Error: %.3f, Cmd: [%.3f, %.3f]", 
                 distance, yaw_error, cmd.linear.x, cmd.angular.z);

        // 对输出指令做斜率限制，减小推进器饱和引起的“满跟踪”滞后
        ros::Time cmd_now = ros::Time::now();
        double dt = (cmd_now - last_cmd_time_).toSec();
        if (dt <= 1e-3) dt = 0.02;

        double max_dv = cmd_linear_acc_limit_ * dt;
        double max_dw = cmd_angular_acc_limit_ * dt;

        double dv = cmd.linear.x - last_cmd_linear_;
        dv = std::max(-max_dv, std::min(max_dv, dv));
        cmd.linear.x = last_cmd_linear_ + dv;

        double dw = cmd.angular.z - last_cmd_angular_;
        dw = std::max(-max_dw, std::min(max_dw, dw));
        cmd.angular.z = last_cmd_angular_ + dw;

        last_cmd_linear_ = cmd.linear.x;
        last_cmd_angular_ = cmd.angular.z;
        last_cmd_time_ = cmd_now;

        cmd_vel_pub_.publish(cmd);

        // 记录本次距离误差，用于下次判断是否在收敛
        last_distance_ = distance;
    }
    
private:
    double getYawFromQuaternion(const geometry_msgs::Quaternion& q) {
        tf::Quaternion quat(q.x, q.y, q.z, q.w);
        double roll, pitch, yaw;
        tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
        return yaw;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "exploration_to_myboat");
    ros::NodeHandle nh;
    
    ExplorationToMyBoat adapter;
    
    ROS_INFO("[ExplorationToMyBoat] Node started, waiting for commands...");
    ros::spin();
    
    return 0;
}
