#include <ros/ros.h>
#include <geometry_msgs/Twist.h>

class MyBoatControl {
public:
    MyBoatControl(const std::string& topic_name = "/cmd_vel", int queue_size = 100)
        : nh_() {
        pub_ = nh_.advertise<geometry_msgs::Twist>(topic_name, queue_size);
    }

    void control(double linear_x, double angular_z) {
        geometry_msgs::Twist msg;
        msg.linear.x = linear_x;
        msg.linear.y = 0.0;
        msg.linear.z = 0.0;
        msg.angular.x = 0.0;
        msg.angular.y = 0.0;
        msg.angular.z = angular_z;
        pub_.publish(msg);
    }

private:
    ros::NodeHandle nh_;
    ros::Publisher pub_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "myboat_simple_controller");
    MyBoatControl boat;

    // 发布控制命令 50 次，每次间隔 0.5 秒
    for (int i = 0; i < 50; ++i) 
    {
        // # 示例：以 0.5 m/s 前进，同时以 0.2 系数转向
        // # 转向值建议: 0.1至0.3        
        boat.control(0.5, -0.2);
        ros::Duration(0.5).sleep();
    }

    // 停止
    boat.control(0.0, 0.0);

    return 0;
}