#!/usr/bin/env python3

import sys
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32


class TwistToThrustNode:
    def __init__(self, linear_scaling=1.0, angular_scaling=1.0):
        self.linear_scaling = linear_scaling
        self.angular_scaling = angular_scaling

        self.left_pub = rospy.Publisher("left_cmd", Float32, queue_size=10)
        self.right_pub = rospy.Publisher("right_cmd", Float32, queue_size=10)

        self.left_msg = Float32()
        self.right_msg = Float32()

        # 订阅全局 /cmd_vel，便于 exploration_to_myboat 和 teleop 统一控制入口
        rospy.Subscriber("/cmd_vel", Twist, self.callback)

        rospy.loginfo(
            "TwistToThrustNode initialized with linear_scaling=%.3f, angular_scaling=%.3f",
            linear_scaling, angular_scaling
        )

    def callback(self, data):
        linear = data.linear.x * self.linear_scaling
        angular = data.angular.z * self.angular_scaling

        left_thrust = linear - angular
        right_thrust = linear + angular

        # 限制推进器最大推力，避免指令过于激进导致船体大幅摆动
        left_thrust = max(-3.0, min(3.0, left_thrust))
        right_thrust = max(-3.0, min(3.0, right_thrust))

        rospy.logdebug("RX Twist: linear.x=%.3f, angular.z=%.3f", data.linear.x, data.angular.z)
        rospy.logdebug("TX Thrust: left=%.3f, right=%.3f", left_thrust, right_thrust)

        self.left_msg.data = left_thrust
        self.right_msg.data = right_thrust
        self.left_pub.publish(self.left_msg)
        self.right_pub.publish(self.right_msg)


if __name__ == '__main__':
    rospy.init_node('twist2thrust', anonymous=True)

    linear_scaling = rospy.get_param('~linear_scaling', 1.0)
    angular_scaling = rospy.get_param('~angular_scaling', 1.0)

    node = TwistToThrustNode(linear_scaling=linear_scaling, angular_scaling=angular_scaling)

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass