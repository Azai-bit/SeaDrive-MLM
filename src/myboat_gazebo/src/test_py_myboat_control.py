#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist

class myboat_control:
    def __init__(self, topic_name='/cmd_vel', queue_size=100):

        self.pub = rospy.Publisher(topic_name, Twist, queue_size=queue_size)

    def control(self, linear_x, angular_z):

        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = angular_z
        self.pub.publish(twist)


if __name__ == '__main__':
    # 初始化 ROS 节点
    rospy.init_node('myboat_simple_controller')

    # 实例化控制类
    boat = myboat_control()

    # 示例：以 0.5 m/s 前进，同时以 0.2 系数转向
    # 转向值建议: 0.1至0.3
    for i in range(50):

        # 调用控制函数,传入线速度和角速度即可
        boat.control(1.0, -0.2)
        rospy.sleep(0.5)

    # 停止
    boat.control(0.0, 0.0)