#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry


class MyboatOdomRelay(object):
    def __init__(self):
        in_topic = rospy.get_param("~in_topic", "/myboat/odom")
        out_topic = rospy.get_param("~out_topic", "/odom_world")

        self.pub = rospy.Publisher(out_topic, Odometry, queue_size=10)
        self.sub = rospy.Subscriber(in_topic, Odometry, self.cb, queue_size=10)

        rospy.loginfo("[myboat_odom_relay] %s -> %s", in_topic, out_topic)

    def cb(self, msg):
        # 这里不做任何修改，直接转发
        self.pub.publish(msg)


def main():
    rospy.init_node("myboat_odom_relay")
    MyboatOdomRelay()
    rospy.spin()


if __name__ == "__main__":
    main()

