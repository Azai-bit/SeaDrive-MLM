#!/usr/bin/env python

import rospy
import tf
from nav_msgs.msg import Odometry


class MyboatOdomTF(object):
    def __init__(self):
        odom_topic = rospy.get_param("~odom_topic", "/myboat/odom")
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.base_frame = rospy.get_param("~base_frame", "myboat/base_link")

        self.br = tf.TransformBroadcaster()
        self.last_stamp = None
        self.sub = rospy.Subscriber(odom_topic, Odometry, self.odom_cb, queue_size=10)

        rospy.loginfo("[myboat_odom_tf] Broadcasting TF %s -> %s from %s",
                      self.world_frame, self.base_frame, odom_topic)

    def odom_cb(self, msg):
        # 避免在同一时间戳上重复发布 TF，触发 TF_REPEATED_DATA 警告
        if self.last_stamp == msg.header.stamp:
            return
        self.last_stamp = msg.header.stamp

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # world_frame -> base_frame，直接使用 odom 中的位姿
        self.br.sendTransform(
            (p.x, p.y, p.z),
            (q.x, q.y, q.z, q.w),
            msg.header.stamp,
            self.base_frame,
            self.world_frame
        )


def main():
    rospy.init_node("myboat_odom_tf")
    MyboatOdomTF()
    rospy.spin()


if __name__ == "__main__":
    main()

