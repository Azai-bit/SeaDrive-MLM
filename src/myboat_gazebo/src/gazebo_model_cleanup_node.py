#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import rospy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import DeleteModel


def _model_exists(name):
    try:
        states = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=0.5)
    except rospy.ROSException:
        return None
    return name in states.name


def main():
    rospy.init_node("gazebo_model_cleanup_node")
    models_param = rospy.get_param("~models", "")
    timeout_s = float(rospy.get_param("~timeout_s", 20.0))
    models = [m.strip() for m in models_param.split(",") if m.strip()]
    if not models:
        rospy.loginfo("[gazebo_model_cleanup] no models configured")
        return

    rospy.wait_for_service("/gazebo/delete_model")
    delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
    deadline = time.monotonic() + max(0.0, timeout_s)

    for name in models:
        while not rospy.is_shutdown():
            exists = _model_exists(name)
            if exists is False:
                rospy.loginfo("[gazebo_model_cleanup] model %s not present", name)
                break

            try:
                resp = delete_model(name)
                if resp.success:
                    rospy.loginfo("[gazebo_model_cleanup] deleted stale model %s", name)
                elif exists is None:
                    rospy.logwarn("[gazebo_model_cleanup] delete %s: %s", name, resp.status_message)
            except rospy.ServiceException as exc:
                rospy.logwarn("[gazebo_model_cleanup] delete %s failed: %s", name, exc)

            if time.monotonic() >= deadline:
                rospy.logwarn("[gazebo_model_cleanup] timeout while cleaning %s", name)
                break
            rospy.sleep(0.2)


if __name__ == "__main__":
    main()
