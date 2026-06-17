#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time

import rospy
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Empty


class CollisionTargetCruiseNode:
    """Force target boat cruise on fixed straight lane via Gazebo model state."""

    def __init__(self):
        rospy.init_node("collision_target_cruise_node")
        self.model_name = rospy.get_param("~model_name", "target_boat")
        self.speed = float(rospy.get_param("~speed", 1.0))
        self.publish_rate = float(rospy.get_param("~publish_rate", 20.0))
        self.init_x = float(rospy.get_param("~init_x", -440.0))
        self.init_y = float(rospy.get_param("~init_y", 260.0))
        self.model_z = float(rospy.get_param("~model_z", 0.65))
        self.init_yaw = float(rospy.get_param("~init_yaw", math.pi))
        self.turn_rate = float(rospy.get_param("~turn_rate", 0.0))
        self.wait_for_trigger = bool(rospy.get_param("~wait_for_trigger", True))
        self.trigger_topic = rospy.get_param("~trigger_topic", "/traj_start_trigger")
        self.model_wait_timeout = float(rospy.get_param("~model_wait_timeout", 45.0))
        self.started = not self.wait_for_trigger

        self._x = self.init_x
        self._y = self.init_y
        self._yaw = self.init_yaw
        self._last_t = None
        self._model_ready = False

        if self.wait_for_trigger:
            rospy.Subscriber(self.trigger_topic, Empty, self._trigger_cb, queue_size=1)

        rospy.wait_for_service("/gazebo/set_model_state")
        self._set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self._model_ready = self._wait_for_model()
        rospy.loginfo(
            "[collision_target_cruise] model=%s speed=%.2f init=(%.2f, %.2f, z=%.2f, yaw=%.2f) turn_rate=%.4f wait_for_trigger=%s trigger_topic=%s",
            self.model_name,
            self.speed,
            self.init_x,
            self.init_y,
            self.model_z,
            self.init_yaw,
            self.turn_rate,
            str(self.wait_for_trigger),
            self.trigger_topic,
        )

    def _trigger_cb(self, _msg):
        if not self.started:
            self.started = True
            self._last_t = None
            rospy.loginfo("[collision_target_cruise] triggered by %s, start moving", self.trigger_topic)

    def _wait_for_model(self):
        started_wall = time.monotonic()
        while not rospy.is_shutdown():
            try:
                states = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=1.0)
            except rospy.ROSException:
                states = None

            if states is not None and self.model_name in states.name:
                rospy.loginfo("[collision_target_cruise] model %s is available", self.model_name)
                return True

            elapsed = time.monotonic() - started_wall
            if self.model_wait_timeout > 0.0 and elapsed >= self.model_wait_timeout:
                rospy.logwarn(
                    "[collision_target_cruise] timed out waiting for model %s after %.1fs",
                    self.model_name,
                    elapsed,
                )
                return False

            rospy.logwarn_throttle(
                5.0,
                "[collision_target_cruise] waiting for model %s in /gazebo/model_states",
                self.model_name,
            )
        return False

    @staticmethod
    def _quat_from_yaw(yaw):
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw * 0.5)
        q.w = math.cos(yaw * 0.5)
        return q

    def run(self):
        rate = rospy.Rate(max(1.0, self.publish_rate))
        while not rospy.is_shutdown():
            if not self._model_ready:
                self._model_ready = self._wait_for_model()
                if not self._model_ready:
                    rate.sleep()
                    continue

            now = rospy.Time.now()
            if self._last_t is None:
                dt = 0.0
            else:
                dt = (now - self._last_t).to_sec()
                if dt < 0.0:
                    dt = 0.0
                # With /use_sim_time, the first valid /clock sample may jump
                # from zero to the current Gazebo time. Do not integrate that
                # startup jump into the target vessel's position.
                if dt > 1.0:
                    rospy.logwarn_throttle(
                        2.0,
                        "[collision_target_cruise] large dt %.3fs ignored for model=%s",
                        dt,
                        self.model_name,
                    )
                    dt = 0.0
            self._last_t = now

            if self.started:
                self._yaw += self.turn_rate * dt
                vx = self.speed * math.cos(self._yaw)
                vy = self.speed * math.sin(self._yaw)
                self._x += vx * dt
                self._y += vy * dt
            else:
                vx = 0.0
                vy = 0.0

            state = ModelState()
            state.model_name = self.model_name
            state.pose.position.x = self._x
            state.pose.position.y = self._y
            state.pose.position.z = self.model_z
            state.pose.orientation = self._quat_from_yaw(self._yaw)
            state.twist.linear.x = vx
            state.twist.linear.y = vy
            state.twist.linear.z = 0.0
            state.twist.angular.x = 0.0
            state.twist.angular.y = 0.0
            state.twist.angular.z = self.turn_rate if self.started else 0.0
            state.reference_frame = "world"

            try:
                self._set_state(state)
            except rospy.ServiceException:
                pass

            rate.sleep()


if __name__ == "__main__":
    CollisionTargetCruiseNode().run()
