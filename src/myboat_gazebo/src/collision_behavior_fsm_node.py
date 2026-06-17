#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math

import rospy
from std_msgs.msg import Bool, String


class CollisionBehaviorFSMNode:
    """Latch VLM decisions into sustained behavior states for the local planner."""

    INIT = "INIT"
    WAIT_TRIGGER = "WAIT_TRIGGER"
    WAIT_VLM = "WAIT_VLM"
    GEN_NEW_TRAJ = "GEN_NEW_TRAJ"
    EXEC_TRAJ = "EXEC_TRAJ"
    REPLAN_TRAJ = "REPLAN_TRAJ"
    RECOVERY = "RECOVERY"

    def __init__(self):
        rospy.init_node("collision_behavior_fsm_node")

        self.vlm_decision_topic = rospy.get_param("~vlm_decision_topic", "/collision/llm_decision")
        self.behavior_decision_topic = rospy.get_param(
            "~behavior_decision_topic", "/collision/behavior_decision"
        )
        self.trigger_topic = rospy.get_param("~trigger_topic", "/collision/trigger")
        self.publish_rate = float(rospy.get_param("~publish_rate", 20.0))
        self.min_behavior_hold_s = float(rospy.get_param("~min_behavior_hold_s", 8.0))
        self.replan_interval_s = float(rospy.get_param("~replan_interval_s", 2.0))
        self.decision_stale_s = float(rospy.get_param("~decision_stale_s", 18.0))
        self.confidence_min = float(rospy.get_param("~confidence_min", 0.25))
        self.keep_course_release_s = float(rospy.get_param("~keep_course_release_s", 3.0))
        self.pretrigger_behavior_enable = self._as_bool(
            rospy.get_param("~pretrigger_behavior_enable", True)
        )
        self.default_duration_s = float(rospy.get_param("~default_duration_s", 10.0))
        self.debug_enable = self._as_bool(rospy.get_param("~debug_enable", True))

        self._state = self.INIT
        self._trigger = False
        self._last_vlm = {}
        self._last_vlm_stamp = rospy.Time(0)
        self._last_vlm_key = None
        self._active_decision = self._make_keep_course_decision({})
        self._active_key = ("KEEP_COURSE", "")
        self._active_since = rospy.Time(0)
        self._last_replan = rospy.Time(0)
        self._behavior_seq = 0
        self._force_replan_seq = 0
        self._last_pub_state = None

        self._pub = rospy.Publisher(self.behavior_decision_topic, String, queue_size=20)
        rospy.Subscriber(self.vlm_decision_topic, String, self._vlm_cb, queue_size=20)
        rospy.Subscriber(self.trigger_topic, Bool, self._trigger_cb, queue_size=20)

        rospy.loginfo(
            "[collision_behavior_fsm] vlm=%s behavior=%s trigger=%s hold=%.1fs replan=%.1fs stale=%.1fs pretrigger=%s",
            self.vlm_decision_topic,
            self.behavior_decision_topic,
            self.trigger_topic,
            self.min_behavior_hold_s,
            self.replan_interval_s,
            self.decision_stale_s,
            str(self.pretrigger_behavior_enable),
        )

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    @staticmethod
    def _safe_float(v, default_v):
        try:
            f = float(v)
        except Exception:
            return float(default_v)
        return f if math.isfinite(f) else float(default_v)

    @staticmethod
    def _canonical_course_action(action):
        text = str(action or "").strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "KEEP": "KEEP_COURSE",
            "STAND_ON": "KEEP_COURSE",
            "MAINTAIN_COURSE": "KEEP_COURSE",
            "STARBOARD": "TURN_STARBOARD",
            "RIGHT": "TURN_STARBOARD",
            "TURN_RIGHT": "TURN_STARBOARD",
            "PORT": "TURN_PORT",
            "LEFT": "TURN_PORT",
            "TURN_LEFT": "TURN_PORT",
        }
        text = aliases.get(text, text)
        return text if text in ("KEEP_COURSE", "TURN_STARBOARD", "TURN_PORT") else "KEEP_COURSE"

    @staticmethod
    def _canonical_speed_action(action):
        text = str(action or "").strip().upper().replace("-", "_").replace(" ", "_")
        if not text:
            return ""
        aliases = {
            "KEEP": "",
            "KEEP_SPEED": "",
            "MAINTAIN_SPEED": "",
            "SLOW": "SLOW_DOWN",
            "DECELERATE": "SLOW_DOWN",
            "REDUCE_SPEED": "SLOW_DOWN",
            "FAST": "SPEED_UP",
            "ACCELERATE": "SPEED_UP",
            "INCREASE_SPEED": "SPEED_UP",
            "STOP": "EMERGENCY_STOP",
            "FULL_STOP": "EMERGENCY_STOP",
        }
        text = aliases.get(text, text)
        return text if text in ("", "SLOW_DOWN", "SPEED_UP", "EMERGENCY_STOP") else ""

    @staticmethod
    def _combine_action(course_action, speed_action):
        course_action = str(course_action or "KEEP_COURSE")
        speed_action = str(speed_action or "")
        return course_action if not speed_action else "%s+%s" % (course_action, speed_action)

    def _split_action(self, decision):
        constraints = decision.get("trajectory_constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}
        course_action = self._canonical_course_action(
            constraints.get("course_action", decision.get("course_action", ""))
        )
        speed_action = self._canonical_speed_action(
            constraints.get("speed_action", decision.get("speed_action", ""))
        )
        if course_action == "KEEP_COURSE":
            speed_action = ""
        return course_action, speed_action

    def _weights_for(self, course_action, speed_action):
        weights = {
            "action": self._combine_action(course_action, speed_action),
            "course_action": course_action,
            "speed_action": speed_action,
            "strength": 1.0,
            "turn_bias": 0.0,
            "keep_course_bias": 0.0,
            "speed_scale": 1.0,
            "clearance_scale": 1.15,
            "predictive_risk_scale": 1.15,
            "collision_penalty_scale": 1.10,
            "target_penalty_scale": 1.10,
        }
        if course_action == "TURN_STARBOARD":
            weights.update(strength=1.9, turn_bias=-1.55, clearance_scale=1.75,
                           predictive_risk_scale=1.75, collision_penalty_scale=1.55,
                           target_penalty_scale=1.45)
        elif course_action == "TURN_PORT":
            weights.update(strength=1.55, turn_bias=1.35, clearance_scale=1.50,
                           predictive_risk_scale=1.45, collision_penalty_scale=1.35,
                           target_penalty_scale=1.30)
        else:
            weights.update(strength=0.0, keep_course_bias=0.0, clearance_scale=1.0,
                           predictive_risk_scale=1.0, collision_penalty_scale=1.0,
                           target_penalty_scale=1.0)

        if speed_action == "SLOW_DOWN":
            weights.update(speed_scale=0.45, clearance_scale=max(weights["clearance_scale"], 1.35),
                           predictive_risk_scale=max(weights["predictive_risk_scale"], 1.30))
        elif speed_action == "SPEED_UP":
            weights.update(speed_scale=1.15)
        elif speed_action == "EMERGENCY_STOP":
            weights.update(speed_scale=0.05, strength=max(weights["strength"], 1.8),
                           clearance_scale=max(weights["clearance_scale"], 1.8),
                           predictive_risk_scale=max(weights["predictive_risk_scale"], 1.8),
                           collision_penalty_scale=max(weights["collision_penalty_scale"], 1.6))
        return weights

    def _constraints_for(self, course_action, speed_action, base_constraints):
        constraints = dict(base_constraints) if isinstance(base_constraints, dict) else {}
        weights = dict(constraints.get("colreg_weights", {})) if isinstance(constraints.get("colreg_weights"), dict) else {}
        generated = self._weights_for(course_action, speed_action)
        generated.update(weights)

        if course_action == "KEEP_COURSE" and not speed_action:
            for key in ("target_linear_x", "target_angular_z", "min_linear_x"):
                constraints.pop(key, None)
            constraints.update(
                {
                    "course_action": "KEEP_COURSE",
                    "speed_action": "",
                    "colreg_action": "KEEP_COURSE",
                    "colreg_weights": {},
                }
            )
            return constraints

        speed_scale = self._clamp(self._safe_float(generated.get("speed_scale", 1.0), 1.0), 0.0, 1.5)
        if speed_action == "EMERGENCY_STOP":
            target_v = 0.0
            min_v = 0.0
        else:
            target_v = self._clamp(0.70 * speed_scale, 0.05, 1.25)
            min_v = 0.03

        if course_action == "TURN_STARBOARD":
            target_w = -0.40
        elif course_action == "TURN_PORT":
            target_w = 0.40
        else:
            target_w = 0.0

        constraints.update(
            {
                "duration_s": self._safe_float(constraints.get("duration_s", self.default_duration_s), self.default_duration_s),
                "target_linear_x": target_v,
                "target_angular_z": target_w,
                "max_linear_acc": self._safe_float(constraints.get("max_linear_acc", 0.22), 0.22),
                "max_angular_acc": self._safe_float(constraints.get("max_angular_acc", 0.34), 0.34),
                "min_linear_x": min_v,
                "course_action": course_action,
                "speed_action": speed_action,
                "colreg_action": self._combine_action(course_action, speed_action),
                "colreg_weights": generated,
            }
        )
        return constraints

    def _make_keep_course_decision(self, src):
        out = dict(src) if isinstance(src, dict) else {}
        constraints = self._constraints_for("KEEP_COURSE", "", {})
        out.update(
            {
                "course_action": "KEEP_COURSE",
                "speed_action": "",
                "colreg_action": "KEEP_COURSE",
                "colreg_weights": {},
                "trajectory_constraints": constraints,
                "behavior_active": False,
            }
        )
        return out

    def _make_behavior_decision(self, src, course_action, speed_action):
        out = dict(src) if isinstance(src, dict) else {}
        base_constraints = out.get("trajectory_constraints", {})
        constraints = self._constraints_for(course_action, speed_action, base_constraints)
        active = course_action != "KEEP_COURSE" or speed_action in ("SLOW_DOWN", "EMERGENCY_STOP")
        out.update(
            {
                "course_action": course_action,
                "speed_action": speed_action,
                "colreg_action": constraints["colreg_action"],
                "colreg_weights": constraints["colreg_weights"],
                "trajectory_constraints": constraints,
                "behavior_active": bool(active),
            }
        )
        return out

    def _vlm_cb(self, msg):
        try:
            decision = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(decision, dict):
            return
        self._last_vlm = decision
        self._last_vlm_stamp = rospy.Time.now()

    def _trigger_cb(self, msg):
        self._trigger = bool(msg.data)

    def _decision_key(self, decision):
        course_action, speed_action = self._split_action(decision)
        return course_action, speed_action

    def _decision_usable(self, decision):
        if not isinstance(decision, dict) or not decision:
            return False
        confidence = self._safe_float(decision.get("confidence", 0.0), 0.0)
        if confidence < self.confidence_min:
            return False
        if decision.get("action") == "KEEP_COURSE" and not decision.get("course_action"):
            return False
        return True

    @staticmethod
    def _is_force_keep_course(decision):
        if not isinstance(decision, dict):
            return False
        if bool(decision.get("force_keep_course", False)):
            return True
        return (
            decision.get("sensor_front_hazard") is False
            and str(decision.get("course_action", "")).strip().upper() == "KEEP_COURSE"
            and not str(decision.get("speed_action", "")).strip()
        )

    def _transition(self, new_state):
        if new_state == self._state:
            return
        old_state = self._state
        self._state = new_state
        if self.debug_enable:
            rospy.loginfo("[collision_behavior_fsm] [%s] -> [%s]", old_state, new_state)

    def _activate_from_vlm(self, decision, now):
        course_action, speed_action = self._split_action(decision)
        self._active_decision = self._make_behavior_decision(decision, course_action, speed_action)
        self._active_key = (course_action, speed_action)
        self._active_since = now
        self._behavior_seq += 1
        self._force_replan_seq += 1
        self._last_replan = now
        self._last_vlm_key = self._last_vlm.get("llm_call_seq", None)
        self._transition(self.GEN_NEW_TRAJ)

    def _force_keep_course(self, decision, now):
        changed = self._active_key != ("KEEP_COURSE", "") or self._state != self.WAIT_TRIGGER
        self._active_decision = self._make_keep_course_decision(decision)
        self._active_key = ("KEEP_COURSE", "")
        self._active_since = now
        self._last_replan = now
        self._last_vlm_key = decision.get("llm_call_seq", None)
        if changed:
            self._behavior_seq += 1
            self._force_replan_seq += 1
        self._transition(self.WAIT_TRIGGER)

    def _maybe_update_state(self):
        now = rospy.Time.now()
        now_s = now.to_sec()
        have_vlm = bool(self._last_vlm)
        vlm_age = (now - self._last_vlm_stamp).to_sec() if have_vlm else 1e9
        usable_vlm = have_vlm and vlm_age <= self.decision_stale_s and self._decision_usable(self._last_vlm)

        if self._state == self.INIT:
            self._transition(self.WAIT_TRIGGER)

        if usable_vlm and self._is_force_keep_course(self._last_vlm):
            self._force_keep_course(self._last_vlm, now)
            return

        behavior_allowed = bool(self._trigger or self.pretrigger_behavior_enable)
        if not behavior_allowed:
            if (now - self._active_since).to_sec() >= self.keep_course_release_s:
                self._active_decision = self._make_keep_course_decision(self._last_vlm)
                self._active_key = ("KEEP_COURSE", "")
                self._transition(self.WAIT_TRIGGER)
            return

        if not usable_vlm:
            if self._trigger:
                self._transition(self.WAIT_VLM)
            return

        incoming_key = self._decision_key(self._last_vlm)
        incoming_seq = self._last_vlm.get("llm_call_seq", None)
        same_call = incoming_seq is not None and incoming_seq == self._last_vlm_key
        hold_elapsed = (now - self._active_since).to_sec()

        should_switch = not same_call and (
            incoming_key != self._active_key or hold_elapsed >= self.min_behavior_hold_s
        )
        if should_switch:
            self._activate_from_vlm(self._last_vlm, now)
            return

        if self._state == self.GEN_NEW_TRAJ:
            self._transition(self.EXEC_TRAJ)
        elif self._state in (self.WAIT_TRIGGER, self.WAIT_VLM):
            self._transition(self.EXEC_TRAJ)
        elif (now_s - self._last_replan.to_sec()) >= self.replan_interval_s:
            self._force_replan_seq += 1
            self._last_replan = now
            self._transition(self.REPLAN_TRAJ)
        elif self._state == self.REPLAN_TRAJ:
            self._transition(self.EXEC_TRAJ)

    def _publish(self):
        out = dict(self._active_decision)
        out.update(
            {
                "stamp": rospy.Time.now().to_sec(),
                "behavior_state": self._state,
                "behavior_seq": int(self._behavior_seq),
                "planner_force_replan_seq": int(self._force_replan_seq),
                "behavior_source": "vlm_behavior_fsm",
                "trigger": bool(self._trigger),
                "active_since_s": self._active_since.to_sec(),
                "vlm_decision_age_s": (
                    (rospy.Time.now() - self._last_vlm_stamp).to_sec()
                    if self._last_vlm_stamp.to_sec() > 0.0
                    else -1.0
                ),
            }
        )
        self._pub.publish(String(data=json.dumps(out, ensure_ascii=True)))
        if self.debug_enable and self._last_pub_state != (self._state, self._active_key):
            self._last_pub_state = (self._state, self._active_key)
            rospy.loginfo(
                "[collision_behavior_fsm] state=%s seq=%d replan_seq=%d action=%s+%s active=%s",
                self._state,
                int(self._behavior_seq),
                int(self._force_replan_seq),
                self._active_key[0],
                self._active_key[1],
                str(out.get("behavior_active", False)),
            )

    def run(self):
        rate = rospy.Rate(max(1.0, self.publish_rate))
        while not rospy.is_shutdown():
            self._maybe_update_state()
            self._publish()
            rate.sleep()


if __name__ == "__main__":
    CollisionBehaviorFSMNode().run()
