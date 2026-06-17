#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import re

import nlopt
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Empty, String


class CollisionTrajectoryPlannerNode:
    """Trajectory planner for collision avoidance.

    VO mode: pure geometric Velocity Obstacle (Fiorini & Shiller 1998).
      - Samples (v, omega); per-sample checks whether the instantaneous
        world velocity lies inside any VO cone (no bspline, no forward
        simulation over full horizon).
      - Path returned is a simple kinematic arc for pure-pursuit following.
    SVO / AUTO mode: pure geometric VO with semantic collision-body inflation
        (vessel-type-aware radius/safety scaling, no bspline, no n-step sim).
    MPC mode: NLopt-based multi-block optimisation (unchanged).
    """

    def __init__(self):
        rospy.init_node("collision_trajectory_planner_node")

        self.ego_odom_topic = rospy.get_param("~ego_odom_topic", "/myboat/odom")
        self.trigger_topic = rospy.get_param("~trigger_topic", "/collision/trigger")
        self.wait_for_start_trigger = bool(rospy.get_param("~wait_for_start_trigger", True))
        self.start_trigger_topic = rospy.get_param("~start_trigger_topic", "/traj_start_trigger")
        self.decision_topic = rospy.get_param("~decision_topic", "/collision/llm_decision")
        self.perception_targets_topic = rospy.get_param(
            "~perception_targets_topic", "/collision/perception/targets"
        )
        self.planner_mode = self._normalize_planner_mode(
            rospy.get_param("~planner_mode", "svo")
        )
        self.cmd_topic = rospy.get_param("~cmd_topic", "/collision/trajectory_cmd_vel")
        self.path_topic = rospy.get_param("~path_topic", "/collision/planned_path")
        self.current_waypoint_topic = rospy.get_param("~current_waypoint_topic", "/collision/current_waypoint")
        self.mpc_near_miss_event_topic = rospy.get_param(
            "~mpc_near_miss_event_topic", "/collision/mpc_near_miss_event"
        )
        self.path_frame = rospy.get_param("~path_frame", "world")
        self.publish_rate = float(rospy.get_param("~publish_rate", 20.0))

        self.straight_horizon_m = float(rospy.get_param("~straight_horizon_m", 25.0))
        self.curve_horizon_m = float(rospy.get_param("~curve_horizon_m", 22.0))
        self.path_sample_count = int(rospy.get_param("~path_sample_count", 70))
        self.curve_lateral_m = float(rospy.get_param("~curve_lateral_m", 4.5))
        self.lookahead_m = float(rospy.get_param("~lookahead_m", 2.5))
        self.kp_heading = float(rospy.get_param("~kp_heading", 1.2))
        self.max_turn_rate = float(rospy.get_param("~max_turn_rate", 0.40))
        self.default_speed = float(rospy.get_param("~default_speed", 0.65))
        self.max_speed = float(rospy.get_param("~max_speed", 1.2))
        self.nav_goal_x = float(rospy.get_param("~nav_goal_x", -410.0))
        self.nav_goal_y = float(rospy.get_param("~nav_goal_y", 260.0))
        self.nav_use_goal_in_avoid = bool(rospy.get_param("~nav_use_goal_in_avoid", True))
        self.waypoint_interp_step_m = float(rospy.get_param("~waypoint_interp_step_m", 0.8))
        self.opt_dt_s = float(rospy.get_param("~opt_dt_s", 0.3))
        self.opt_min_horizon_s = float(rospy.get_param("~opt_min_horizon_s", 3.0))
        self.opt_max_horizon_s = float(rospy.get_param("~opt_max_horizon_s", 12.0))
        self.opt_replan_interval_s = float(rospy.get_param("~opt_replan_interval_s", 0.5))
        self.opt_speed_samples = int(rospy.get_param("~opt_speed_samples", 7))
        self.opt_turn_samples = int(rospy.get_param("~opt_turn_samples", 17))
        self.avoid_constraint_hold_s = float(rospy.get_param("~avoid_constraint_hold_s", 4.0))
        self.vo_time_horizon_s = float(rospy.get_param("~vo_time_horizon_s", 10.0))
        self.vo_sample_dt_s = float(rospy.get_param("~vo_sample_dt_s", 0.5))
        self.vo_ego_radius_m = float(rospy.get_param("~vo_ego_radius_m", 1.2))
        self.vo_target_radius_m = float(rospy.get_param("~vo_target_radius_m", 1.4))
        self.vo_safety_margin_m = float(rospy.get_param("~vo_safety_margin_m", 0.6))
        self.vo_progress_weight = float(rospy.get_param("~vo_progress_weight", 1.2))
        self.vo_heading_weight = float(rospy.get_param("~vo_heading_weight", 0.7))
        self.vo_clearance_weight = float(rospy.get_param("~vo_clearance_weight", 2.5))
        self.vo_collision_penalty = float(rospy.get_param("~vo_collision_penalty", 1200.0))
        self.mpc_horizon_s = float(rospy.get_param("~mpc_horizon_s", 8.0))
        self.mpc_sample_dt_s = float(rospy.get_param("~mpc_sample_dt_s", 0.4))
        self.mpc_goal_weight = float(rospy.get_param("~mpc_goal_weight", 3.8))
        self.mpc_progress_weight = float(rospy.get_param("~mpc_progress_weight", 1.2))
        self.mpc_heading_weight = float(rospy.get_param("~mpc_heading_weight", 0.8))
        self.mpc_clearance_weight = float(rospy.get_param("~mpc_clearance_weight", 3.0))
        self.mpc_collision_penalty = float(rospy.get_param("~mpc_collision_penalty", 2200.0))
        self.mpc_smooth_weight = float(rospy.get_param("~mpc_smooth_weight", 2.2))
        self.mpc_feasibility_weight = float(rospy.get_param("~mpc_feasibility_weight", 3.5))
        self.mpc_terminal_speed_weight = float(rospy.get_param("~mpc_terminal_speed_weight", 0.9))
        self.mpc_predictive_risk_weight = float(rospy.get_param("~mpc_predictive_risk_weight", 8.5))
        self.mpc_max_speed = float(rospy.get_param("~mpc_max_speed", 1.1))
        self.mpc_turn_weight = float(rospy.get_param("~mpc_turn_weight", 1.6))
        self.mpc_turn_change_weight = float(rospy.get_param("~mpc_turn_change_weight", 4.5))
        self.mpc_yaw_smooth_weight = float(rospy.get_param("~mpc_yaw_smooth_weight", 0.35))
        self.mpc_tracking_blend = float(rospy.get_param("~mpc_tracking_blend", 0.80))
        self.mpc_lookahead_m = float(rospy.get_param("~mpc_lookahead_m", 4.5))
        self.mpc_heading_gain = float(rospy.get_param("~mpc_heading_gain", 0.55))
        self.mpc_control_blocks = int(rospy.get_param("~mpc_control_blocks", 4))
        self.mpc_solver_maxiter = int(rospy.get_param("~mpc_solver_maxiter", 60))
        self.mpc_solver_ftol = float(rospy.get_param("~mpc_solver_ftol", 1e-3))
        self.mpc_nlopt_algorithm = int(rospy.get_param("~mpc_nlopt_algorithm", 11))
        self.mpc_nlopt_maxtime_s = float(rospy.get_param("~mpc_nlopt_maxtime_s", 0.04))
        self.mpc_nlopt_fd_eps = float(rospy.get_param("~mpc_nlopt_fd_eps", 1e-3))
        self.mpc_geometry_avoidance_enable = self._as_bool(
            rospy.get_param("~mpc_geometry_avoidance_enable", False)
        )
        self.mpc_colreg_enable = self._as_bool(rospy.get_param("~mpc_colreg_enable", True))
        self.mpc_colreg_only_when_trigger = self._as_bool(
            rospy.get_param("~mpc_colreg_only_when_trigger", True)
        )
        self.mpc_colreg_turn_bias_weight = float(rospy.get_param("~mpc_colreg_turn_bias_weight", 7.5))
        self.mpc_colreg_keep_course_weight = float(rospy.get_param("~mpc_colreg_keep_course_weight", 3.0))
        self.mpc_colreg_speed_weight = float(rospy.get_param("~mpc_colreg_speed_weight", 1.5))
        self.mpc_colreg_min_turn_rate = float(rospy.get_param("~mpc_colreg_min_turn_rate", 0.14))
        self.mpc_goal_rejoin_enable = self._as_bool(rospy.get_param("~mpc_goal_rejoin_enable", True))
        self.mpc_goal_rejoin_corridor_width_m = float(rospy.get_param("~mpc_goal_rejoin_corridor_width_m", 6.0))
        self.mpc_goal_rejoin_lookahead_m = float(rospy.get_param("~mpc_goal_rejoin_lookahead_m", 28.0))
        self.mpc_goal_rejoin_emergency_distance_m = float(
            rospy.get_param("~mpc_goal_rejoin_emergency_distance_m", 4.2)
        )
        self.mpc_goal_rejoin_speed_scale = float(rospy.get_param("~mpc_goal_rejoin_speed_scale", 0.90))
        self.mpc_near_miss_clearance_m = float(rospy.get_param("~mpc_near_miss_clearance_m", 1.0))
        self.mpc_near_miss_turn_threshold = float(rospy.get_param("~mpc_near_miss_turn_threshold", 0.08))
        self.mpc_near_miss_min_interval_s = float(rospy.get_param("~mpc_near_miss_min_interval_s", 2.0))
        self.parallel_nav_enable = bool(rospy.get_param("~parallel_nav_enable", False))
        self.parallel_nav_horizon_s = float(rospy.get_param("~parallel_nav_horizon_s", 9.0))
        self.parallel_nav_sample_dt_s = float(rospy.get_param("~parallel_nav_sample_dt_s", 0.4))
        self.parallel_nav_goal_weight = float(rospy.get_param("~parallel_nav_goal_weight", 1.6))
        self.parallel_nav_heading_weight = float(rospy.get_param("~parallel_nav_heading_weight", 0.9))
        self.parallel_nav_clearance_weight = float(rospy.get_param("~parallel_nav_clearance_weight", 3.2))
        self.parallel_nav_collision_penalty = float(rospy.get_param("~parallel_nav_collision_penalty", 1400.0))
        self.parallel_nav_smooth_weight = float(rospy.get_param("~parallel_nav_smooth_weight", 1.4))
        self.parallel_nav_semantic_weight_enable = self._as_bool(
            rospy.get_param("~parallel_nav_semantic_weight_enable", True)
        )
        self.parallel_nav_vlm_bias_weight = float(rospy.get_param("~parallel_nav_vlm_bias_weight", 1.1))
        self.parallel_nav_blend_alpha = float(rospy.get_param("~parallel_nav_blend_alpha", 0.65))
        self.channel_constraints_enable = bool(rospy.get_param("~channel_constraints_enable", False))
        self.channel_left_y = float(rospy.get_param("~channel_left_y", 254.0))
        self.channel_right_y = float(rospy.get_param("~channel_right_y", 266.0))
        self.channel_x_min = float(rospy.get_param("~channel_x_min", -446.0))
        self.channel_x_max = float(rospy.get_param("~channel_x_max", -422.0))
        self.channel_x_step = float(rospy.get_param("~channel_x_step", 12.0))
        self.channel_obstacle_radius_m = float(rospy.get_param("~channel_obstacle_radius_m", 0.75))
        self.channel_boundary_penalty = float(rospy.get_param("~channel_boundary_penalty", 5000.0))
        self.channel_boundary_margin_m = float(rospy.get_param("~channel_boundary_margin_m", 0.0))
        self.goal_retreat_penalty = float(rospy.get_param("~goal_retreat_penalty", 1800.0))
        self.goal_retreat_margin_m = float(rospy.get_param("~goal_retreat_margin_m", 0.4))
        self.nav_linear_response_tau_s = float(rospy.get_param("~nav_linear_response_tau_s", 1.8))
        self.nav_angular_response_tau_s = float(rospy.get_param("~nav_angular_response_tau_s", 2.6))
        self.nav_response_deadtime_s = float(rospy.get_param("~nav_response_deadtime_s", 0.5))
        self.nav_latency_margin_s = float(rospy.get_param("~nav_latency_margin_s", 0.8))
        self.nav_latency_buffer_max_m = float(rospy.get_param("~nav_latency_buffer_max_m", 1.6))
        self.nav_predictive_gate_scale = float(rospy.get_param("~nav_predictive_gate_scale", 1.8))
        self.nav_predictive_time_bias_s = float(rospy.get_param("~nav_predictive_time_bias_s", 4.0))
        self.nav_predictive_risk_weight = float(rospy.get_param("~nav_predictive_risk_weight", 9.0))
        self.vo_predictive_risk_weight = float(rospy.get_param("~vo_predictive_risk_weight", 7.0))
        self.identification_mode_enable = bool(rospy.get_param("~identification_mode_enable", True))
        self.identification_assoc_confidence_min = float(rospy.get_param("~identification_assoc_confidence_min", 0.55))
        self.identification_camera_hfov_deg = float(rospy.get_param("~identification_camera_hfov_deg", 90.0))
        self.identification_fov_margin_deg = float(rospy.get_param("~identification_fov_margin_deg", 8.0))
        self.identification_speed = float(rospy.get_param("~identification_speed", 0.14))
        self.identification_max_speed = float(rospy.get_param("~identification_max_speed", 0.22))
        self.identification_turn_rate_limit = float(rospy.get_param("~identification_turn_rate_limit", 0.20))
        self.identification_fov_weight = float(rospy.get_param("~identification_fov_weight", 18.0))
        self.identification_center_weight = float(rospy.get_param("~identification_center_weight", 0.35))
        self.identification_progress_weight = float(rospy.get_param("~identification_progress_weight", 0.20))
        self.identification_risk_weight = float(rospy.get_param("~identification_risk_weight", 2.0))
        self.debug_enable = bool(rospy.get_param("~debug_enable", True))
        self.plain_mpc_profile_enable = bool(rospy.get_param("~plain_mpc_profile_enable", True))
        self.plain_mpc_target_radius_scale = float(rospy.get_param("~plain_mpc_target_radius_scale", 0.76))
        self.plain_mpc_safety_margin_scale = float(rospy.get_param("~plain_mpc_safety_margin_scale", 0.72))
        self.plain_mpc_clearance_weight_scale = float(rospy.get_param("~plain_mpc_clearance_weight_scale", 0.72))
        self.plain_mpc_collision_penalty_scale = float(rospy.get_param("~plain_mpc_collision_penalty_scale", 0.74))
        self.plain_mpc_predictive_weight_scale = float(rospy.get_param("~plain_mpc_predictive_weight_scale", 0.72))
        self.plain_mpc_speed_scale = float(rospy.get_param("~plain_mpc_speed_scale", 1.12))
        self.plain_mpc_max_speed_scale = float(rospy.get_param("~plain_mpc_max_speed_scale", 1.18))

        self._apply_plain_mpc_profile()

        self._ego = None
        self._trigger = False
        self._started = not self.wait_for_start_trigger
        self._llm_decision = {}
        self._decision_signature = ""
        self._decision_force_replan_seq = -1
        self._last_applied_force_replan_seq = -1
        self._last_decision_signature = ""
        self._perception_targets = []
        self._planned_xy = []
        self._plan_mode = "none"
        self._last_cmd_v = 0.0
        self._last_cmd_w = 0.0
        self._last_best_v = 0.0
        self._last_best_w = 0.0
        self._last_plan_cost = 0.0
        self._last_vo_min_clearance = -1.0
        self._last_vo_target_count = 0
        self._last_vo_violations = 0
        self._last_parallel_nav_min_clearance = -1.0
        self._last_parallel_nav_target_count = 0
        self._last_parallel_nav_cost = 0.0
        self._last_parallel_nav_blend = 0.0
        self._last_parallel_nav_mean_vlm_weight = 0.0
        self._last_identification_pending_count = 0
        self._last_identification_out_of_fov_count = 0
        self._mpc_goal_rejoin_active = False
        self._last_replan_t = 0.0
        self._last_trigger_state = False
        self._last_cmd_t = 0.0
        self._constraint_hold_until_t = 0.0
        self._latched_cset = None
        self._avoid_anchor_xyyaw = None
        self._mpc_near_miss_active = False
        self._mpc_near_miss_seq = 0
        self._last_mpc_near_miss_event_t = 0.0

        self._path_pub = rospy.Publisher(self.path_topic, Path, queue_size=10)
        self._current_wp_pub = rospy.Publisher(self.current_waypoint_topic, PoseStamped, queue_size=10)
        self._cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=20)
        self._mpc_near_miss_pub = rospy.Publisher(
            self.mpc_near_miss_event_topic,
            String,
            queue_size=20,
        )

        rospy.Subscriber(self.ego_odom_topic, Odometry, self._ego_cb, queue_size=20)
        rospy.Subscriber(self.trigger_topic, Bool, self._trigger_cb, queue_size=20)
        if self.wait_for_start_trigger:
            rospy.Subscriber(self.start_trigger_topic, Empty, self._start_trigger_cb, queue_size=1)
        rospy.Subscriber(self.decision_topic, String, self._decision_cb, queue_size=20)
        if self.perception_targets_topic:
            rospy.Subscriber(
                self.perception_targets_topic,
                String,
                self._perception_targets_cb,
                queue_size=20,
            )

        rospy.loginfo(
            "[collision_traj_planner] mode=%s mpc_geom_avoid=%s parallel_nav=%s nav_semantic_weight=%s channel_constraints=%s wait_for_start=%s start_trigger=%s ego=%s trigger=%s decision=%s targets=%s cmd=%s path=%s",
            self.planner_mode,
            str(self.mpc_geometry_avoidance_enable),
            str(self.parallel_nav_enable),
            str(self.parallel_nav_semantic_weight_enable),
            str(self.channel_constraints_enable),
            str(self.wait_for_start_trigger),
            self.start_trigger_topic,
            self.ego_odom_topic,
            self.trigger_topic,
            self.decision_topic,
            self.perception_targets_topic,
            self.cmd_topic,
            self.path_topic,
        )

    @staticmethod
    def _normalize_planner_mode(mode_raw):
        m = str(mode_raw).strip().lower()
        if m in ("mpc", "dmpc", "smpc", "model_predictive_control", "model-predictive-control"):
            return "mpc"
        if m in ("vo", "velocity_obstacle", "velocity-obstacle"):
            return "vo"
        if m in ("svo", "semantic_vo", "semantic-vo", "vlm"):
            return "svo"
        return "svo"

    def _apply_plain_mpc_profile(self):
        if self.planner_mode != "mpc":
            return
        if not self.mpc_geometry_avoidance_enable:
            self.mpc_clearance_weight = 0.0
            self.mpc_collision_penalty = 0.0
            self.mpc_predictive_risk_weight = 0.0
            self.vo_predictive_risk_weight = 0.0
            rospy.loginfo(
                "[collision_traj_planner] MPC geometry avoidance disabled: target clearance/collision/predictive costs are zeroed"
            )
            return
        if self.parallel_nav_semantic_weight_enable:
            return
        if not self.plain_mpc_profile_enable:
            return

        self.vo_target_radius_m = max(
            0.4,
            self.vo_target_radius_m * max(0.2, self.plain_mpc_target_radius_scale),
        )
        self.vo_safety_margin_m = max(
            0.15,
            self.vo_safety_margin_m * max(0.2, self.plain_mpc_safety_margin_scale),
        )
        self.mpc_clearance_weight = max(0.1, self.mpc_clearance_weight * max(0.1, self.plain_mpc_clearance_weight_scale))
        self.mpc_collision_penalty = max(
            10.0,
            self.mpc_collision_penalty * max(0.1, self.plain_mpc_collision_penalty_scale),
        )
        self.mpc_predictive_risk_weight = max(
            0.1,
            self.mpc_predictive_risk_weight * max(0.1, self.plain_mpc_predictive_weight_scale),
        )
        self.default_speed = self._clamp(
            self.default_speed * max(0.2, self.plain_mpc_speed_scale),
            0.05,
            self.max_speed,
        )
        self.mpc_max_speed = self._clamp(
            max(self.default_speed, self.mpc_max_speed * max(0.2, self.plain_mpc_max_speed_scale)),
            self.default_speed,
            self.max_speed,
        )

        rospy.loginfo(
            "[collision_traj_planner] applied reduced plain-MPC profile: target_r=%.2f safety_margin=%.2f clearance_w=%.2f collision_penalty=%.2f predrisk_w=%.2f default_v=%.2f vmax=%.2f",
            self.vo_target_radius_m,
            self.vo_safety_margin_m,
            self.mpc_clearance_weight,
            self.mpc_collision_penalty,
            self.mpc_predictive_risk_weight,
            self.default_speed,
            self.mpc_max_speed,
        )

    def _publish_current_waypoint(self, wp, frame_id, yaw):
        if wp is None:
            return
        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = frame_id
        ps.pose.position.x = float(wp[0])
        ps.pose.position.y = float(wp[1])
        ps.pose.position.z = 0.0
        ps.pose.orientation.z = math.sin(0.5 * yaw)
        ps.pose.orientation.w = math.cos(0.5 * yaw)
        self._current_wp_pub.publish(ps)

    @staticmethod
    def _norm_pi(a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    @staticmethod
    def _yaw_from_quat(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _ego_cb(self, msg):
        self._ego = msg

    def _trigger_cb(self, msg):
        self._trigger = bool(msg.data)

    def _start_trigger_cb(self, _msg):
        if not self._started:
            self._started = True
            rospy.loginfo(
                "[collision_traj_planner] start trigger received from %s, planner enabled",
                self.start_trigger_topic,
            )

    def _decision_cb(self, msg):
        try:
            self._llm_decision = json.loads(msg.data)
        except Exception:
            self._llm_decision = {}
            self._decision_signature = ""
            self._decision_force_replan_seq = -1
            return
        if not isinstance(self._llm_decision, dict):
            self._llm_decision = {}
            self._decision_signature = ""
            self._decision_force_replan_seq = -1
            return
        constraints = self._llm_decision.get("trajectory_constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}
        weights = constraints.get("colreg_weights", {})
        if not isinstance(weights, dict):
            weights = self._llm_decision.get("colreg_weights", {})
        if not isinstance(weights, dict):
            weights = {}
        self._decision_signature = "|".join(
            [
                str(self._llm_decision.get("behavior_seq", "")),
                str(self._llm_decision.get("behavior_state", "")),
                str(self._llm_decision.get("course_action", constraints.get("course_action", ""))),
                str(self._llm_decision.get("speed_action", constraints.get("speed_action", ""))),
                str(self._llm_decision.get("colreg_action", constraints.get("colreg_action", ""))),
                str(weights.get("turn_bias", "")),
                str(weights.get("speed_scale", "")),
            ]
        )
        self._decision_force_replan_seq = int(
            self._safe_float(self._llm_decision.get("planner_force_replan_seq", -1), -1)
        )

    def _perception_targets_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            self._perception_targets = []
            return
        targets = data.get("targets", []) if isinstance(data, dict) else []
        if isinstance(targets, list):
            self._perception_targets = targets
        else:
            self._perception_targets = []

    def _enter_constraint_hold(self, cset, now_s):
        self._latched_cset = dict(cset)
        self._constraint_hold_until_t = max(self._constraint_hold_until_t, now_s + max(0.0, self.avoid_constraint_hold_s))

    def _active_constraint_set(self, now_s):
        if now_s < self._constraint_hold_until_t and isinstance(self._latched_cset, dict):
            return dict(self._latched_cset)
        return self._extract_constraint_set()

    @staticmethod
    def _to_anchor_frame(px, py, anchor):
        ax, ay, ayaw = anchor
        dx = px - ax
        dy = py - ay
        x_rel = math.cos(ayaw) * dx + math.sin(ayaw) * dy
        y_rel = -math.sin(ayaw) * dx + math.cos(ayaw) * dy
        return x_rel, y_rel

    @staticmethod
    def _bspline4(p0, p1, p2, p3, u):
        b0 = ((1.0 - u) ** 3) / 6.0
        b1 = (3.0 * u**3 - 6.0 * u**2 + 4.0) / 6.0
        b2 = (-3.0 * u**3 + 3.0 * u**2 + 3.0 * u + 1.0) / 6.0
        b3 = (u**3) / 6.0
        x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
        y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
        return (x, y)

    def _make_straight_path(self, x0, y0, yaw):
        n = max(20, self.path_sample_count)
        pts = []
        for i in range(n):
            t = float(i) / float(max(1, n - 1))
            d = t * self.straight_horizon_m
            pts.append((x0 + d * math.cos(yaw), y0 + d * math.sin(yaw)))
        return pts

    def _make_goal_path(self, x0, y0):
        dist = math.hypot(self.nav_goal_x - x0, self.nav_goal_y - y0)
        n = max(8, int(max(1.0, dist) / max(0.2, self.waypoint_interp_step_m)) + 1)
        pts = []
        for i in range(n):
            t = float(i) / float(max(1, n - 1))
            pts.append((x0 + t * (self.nav_goal_x - x0), y0 + t * (self.nav_goal_y - y0)))
        return pts

    def _make_curve_path(self, x0, y0, yaw, turn_right=True):
        f = (math.cos(yaw), math.sin(yaw))
        l = (-math.sin(yaw), math.cos(yaw))
        lat_sign = -1.0 if turn_right else 1.0
        lat = lat_sign * self.curve_lateral_m

        ctrl = [
            (x0, y0),
            (x0 + 4.0 * f[0], y0 + 4.0 * f[1]),
            (x0 + 8.0 * f[0] + 0.6 * lat * l[0], y0 + 8.0 * f[1] + 0.6 * lat * l[1]),
            (x0 + 12.0 * f[0] + 1.0 * lat * l[0], y0 + 12.0 * f[1] + 1.0 * lat * l[1]),
            (x0 + 17.0 * f[0] + 0.7 * lat * l[0], y0 + 17.0 * f[1] + 0.7 * lat * l[1]),
            (x0 + self.curve_horizon_m * f[0] + 0.2 * lat * l[0], y0 + self.curve_horizon_m * f[1] + 0.2 * lat * l[1]),
        ]

        segs = len(ctrl) - 3
        sample_total = max(30, self.path_sample_count)
        pts = []
        for k in range(sample_total):
            s = (float(k) / float(max(1, sample_total - 1))) * segs
            i = int(min(segs - 1, max(0, math.floor(s))))
            u = s - float(i)
            pts.append(self._bspline4(ctrl[i], ctrl[i + 1], ctrl[i + 2], ctrl[i + 3], u))
        return pts

    def _publish_path(self, pts, frame_id):
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = frame_id

        for i, p in enumerate(pts):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = p[0]
            ps.pose.position.y = p[1]
            ps.pose.position.z = 0.0

            if i < len(pts) - 1:
                dx = pts[i + 1][0] - p[0]
                dy = pts[i + 1][1] - p[1]
                yaw = math.atan2(dy, dx)
            else:
                yaw = 0.0
            ps.pose.orientation.z = math.sin(yaw * 0.5)
            ps.pose.orientation.w = math.cos(yaw * 0.5)
            path.poses.append(ps)

        self._path_pub.publish(path)

    def _select_lookahead(self, x, y, lookahead_m=None):
        if not self._planned_xy:
            return None

        target_lookahead = self.lookahead_m if lookahead_m is None else max(0.5, float(lookahead_m))

        # Find nearest point first.
        nearest = 0
        nearest_d2 = 1e18
        for i, p in enumerate(self._planned_xy):
            dx = p[0] - x
            dy = p[1] - y
            d2 = dx * dx + dy * dy
            if d2 < nearest_d2:
                nearest_d2 = d2
                nearest = i

        # Move forward along path until lookahead distance reached.
        acc = 0.0
        for i in range(nearest, len(self._planned_xy) - 1):
            p0 = self._planned_xy[i]
            p1 = self._planned_xy[i + 1]
            ds = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            acc += ds
            if acc >= target_lookahead:
                return p1
        return self._planned_xy[-1]

    def _constraint_speed(self):
        # Kept for backward compatibility; optimization path no longer uses template speeds.
        c = (self._llm_decision.get("trajectory_constraints") or {})
        v = c.get("target_linear_x", self.default_speed)
        try:
            v = float(v)
        except Exception:
            v = self.default_speed
        return max(0.05, min(self.max_speed, v))

    def _constraint_turn_dir(self):
        # Kept for backward compatibility; optimization path no longer uses template turn dirs.
        c = (self._llm_decision.get("trajectory_constraints") or {})
        try:
            w = float(c.get("target_angular_z", 0.0))
            if abs(w) < 1e-3:
                return None
            return w < 0.0
        except Exception:
            return None

    def _colreg_constraints(self):
        if not isinstance(self._llm_decision, dict):
            return "", {}
        source = str(self._llm_decision.get("classification_source", self._llm_decision.get("source", ""))).strip()
        pending_safety = source == "pending_safety_pre_llm"
        behavior_active = bool(self._llm_decision.get("behavior_active", False))
        if self.mpc_colreg_only_when_trigger and not self._trigger and not pending_safety and not behavior_active:
            return "", {}
        constraints = self._llm_decision.get("trajectory_constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}
        weights = constraints.get("colreg_weights", {})
        if not isinstance(weights, dict):
            weights = self._llm_decision.get("colreg_weights", {})
        if not isinstance(weights, dict):
            weights = {}
        action = str(
            constraints.get(
                "colreg_action",
                self._llm_decision.get("colreg_action", weights.get("action", "")),
            )
            or ""
        ).strip().upper()
        course_action = self._canonical_course_action(
            constraints.get(
                "course_action",
                self._llm_decision.get("course_action", weights.get("course_action", "")),
            )
        )
        speed_action = self._canonical_speed_action(
            constraints.get(
                "speed_action",
                self._llm_decision.get("speed_action", weights.get("speed_action", "")),
            )
        )
        if not str(constraints.get("course_action", self._llm_decision.get("course_action", "")) or "").strip():
            legacy_course, legacy_speed = self._split_colreg_action(action)
            course_action = legacy_course
            if not str(constraints.get("speed_action", self._llm_decision.get("speed_action", "")) or "").strip():
                speed_action = legacy_speed
        if course_action == "KEEP_COURSE":
            speed_action = ""
        action = self._combine_colreg_action(course_action, speed_action)
        if action and not weights:
            weights = self._default_colreg_weights(course_action, speed_action)
        if action and isinstance(weights, dict):
            weights = dict(weights)
            weights.setdefault("action", action)
            weights.setdefault("course_action", course_action)
            weights.setdefault("speed_action", speed_action)
        return action, weights

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
            "RIGHT_TURN": "TURN_STARBOARD",
            "PORT": "TURN_PORT",
            "LEFT": "TURN_PORT",
            "TURN_LEFT": "TURN_PORT",
            "LEFT_TURN": "TURN_PORT",
        }
        text = aliases.get(text, text)
        return text if text in ("KEEP_COURSE", "TURN_STARBOARD", "TURN_PORT") else "TURN_STARBOARD"

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
            "EMERGENCY": "EMERGENCY_STOP",
            "FULL_STOP": "EMERGENCY_STOP",
        }
        text = aliases.get(text, text)
        return text if text in ("", "SLOW_DOWN", "SPEED_UP", "EMERGENCY_STOP") else ""

    @classmethod
    def _split_colreg_action(cls, action):
        text = str(action or "").strip().upper().replace("-", "_").replace(" ", "_")
        parts = [p for p in re.split(r"[+/,;|]+", text) if p]
        course_action = ""
        speed_action = ""
        for part in parts or [text]:
            if part in ("KEEP", "KEEP_COURSE", "STAND_ON", "MAINTAIN_COURSE", "STARBOARD", "RIGHT", "TURN_RIGHT", "RIGHT_TURN", "TURN_STARBOARD", "PORT", "LEFT", "TURN_LEFT", "LEFT_TURN", "TURN_PORT"):
                course_action = cls._canonical_course_action(part)
            elif part in ("SLOW", "SLOW_DOWN", "DECELERATE", "REDUCE_SPEED", "FAST", "SPEED_UP", "ACCELERATE", "INCREASE_SPEED", "STOP", "EMERGENCY", "FULL_STOP", "EMERGENCY_STOP"):
                speed_action = cls._canonical_speed_action(part)
        return course_action or "KEEP_COURSE", speed_action

    @classmethod
    def _combine_colreg_action(cls, course_action, speed_action):
        course = cls._canonical_course_action(course_action)
        speed = cls._canonical_speed_action(speed_action)
        return course if not speed else "%s+%s" % (course, speed)

    def _default_colreg_weights(self, action, speed_action=None):
        if speed_action is None:
            course_action, speed_action = self._split_colreg_action(action)
        else:
            course_action = self._canonical_course_action(action)
            speed_action = self._canonical_speed_action(speed_action)
        action = self._combine_colreg_action(course_action, speed_action)
        weights = {
            "action": action,
            "course_action": course_action,
            "speed_action": speed_action,
            "strength": 1.0,
            "turn_bias": 0.0,
            "keep_course_bias": 0.0,
            "speed_scale": 1.0,
            "clearance_scale": 1.0,
            "predictive_risk_scale": 1.0,
            "collision_penalty_scale": 1.0,
            "target_penalty_scale": 1.0,
        }
        if course_action == "KEEP_COURSE":
            weights.update(
                strength=0.0,
                keep_course_bias=0.0,
                clearance_scale=1.0,
                predictive_risk_scale=1.0,
                collision_penalty_scale=1.0,
                target_penalty_scale=1.0,
            )
        elif course_action == "TURN_STARBOARD":
            weights.update(
                strength=1.55,
                turn_bias=-1.25,
                clearance_scale=1.55,
                predictive_risk_scale=1.55,
                collision_penalty_scale=1.45,
                target_penalty_scale=1.35,
            )
        elif course_action == "TURN_PORT":
            weights.update(
                strength=1.05,
                turn_bias=1.0,
                clearance_scale=1.25,
                predictive_risk_scale=1.20,
                collision_penalty_scale=1.15,
                target_penalty_scale=1.12,
            )
        if speed_action == "SLOW_DOWN":
            weights.update(
                speed_scale=0.45,
                clearance_scale=max(float(weights.get("clearance_scale", 1.0)), 1.20),
                predictive_risk_scale=max(float(weights.get("predictive_risk_scale", 1.0)), 1.15),
                collision_penalty_scale=max(float(weights.get("collision_penalty_scale", 1.0)), 1.10),
            )
        elif speed_action == "SPEED_UP":
            weights.update(
                speed_scale=1.15,
                clearance_scale=max(float(weights.get("clearance_scale", 1.0)), 1.05),
                predictive_risk_scale=max(float(weights.get("predictive_risk_scale", 1.0)), 1.05),
            )
        elif speed_action == "EMERGENCY_STOP":
            weights.update(
                strength=max(float(weights.get("strength", 1.0)), 1.5),
                speed_scale=0.05,
                clearance_scale=max(float(weights.get("clearance_scale", 1.0)), 1.45),
                predictive_risk_scale=max(float(weights.get("predictive_risk_scale", 1.0)), 1.45),
                collision_penalty_scale=max(float(weights.get("collision_penalty_scale", 1.0)), 1.35),
                target_penalty_scale=max(float(weights.get("target_penalty_scale", 1.0)), 1.25),
            )
        return weights

    def _colreg_weight(self, weights, key, default_v, lo, hi):
        return self._clamp(self._safe_float(weights.get(key, default_v), default_v), lo, hi)

    def _colreg_mpc_bias_cost(self, controls):
        if (not self.mpc_colreg_enable) or (not controls):
            return 0.0
        action, weights = self._colreg_constraints()
        if not action and not weights:
            return 0.0

        strength = self._colreg_weight(weights, "strength", 1.0, 0.0, 2.5)
        turn_bias = self._colreg_weight(weights, "turn_bias", 0.0, -2.0, 2.0)
        keep_course_bias = self._colreg_weight(weights, "keep_course_bias", 0.0, 0.0, 2.5)
        speed_scale = self._colreg_weight(weights, "speed_scale", 1.0, 0.05, 1.5)
        if strength <= 1e-6 and abs(turn_bias) <= 1e-6 and keep_course_bias <= 1e-6 and abs(speed_scale - 1.0) <= 1e-6:
            return 0.0

        desired_w = self._clamp(
            0.58 * self.max_turn_rate * turn_bias,
            -abs(self.max_turn_rate),
            abs(self.max_turn_rate),
        )
        desired_v = self._clamp(self.default_speed * speed_scale, 0.03, self.max_speed)
        speed_active = abs(speed_scale - 1.0) > 0.02 or action in (
            "SLOW_DOWN",
            "SPEED_UP",
            "EMERGENCY_STOP",
        )

        cost = 0.0
        for i, (v, omega) in enumerate(controls):
            time_weight = 1.0 + max(0.0, 1.0 - float(i) / float(max(1, len(controls))))
            if abs(turn_bias) > 1e-3:
                cost += time_weight * self.mpc_colreg_turn_bias_weight * strength * ((float(omega) - desired_w) ** 2)
            if keep_course_bias > 1e-3:
                cost += time_weight * self.mpc_colreg_keep_course_weight * keep_course_bias * (float(omega) ** 2)
            if speed_active:
                cost += time_weight * self.mpc_colreg_speed_weight * strength * ((float(v) - desired_v) ** 2)
        return cost

    @staticmethod
    def _safe_float(v, default_v):
        try:
            return float(v)
        except Exception:
            return float(default_v)

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off", "", "none", "null"):
            return False
        return bool(value)

    def _apply_slew_limit(self, target_v, target_w, cset, dt):
        a_v = max(0.02, float(cset.get("max_linear_acc", 0.20)))
        a_w = max(0.05, float(cset.get("max_angular_acc", 0.35)))
        dv = self._clamp(target_v - self._last_cmd_v, -a_v * dt, a_v * dt)
        dw = self._clamp(target_w - self._last_cmd_w, -a_w * dt, a_w * dt)
        v = self._last_cmd_v + dv
        w = self._last_cmd_w + dw
        v = self._clamp(v, 0.0, self.max_speed)
        w = self._clamp(w, -abs(self.max_turn_rate), abs(self.max_turn_rate))
        return v, w

    def _apply_colreg_turn_floor(self, target_w):
        action, weights = self._colreg_constraints()
        if not action or not isinstance(weights, dict):
            return target_w
        course_action = self._canonical_course_action(weights.get("course_action", action))
        if course_action == "KEEP_COURSE":
            return target_w
        min_w = self._clamp(
            abs(float(self.mpc_colreg_min_turn_rate)),
            0.0,
            abs(float(self.max_turn_rate)),
        )
        if min_w <= 1e-6:
            return target_w
        if course_action == "TURN_STARBOARD":
            return min(float(target_w), -min_w)
        if course_action == "TURN_PORT":
            return max(float(target_w), min_w)
        return target_w

    def _maybe_publish_mpc_near_miss_event(self, active_avoid, cmd, now_s):
        if self.planner_mode != "mpc":
            self._mpc_near_miss_active = False
            return
        min_clearance = self._safe_float(getattr(self, "_last_vo_min_clearance", -1.0), -1.0)
        violations = int(getattr(self, "_last_vo_violations", 0) or 0)
        target_count = int(getattr(self, "_last_vo_target_count", 0) or 0)
        proximity = bool(
            target_count > 0
            and (
                violations > 0
                or (
                    math.isfinite(min_clearance)
                    and min_clearance >= 0.0
                    and min_clearance <= max(0.0, float(self.mpc_near_miss_clearance_m))
                )
            )
        )
        detour = bool(
            abs(float(getattr(cmd.angular, "z", 0.0) or 0.0))
            >= max(0.0, float(self.mpc_near_miss_turn_threshold))
            or abs(float(getattr(self, "_last_best_w", 0.0) or 0.0))
            >= max(0.0, float(self.mpc_near_miss_turn_threshold))
        )
        active = bool(active_avoid and proximity and detour)
        if active and not self._mpc_near_miss_active:
            if (now_s - float(self._last_mpc_near_miss_event_t or 0.0)) >= max(
                0.0, float(self.mpc_near_miss_min_interval_s)
            ):
                self._mpc_near_miss_seq += 1
                self._last_mpc_near_miss_event_t = now_s
                event = {
                    "event": "mpc_near_miss_detour",
                    "event_id": int(self._mpc_near_miss_seq),
                    "stamp": float(now_s),
                    "planner_mode": self.planner_mode,
                    "plan_mode": self._plan_mode,
                    "min_clearance_m": float(min_clearance),
                    "violations": int(violations),
                    "target_count": int(target_count),
                    "cmd_linear_x": float(getattr(cmd.linear, "x", 0.0) or 0.0),
                    "cmd_angular_z": float(getattr(cmd.angular, "z", 0.0) or 0.0),
                    "best_w": float(getattr(self, "_last_best_w", 0.0) or 0.0),
                    "clearance_threshold_m": float(self.mpc_near_miss_clearance_m),
                    "turn_threshold_rad_s": float(self.mpc_near_miss_turn_threshold),
                }
                self._mpc_near_miss_pub.publish(String(data=json.dumps(event, ensure_ascii=True)))
                if self.debug_enable:
                    rospy.loginfo(
                        "[collision_traj_planner] MPC near-miss detour event #%d | clear=%.3f viol=%d targets=%d cmd_w=%.3f best_w=%.3f",
                        int(self._mpc_near_miss_seq),
                        float(min_clearance),
                        int(violations),
                        int(target_count),
                        float(getattr(cmd.angular, "z", 0.0) or 0.0),
                        float(getattr(self, "_last_best_w", 0.0) or 0.0),
                    )
        self._mpc_near_miss_active = active

    def _extract_constraint_set(self):
        cset = dict(self._vo_constraint_set())
        constraints = self._llm_decision.get("trajectory_constraints", {}) if isinstance(self._llm_decision, dict) else {}
        if not isinstance(constraints, dict):
            constraints = {}

        action, weights = self._colreg_constraints()
        course_action = self._canonical_course_action(
            constraints.get(
                "course_action",
                self._llm_decision.get("course_action", weights.get("course_action", "")) if isinstance(self._llm_decision, dict) else "",
            )
        )
        speed_action = self._canonical_speed_action(
            constraints.get(
                "speed_action",
                self._llm_decision.get("speed_action", weights.get("speed_action", "")) if isinstance(self._llm_decision, dict) else "",
            )
        )
        if course_action == "KEEP_COURSE":
            speed_action = ""
        neutral_keep_course = bool(course_action == "KEEP_COURSE" and not speed_action)
        if action and isinstance(weights, dict) and not neutral_keep_course:
            speed_scale = self._clamp(self._safe_float(weights.get("speed_scale", 1.0), 1.0), 0.0, 1.5)
            if speed_action == "EMERGENCY_STOP":
                cset["target_linear_x"] = 0.0
                cset["min_linear_x"] = 0.0
                cset["target_angular_z"] = 0.0
            elif speed_action in ("SLOW_DOWN", "SPEED_UP"):
                cset["target_linear_x"] = self._clamp(self.default_speed * speed_scale, 0.0, self.max_speed)
            if speed_action != "EMERGENCY_STOP":
                if course_action == "TURN_STARBOARD":
                    cset["target_angular_z"] = -abs(self.max_turn_rate)
                elif course_action == "TURN_PORT":
                    cset["target_angular_z"] = abs(self.max_turn_rate)
                else:
                    cset["target_angular_z"] = 0.0
            cset["course_action"] = course_action
            cset["speed_action"] = speed_action
            cset["colreg_action"] = action
            cset["colreg_weights"] = dict(weights)

        for key in (
            "duration_s",
            "target_linear_x",
            "target_angular_z",
            "max_linear_acc",
            "max_angular_acc",
            "min_linear_x",
        ):
            if key in constraints:
                cset[key] = constraints[key]
        cset["target_linear_x"] = self._clamp(
            self._safe_float(cset.get("target_linear_x", self.default_speed), self.default_speed),
            0.0,
            self.max_speed,
        )
        cset["target_angular_z"] = self._clamp(
            self._safe_float(cset.get("target_angular_z", 0.0), 0.0),
            -abs(self.max_turn_rate),
            abs(self.max_turn_rate),
        )
        cset["min_linear_x"] = self._clamp(
            self._safe_float(cset.get("min_linear_x", 0.0), 0.0),
            0.0,
            self.max_speed,
        )
        cset["target_linear_x"] = max(float(cset["target_linear_x"]), float(cset["min_linear_x"]))
        return cset

    @staticmethod
    def _anchor_rel_to_world(ax, ay, ayaw, forward_m, lateral_m):
        wx = ax + forward_m * math.cos(ayaw) - lateral_m * math.sin(ayaw)
        wy = ay + forward_m * math.sin(ayaw) + lateral_m * math.cos(ayaw)
        return wx, wy

    def _resample_polyline(self, ctrl_pts, step_m):
        if len(ctrl_pts) <= 1:
            return list(ctrl_pts)
        pts = [ctrl_pts[0]]
        for i in range(len(ctrl_pts) - 1):
            p0 = ctrl_pts[i]
            p1 = ctrl_pts[i + 1]
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            seg = math.hypot(dx, dy)
            if seg < 1e-6:
                continue
            n = max(1, int(seg / max(0.2, step_m)))
            for k in range(1, n + 1):
                t = float(k) / float(n)
                pts.append((p0[0] + t * dx, p0[1] + t * dy))
        return pts

    def _make_path_from_insert_waypoints(self, x, y, yaw, cset):
        if self._avoid_anchor_xyyaw is None:
            self._avoid_anchor_xyyaw = (x, y, yaw)
        ax, ay, ayaw = self._avoid_anchor_xyyaw

        ctrl = [(x, y)]
        for wp in cset.get("insert_waypoints", []):
            wx, wy = self._anchor_rel_to_world(
                ax,
                ay,
                ayaw,
                float(wp.get("forward_m", 0.0)),
                float(wp.get("lateral_m", 0.0)),
            )
            ctrl.append((wx, wy))

        if self.nav_use_goal_in_avoid:
            ctrl.append((self.nav_goal_x, self.nav_goal_y))

        return self._resample_polyline(ctrl, self.waypoint_interp_step_m)

    def _ego_body_speed(self):
        if self._ego is None:
            return float(self._last_cmd_v)
        return float(getattr(self._ego.twist.twist.linear, "x", self._last_cmd_v))

    def _ego_yaw_rate(self):
        if self._ego is None:
            return float(self._last_cmd_w)
        return float(getattr(self._ego.twist.twist.angular, "z", self._last_cmd_w))

    def _simulate_control_profile_trace(self, x0, y0, yaw0, desired_controls, dt, cset):
        pts = [(x0, y0)]
        yaws = [yaw0]
        x = x0
        y = y0
        yaw = yaw0
        cmd_v = float(self._last_cmd_v)
        cmd_w = float(self._last_cmd_w)
        act_v = float(self._ego_body_speed())
        act_w = float(self._ego_yaw_rate())
        cmd_vs = [cmd_v]
        cmd_ws = [cmd_w]
        act_vs = [act_v]
        act_ws = [act_w]

        a_v = max(0.02, float(cset.get("max_linear_acc", 0.20)))
        a_w = max(0.05, float(cset.get("max_angular_acc", 0.35)))
        tau_v = max(0.05, float(self.nav_linear_response_tau_s))
        tau_w = max(0.05, float(self.nav_angular_response_tau_s))
        delay_steps = max(0, int(round(max(0.0, self.nav_response_deadtime_s) / max(1e-3, dt))))
        delayed_cmds = [(cmd_v, cmd_w)] * delay_steps

        for desired_v, desired_w in desired_controls:
            cmd_v += self._clamp(float(desired_v) - cmd_v, -a_v * dt, a_v * dt)
            cmd_w += self._clamp(float(desired_w) - cmd_w, -a_w * dt, a_w * dt)
            cmd_v = self._clamp(cmd_v, 0.0, self.max_speed)
            cmd_w = self._clamp(cmd_w, -abs(self.max_turn_rate), abs(self.max_turn_rate))

            if delay_steps > 0:
                delayed_cmds.append((cmd_v, cmd_w))
                eff_cmd_v, eff_cmd_w = delayed_cmds.pop(0)
            else:
                eff_cmd_v, eff_cmd_w = cmd_v, cmd_w

            act_v += self._clamp(dt / tau_v, 0.0, 1.0) * (eff_cmd_v - act_v)
            act_w += self._clamp(dt / tau_w, 0.0, 1.0) * (eff_cmd_w - act_w)
            x += act_v * math.cos(yaw) * dt
            y += act_v * math.sin(yaw) * dt
            yaw = self._norm_pi(yaw + act_w * dt)
            pts.append((x, y))
            yaws.append(yaw)
            cmd_vs.append(cmd_v)
            cmd_ws.append(cmd_w)
            act_vs.append(act_v)
            act_ws.append(act_w)

        return {
            "pts": pts,
            "yaws": yaws,
            "cmd_vs": cmd_vs,
            "cmd_ws": cmd_ws,
            "act_vs": act_vs,
            "act_ws": act_ws,
        }

    def _simulate_control_profile(self, x0, y0, yaw0, desired_controls, dt, cset):
        trace = self._simulate_control_profile_trace(x0, y0, yaw0, desired_controls, dt, cset)
        return trace["pts"], trace["yaws"][-1], trace["act_vs"][-1], trace["act_ws"][-1]

    def _simulate_constant_control(self, x0, y0, yaw0, v, w, n, dt, cset):
        desired_controls = [(v, w)] * max(1, n)
        pts, yaw, _, _ = self._simulate_control_profile(x0, y0, yaw0, desired_controls, dt, cset)
        return pts, yaw

    def _simulate_two_phase_control(self, x0, y0, yaw0, v, w1, w2, n1, n2, dt, cset):
        desired_controls = ([(v, w1)] * max(1, n1)) + ([(v, w2)] * max(1, n2))
        pts, yaw, _, _ = self._simulate_control_profile(x0, y0, yaw0, desired_controls, dt, cset)
        return pts, yaw

    def _expand_mpc_controls(self, decision, n_steps):
        blocks = max(1, int(len(decision) / 2))
        block_steps = max(1, int(math.ceil(float(max(1, n_steps)) / float(blocks))))
        controls = []
        for i in range(blocks):
            controls.extend([(float(decision[2 * i]), float(decision[2 * i + 1]))] * block_steps)
        return controls[: max(1, n_steps)]

    def _evaluate_mpc_decision(self, decision, x, y, yaw, dt, n_steps, cset, targets):
        controls = self._expand_mpc_controls(decision, n_steps)
        trace = self._simulate_control_profile_trace(x, y, yaw, controls, dt, cset)
        pts = trace["pts"]
        yaw_end = trace["yaws"][-1]
        end = pts[-1]
        goal_dist0 = max(1.0, math.hypot(self.nav_goal_x - x, self.nav_goal_y - y))
        goal_dist = math.hypot(self.nav_goal_x - end[0], self.nav_goal_y - end[1])
        progress = goal_dist0 - goal_dist
        goal_yaw = math.atan2(self.nav_goal_y - y, self.nav_goal_x - x)
        heading_err = abs(self._norm_pi(goal_yaw - yaw_end))
        if self.mpc_geometry_avoidance_enable:
            min_clearance, violations, clearance_cost, deficit_cost = self._min_weighted_clearance_to_targets(
                pts,
                dt,
                targets,
            )
            predictive_risk_cost, _ = self._predictive_target_risk_cost(pts, dt, targets, weighted=True)
        else:
            min_clearance = -1.0
            violations = 0
            clearance_cost = 0.0
            deficit_cost = 0.0
            predictive_risk_cost = 0.0
        _channel_margin, channel_violations, channel_deficit_cost = self._channel_boundary_cost(pts)
        _max_goal_retreat, goal_retreat_cost = self._goal_retreat_cost(pts, x, y)
        smoothness_cost = self._calc_mpc_bspline_smoothness_cost(trace, dt)
        feasibility_cost = self._calc_mpc_bspline_feasibility_cost(trace, dt, cset)
        _colreg_action, colreg_weights = self._colreg_constraints()
        if self.mpc_geometry_avoidance_enable:
            clearance_scale = self._colreg_weight(colreg_weights, "clearance_scale", 1.0, 0.3, 2.5)
            predictive_scale = self._colreg_weight(colreg_weights, "predictive_risk_scale", 1.0, 0.3, 2.5)
            collision_scale = self._colreg_weight(colreg_weights, "collision_penalty_scale", 1.0, 0.3, 2.5)
        else:
            clearance_scale = 0.0
            predictive_scale = 0.0
            collision_scale = 0.0

        cost = 0.0
        cost += self.mpc_goal_weight * goal_dist
        cost -= self.mpc_progress_weight * progress
        cost += self.mpc_heading_weight * heading_err
        cost += self.mpc_smooth_weight * smoothness_cost
        cost += self.mpc_feasibility_weight * feasibility_cost

        first_v = float(decision[0]) if len(decision) >= 2 else self.default_speed
        cost += self.mpc_terminal_speed_weight * ((first_v - self.default_speed) ** 2)
        if self.mpc_geometry_avoidance_enable:
            cost += self.mpc_clearance_weight * clearance_scale * clearance_cost
            cost += self.mpc_collision_penalty * collision_scale * deficit_cost
            cost += self.mpc_predictive_risk_weight * predictive_scale * predictive_risk_cost
        cost += self._colreg_mpc_bias_cost(controls)
        cost += self.goal_retreat_penalty * goal_retreat_cost
        if self.mpc_geometry_avoidance_enable and violations > 0:
            cost += self.mpc_collision_penalty * collision_scale * float(violations)
        if channel_violations > 0:
            cost += self.channel_boundary_penalty * (float(channel_violations) + float(channel_deficit_cost))

        return {
            "cost": cost,
            "pts": pts,
            "yaw_end": yaw_end,
            "controls": controls,
            "min_clearance": min_clearance,
            "violations": int(violations) + int(channel_violations),
            "smoothness_cost": smoothness_cost,
            "feasibility_cost": feasibility_cost,
            "target_count": len(targets),
        }

    def _calc_mpc_bspline_smoothness_cost(self, trace, dt):
        pts = trace.get("pts", [])
        yaws = trace.get("yaws", [])
        if len(pts) < 4:
            return 0.0

        pt_dist = 0.0
        for i in range(len(pts) - 1):
            pt_dist += math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        pt_dist /= float(max(1, len(pts) - 1))
        pt_dist = max(0.2, pt_dist)

        cost = 0.0
        yaw_scale = max(0.2, dt)
        for i in range(len(pts) - 3):
            jerk_x = (pts[i + 3][0] - 3.0 * pts[i + 2][0] + 3.0 * pts[i + 1][0] - pts[i][0]) / pt_dist
            jerk_y = (pts[i + 3][1] - 3.0 * pts[i + 2][1] + 3.0 * pts[i + 1][1] - pts[i][1]) / pt_dist
            cost += jerk_x * jerk_x + jerk_y * jerk_y

            if len(yaws) > i + 3:
                yaw_jerk = self._norm_pi(
                    yaws[i + 3] - 3.0 * yaws[i + 2] + 3.0 * yaws[i + 1] - yaws[i]
                ) / yaw_scale
                cost += self.mpc_yaw_smooth_weight * yaw_jerk * yaw_jerk
        return cost

    def _calc_mpc_bspline_feasibility_cost(self, trace, dt, cset):
        act_vs = trace.get("act_vs", [])
        act_ws = trace.get("act_ws", [])
        if len(act_vs) < 2 or dt <= 1e-4:
            return 0.0

        max_lin_v = self._clamp(self.mpc_max_speed, 0.05, self.max_speed)
        max_lin_a = max(0.02, float(cset.get("max_linear_acc", 0.20)))
        max_ang_w = abs(self.max_turn_rate)
        max_ang_a = max(0.05, float(cset.get("max_angular_acc", 0.35)))

        cost = 0.0
        for v in act_vs[1:]:
            vd = abs(float(v)) - max_lin_v
            if vd > 0.0:
                cost += vd * vd
        for w in act_ws[1:]:
            wd = abs(float(w)) - max_ang_w
            if wd > 0.0:
                cost += self.mpc_turn_weight * wd * wd

        for i in range(len(act_vs) - 1):
            acc_v = (float(act_vs[i + 1]) - float(act_vs[i])) / dt
            ad_v = abs(acc_v) - max_lin_a
            if ad_v > 0.0:
                cost += ad_v * ad_v

            acc_w = (float(act_ws[i + 1]) - float(act_ws[i])) / dt
            ad_w = abs(acc_w) - max_ang_a
            if ad_w > 0.0:
                cost += self.mpc_turn_change_weight * ad_w * ad_w
        return cost

    def _finite_difference_mpc_gradient(self, decision, base_cost, objective_fn, bounds):
        eps_base = max(1e-5, float(self.mpc_nlopt_fd_eps))
        grad = [0.0] * len(decision)
        for i, val in enumerate(decision):
            lo, hi = bounds[i]
            step = eps_base * max(1.0, abs(float(val)))
            x_hi = list(decision)
            x_lo = list(decision)
            x_hi[i] = self._clamp(float(val) + step, lo, hi)
            x_lo[i] = self._clamp(float(val) - step, lo, hi)
            diff = x_hi[i] - x_lo[i]
            if diff < 1e-9:
                grad[i] = 0.0
                continue
            cost_hi = float(objective_fn(x_hi)["cost"])
            cost_lo = float(objective_fn(x_lo)["cost"])
            grad[i] = (cost_hi - cost_lo) / diff
        return grad

    def _candidate_samples(self, target, lo, hi, count):
        n = max(3, count)
        if hi <= lo:
            return [lo]
        vals = []
        for i in range(n):
            t = float(i) / float(max(1, n - 1))
            vals.append(lo + (hi - lo) * t)
        vals.append(self._clamp(target, lo, hi))
        uniq = []
        for v in vals:
            if all(abs(v - u) > 1e-6 for u in uniq):
                uniq.append(v)
        return uniq

    def _decision_confidence(self):
        if not isinstance(self._llm_decision, dict):
            return 0.0
        return self._clamp(
            self._safe_float(self._llm_decision.get("confidence", 0.0), 0.0),
            0.0,
            1.0,
        )

    def _semantic_candidate_targets(self):
        out = []
        raw_targets = self._perception_targets if isinstance(self._perception_targets, list) else []
        for t in raw_targets:
            if not isinstance(t, dict):
                continue
            if bool(t.get("semantic_candidate", True)):
                out.append(t)
        return out

    def _track_classification_map(self):
        out = {}
        if not isinstance(self._llm_decision, dict):
            return out
        raw_items = self._llm_decision.get("track_classifications", [])
        if not isinstance(raw_items, list):
            return out
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            target_id = str(item.get("target_id", "")).strip()
            if not target_id:
                continue
            out[target_id] = {
                "vessel_type": str(item.get("vessel_type", "Unknown")).strip() or "Unknown",
                "association_confidence": self._clamp(
                    self._safe_float(item.get("association_confidence", 0.0), 0.0),
                    0.0,
                    1.0,
                ),
            }
        return out

    def _pending_identification_target_ids(self):
        pending_ids = []
        cls_map = self._track_classification_map()
        for i, target in enumerate(self._semantic_candidate_targets(), start=1):
            if not isinstance(target, dict):
                continue
            target_id = str(target.get("id", "track_%d" % i)).strip()
            cls = cls_map.get(target_id, {})
            vessel_type = str(cls.get("vessel_type", "Unknown")).strip().lower()
            assoc = self._clamp(
                self._safe_float(cls.get("association_confidence", 0.0), 0.0),
                0.0,
                1.0,
            )
            if vessel_type in ("", "unknown") or assoc < self.identification_assoc_confidence_min:
                pending_ids.append(target_id)
        return pending_ids

    def _identification_constraint_set(self):
        return {
            "duration_s": 5.0,
            "target_linear_x": self._clamp(self.identification_speed, 0.0, self.max_speed),
            "target_angular_z": 0.0,
            "max_linear_acc": 0.14,
            "max_angular_acc": 0.22,
            "min_linear_x": 0.0,
        }

    @staticmethod
    def _path_yaw_samples(pts, fallback_yaw):
        if not pts:
            return []
        if len(pts) == 1:
            return [fallback_yaw]
        yaws = []
        for i in range(len(pts)):
            if i < len(pts) - 1:
                dx = pts[i + 1][0] - pts[i][0]
                dy = pts[i + 1][1] - pts[i][1]
            else:
                dx = pts[i][0] - pts[i - 1][0]
                dy = pts[i][1] - pts[i - 1][1]
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                yaws.append(float(fallback_yaw))
            else:
                yaws.append(math.atan2(dy, dx))
        return yaws

    def _identification_fov_cost(self, pts, dt, targets, fallback_yaw):
        if not pts or not targets:
            return 0.0, 0
        yaws = self._path_yaw_samples(pts, fallback_yaw)
        half_fov = math.radians(max(10.0, 0.5 * self.identification_camera_hfov_deg))
        margin = math.radians(max(0.0, self.identification_fov_margin_deg))
        eff_half_fov = max(math.radians(5.0), half_fov - margin)
        total_cost = 0.0
        out_of_fov_count = 0
        pending_ids = set(self._pending_identification_target_ids())
        for target in targets:
            priority = 2.0 if str(target.get("id", "")) in pending_ids else 0.8
            for i, pt in enumerate(pts):
                tt = float(i) * dt
                tx = target["x"] + target["vx"] * tt
                ty = target["y"] + target["vy"] * tt
                bearing = self._norm_pi(math.atan2(ty - pt[1], tx - pt[0]) - yaws[min(i, len(yaws) - 1)])
                abs_bearing = abs(bearing)
                time_weight = 1.0 + max(0.0, (3.0 - tt) / 3.0)
                if abs_bearing > eff_half_fov:
                    exceed = abs_bearing - eff_half_fov
                    total_cost += priority * self.identification_fov_weight * time_weight * ((exceed / eff_half_fov) ** 2)
                    out_of_fov_count += 1
                else:
                    total_cost += priority * self.identification_center_weight * time_weight * ((abs_bearing / eff_half_fov) ** 2)
        return total_cost, out_of_fov_count

    @staticmethod
    def _vessel_profile(vessel_type):
        vessel_key = str(vessel_type or "Unknown").strip().lower()
        profiles = {
            "lifeboat": {"name": "Lifeboat", "radius_scale": 1.35, "safety_scale": 1.45, "penalty_scale": 1.30},
            "fishing": {"name": "Fishing", "radius_scale": 1.28, "safety_scale": 1.40, "penalty_scale": 1.25},
            "usv": {"name": "USV", "radius_scale": 1.12, "safety_scale": 1.18, "penalty_scale": 1.10},
            "smallvessel": {"name": "SmallVessel", "radius_scale": 1.18, "safety_scale": 1.24, "penalty_scale": 1.14},
            "small vessel": {"name": "SmallVessel", "radius_scale": 1.18, "safety_scale": 1.24, "penalty_scale": 1.14},
            "unknown": {"name": "Unknown", "radius_scale": 1.22, "safety_scale": 1.30, "penalty_scale": 1.18},
        }
        return profiles.get(vessel_key, profiles["unknown"])

    def _extract_weighted_nav_targets_world(self, ego_x, ego_y, ego_yaw):
        raw_targets = self._extract_vo_targets_world(ego_x, ego_y, ego_yaw)
        cls_map = self._track_classification_map()
        global_conf = self._decision_confidence()
        colreg_action, colreg_weights = self._colreg_constraints()
        colreg_target_scale = 1.0
        if self.mpc_colreg_enable and self.parallel_nav_semantic_weight_enable:
            colreg_target_scale = self._colreg_weight(
                colreg_weights,
                "target_penalty_scale",
                1.0,
                0.5,
                2.5,
            )
        weighted_targets = []
        for tar in raw_targets:
            is_semantic_candidate = bool(tar.get("semantic_candidate", True))
            cls = cls_map.get(tar.get("id", ""), {}) if is_semantic_candidate else {}
            assoc = self._clamp(
                self._safe_float(cls.get("association_confidence", 0.0), 0.0),
                0.0,
                1.0,
            )
            if self.parallel_nav_semantic_weight_enable and is_semantic_candidate:
                vlm_weight = self._clamp(0.45 * global_conf + 0.55 * assoc, 0.0, 1.0)
            else:
                vlm_weight = 0.0
            if is_semantic_candidate:
                profile = self._vessel_profile(cls.get("vessel_type", "Unknown"))
            else:
                profile = {"name": str(tar.get("obstacle_kind", "static_obstacle")), "radius_scale": 1.0, "safety_scale": 1.0, "penalty_scale": 1.0}
            out = dict(tar)
            out["vessel_type"] = profile["name"]
            out["association_confidence"] = assoc
            out["vlm_weight"] = vlm_weight
            base_radius = self.vo_target_radius_m
            if not is_semantic_candidate:
                geom_radius = 0.5 * math.hypot(
                    max(0.2, float(tar.get("size_x_m", 0.0) or 0.0)),
                    max(0.2, float(tar.get("size_y_m", 0.0) or 0.0)),
                )
                base_radius = max(0.45, geom_radius)
            out["radius_m"] = max(
                0.5,
                base_radius * (1.0 + vlm_weight * (profile["radius_scale"] - 1.0)),
            )
            out["safety_margin_m"] = max(
                0.2,
                self.vo_safety_margin_m * (1.0 + vlm_weight * (profile["safety_scale"] - 1.0)),
            )
            semantic_penalty_scale = 1.0 + vlm_weight * (profile["penalty_scale"] - 1.0)
            out["penalty_scale"] = semantic_penalty_scale * (colreg_target_scale if is_semantic_candidate else 1.0)
            out["colreg_action"] = colreg_action
            out["colreg_target_penalty_scale"] = colreg_target_scale
            weighted_targets.append(out)
        return weighted_targets

    def _min_weighted_clearance_to_targets(self, pts, dt, targets):
        if not pts or not targets:
            return 1e9, 0, 0.0, 0.0
        ego_speed = abs(self._ego_body_speed())
        min_d = 1e9
        violations = 0
        clearance_cost = 0.0
        deficit_cost = 0.0
        for tar in targets:
            local_min = 1e9
            safe_d = max(
                0.5,
                self.vo_ego_radius_m
                + float(tar.get("radius_m", self.vo_target_radius_m))
                + float(tar.get("safety_margin_m", self.vo_safety_margin_m)),
            )
            tar_speed = math.hypot(float(tar.get("vx", 0.0)), float(tar.get("vy", 0.0)))
            latency_buffer = min(
                self.nav_latency_buffer_max_m,
                max(0.0, ego_speed + 0.5 * tar_speed) * max(0.0, self.nav_latency_margin_s),
            )
            safe_d += latency_buffer
            for i, p in enumerate(pts):
                tt = float(i) * dt
                tx = tar["x"] + tar["vx"] * tt
                ty = tar["y"] + tar["vy"] * tt
                d = math.hypot(tx - p[0], ty - p[1])
                if d < local_min:
                    local_min = d
            scale = max(0.5, float(tar.get("penalty_scale", 1.0)))
            clearance_cost += scale / max(0.1, local_min)
            deficit_cost += scale * max(0.0, safe_d - local_min)
            if local_min < min_d:
                min_d = local_min
            if local_min < safe_d:
                violations += 1
        return min_d, violations, clearance_cost, deficit_cost

    def _predictive_target_risk_cost(self, pts, dt, targets, weighted=False):
        if not pts or not targets:
            return 0.0, -1.0
        time_bias_s = max(0.5, float(self.nav_predictive_time_bias_s))
        gate_scale = max(1.0, float(self.nav_predictive_gate_scale))
        ego_speed = abs(self._ego_body_speed())
        total_cost = 0.0
        earliest_risk_t = -1.0
        for tar in targets:
            tar_speed = math.hypot(float(tar.get("vx", 0.0)), float(tar.get("vy", 0.0)))
            latency_buffer = min(
                self.nav_latency_buffer_max_m,
                max(0.0, ego_speed + 0.5 * tar_speed) * max(0.0, self.nav_latency_margin_s),
            )
            base_safe_d = max(
                0.5,
                self.vo_ego_radius_m
                + float(tar.get("radius_m", self.vo_target_radius_m))
                + float(tar.get("safety_margin_m", self.vo_safety_margin_m))
                + latency_buffer,
            )
            predictive_gate = gate_scale * base_safe_d
            target_scale = max(0.5, float(tar.get("penalty_scale", 1.0))) if weighted else 1.0
            for i, p in enumerate(pts):
                tt = float(i) * dt
                tx = tar["x"] + tar["vx"] * tt
                ty = tar["y"] + tar["vy"] * tt
                d = math.hypot(tx - p[0], ty - p[1])
                margin = predictive_gate - d
                if margin <= 0.0:
                    continue
                time_weight = 1.0 + max(0.0, (time_bias_s - tt) / time_bias_s)
                proximity = margin / max(0.1, predictive_gate)
                total_cost += target_scale * time_weight * (proximity ** 2)
                if earliest_risk_t < 0.0 or tt < earliest_risk_t:
                    earliest_risk_t = tt
        return total_cost, earliest_risk_t

    def _filter_goal_rejoin_targets(self, x, y, targets):
        if (not self.mpc_goal_rejoin_enable) or self._trigger:
            return list(targets)
        if not targets:
            return []
        goal_dx = self.nav_goal_x - float(x)
        goal_dy = self.nav_goal_y - float(y)
        goal_dist = math.hypot(goal_dx, goal_dy)
        if goal_dist < 1e-6:
            return []
        gx = goal_dx / goal_dist
        gy = goal_dy / goal_dist
        corridor = max(0.5, float(self.mpc_goal_rejoin_corridor_width_m))
        lookahead = max(2.0, float(self.mpc_goal_rejoin_lookahead_m))
        emergency = max(0.5, float(self.mpc_goal_rejoin_emergency_distance_m))

        out = []
        for tar in targets:
            if str(tar.get("obstacle_kind", "")).startswith("channel_"):
                out.append(tar)
                continue
            tx = float(tar.get("x", x))
            ty = float(tar.get("y", y))
            dx = tx - float(x)
            dy = ty - float(y)
            dist = math.hypot(dx, dy)
            along = dx * gx + dy * gy
            lateral = abs((-gy * dx) + (gx * dy))
            tar_vx = float(tar.get("vx", 0.0))
            tar_vy = float(tar.get("vy", 0.0))
            closing_to_goal_line = (tar_vx * gx + tar_vy * gy) < -0.05

            if dist <= emergency:
                out.append(tar)
            elif (-2.0 <= along <= lookahead) and lateral <= corridor:
                out.append(tar)
            elif closing_to_goal_line and (-4.0 <= along <= lookahead) and lateral <= (1.5 * corridor):
                out.append(tar)
        return out

    def _goal_rejoin_direct_plan(self, x, y, yaw):
        pts = self._make_goal_path(x, y)
        goal_yaw = math.atan2(self.nav_goal_y - y, self.nav_goal_x - x)
        yaw_err = self._norm_pi(goal_yaw - yaw)
        dist = math.hypot(self.nav_goal_x - x, self.nav_goal_y - y)
        speed_scale = self._clamp(float(self.mpc_goal_rejoin_speed_scale), 0.2, 1.2)
        target_v = self._clamp(self.default_speed * speed_scale, 0.05, min(self.max_speed, self.mpc_max_speed))
        if dist < 6.0:
            target_v *= self._clamp(dist / 6.0, 0.20, 1.0)
        target_w = self._clamp(self.mpc_heading_gain * yaw_err, -abs(self.max_turn_rate), abs(self.max_turn_rate))
        return pts, target_v, target_w, -1.0, 1e9, 0, 0

    def _optimize_identification_path(self, x, y, yaw):
        cset = self._identification_constraint_set()
        dt = self._clamp(self.parallel_nav_sample_dt_s, 0.2, 0.6)
        horizon_s = 4.5
        n = max(8, int(horizon_s / dt))
        targets = [
            t for t in self._extract_weighted_nav_targets_world(x, y, yaw)
            if bool(t.get("semantic_candidate", True))
        ]

        if not targets:
            pts = self._make_straight_path(x, y, yaw)
            return pts, 0.0, 0.0, 0.0, 0, 0

        v_samples = self._candidate_samples(
            self.identification_speed,
            0.0,
            self._clamp(self.identification_max_speed, 0.05, self.max_speed),
            max(3, self.opt_speed_samples // 2),
        )
        w_limit = self._clamp(self.identification_turn_rate_limit, 0.05, abs(self.max_turn_rate))
        w_samples = self._candidate_samples(0.0, -w_limit, w_limit, max(7, self.opt_turn_samples // 2))

        goal_dist0 = max(1.0, math.hypot(self.nav_goal_x - x, self.nav_goal_y - y))
        best = None
        best_cost = 1e18
        best_v = 0.0
        best_w = 0.0
        best_out_of_fov = 0

        for v in v_samples:
            for w in w_samples:
                pts, yaw_end = self._simulate_constant_control(x, y, yaw, v, w, n, dt, cset)
                end = pts[-1]
                goal_dist = math.hypot(self.nav_goal_x - end[0], self.nav_goal_y - end[1])
                progress = goal_dist0 - goal_dist
                fov_cost, out_of_fov_count = self._identification_fov_cost(pts, dt, targets, yaw)
                predictive_risk_cost, _ = self._predictive_target_risk_cost(pts, dt, targets, weighted=True)
                _channel_margin, channel_violations, channel_deficit_cost = self._channel_boundary_cost(pts)
                _max_goal_retreat, goal_retreat_cost = self._goal_retreat_cost(pts, x, y)

                cost = 0.0
                cost += fov_cost
                cost += self.identification_risk_weight * predictive_risk_cost
                cost += self.goal_retreat_penalty * goal_retreat_cost
                cost += 0.8 * ((v - self._last_cmd_v) ** 2)
                cost += 0.6 * ((w - self._last_cmd_w) ** 2)
                cost += 0.2 * (w ** 2)
                cost -= self.identification_progress_weight * progress
                cost += 0.4 * float(out_of_fov_count)
                cost += 0.15 * abs(self._norm_pi(yaw_end - yaw))
                if channel_violations > 0:
                    cost += self.channel_boundary_penalty * (float(channel_violations) + float(channel_deficit_cost))

                if cost < best_cost:
                    best_cost = cost
                    best = pts
                    best_v = v
                    best_w = w
                    best_out_of_fov = out_of_fov_count

        if best is None:
            pts = self._make_straight_path(x, y, yaw)
            return pts, 0.0, 0.0, 1e9, len(self._pending_identification_target_ids()), 0

        return best, best_v, best_w, best_cost, len(self._pending_identification_target_ids()), best_out_of_fov

    def _compute_parallel_nav_blend_alpha(self, cset, target_count):
        return self._clamp(float(self.parallel_nav_blend_alpha), 0.15, 0.90)

    @staticmethod
    def _interp_path_ratio(path, ratio):
        if not path:
            return None
        if len(path) == 1:
            return path[0]
        u = max(0.0, min(1.0, ratio)) * float(len(path) - 1)
        i = int(math.floor(u))
        if i >= len(path) - 1:
            return path[-1]
        t = u - float(i)
        p0 = path[i]
        p1 = path[i + 1]
        return (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))

    def _blend_paths(self, base_path, nav_path, alpha):
        if not base_path:
            return list(nav_path)
        if not nav_path:
            return list(base_path)
        if alpha <= 1e-3:
            return list(base_path)
        if alpha >= 1.0 - 1e-3:
            return list(nav_path)
        sample_count = max(len(base_path), len(nav_path))
        pts = []
        for i in range(sample_count):
            ratio = float(i) / float(max(1, sample_count - 1))
            p_base = self._interp_path_ratio(base_path, ratio)
            p_nav = self._interp_path_ratio(nav_path, ratio)
            pts.append(
                (
                    (1.0 - alpha) * p_base[0] + alpha * p_nav[0],
                    (1.0 - alpha) * p_base[1] + alpha * p_nav[1],
                )
            )
        return pts

    def _optimize_parallel_nav_path(self, x, y, yaw, cset):
        return self._optimize_vo_path(x, y, yaw, cset=cset, weighted=True)

    def _vo_constraint_set(self):
        return {
            "duration_s": self._clamp(self.vo_time_horizon_s, 3.0, 20.0),
            "target_linear_x": self._clamp(self.default_speed, 0.05, self.max_speed),
            "target_angular_z": 0.0,
            "max_linear_acc": 0.30,
            "max_angular_acc": 0.45,
            "min_linear_x": 0.0,
        }

    def _mpc_constraint_set(self):
        return {
            "duration_s": self._clamp(self.mpc_horizon_s, 3.0, 20.0),
            "target_linear_x": self._clamp(self.default_speed, 0.05, self.max_speed),
            "target_angular_z": 0.0,
            "max_linear_acc": 0.26,
            "max_angular_acc": 0.38,
            "min_linear_x": 0.03,
        }

    @staticmethod
    def _body_to_world(vx_body, vy_body, yaw):
        vx = vx_body * math.cos(yaw) - vy_body * math.sin(yaw)
        vy = vx_body * math.sin(yaw) + vy_body * math.cos(yaw)
        return vx, vy

    def _ego_world_velocity(self, yaw):
        if self._ego is None:
            return 0.0, 0.0
        tw = self._ego.twist.twist
        vx_body = float(getattr(tw.linear, "x", 0.0))
        vy_body = float(getattr(tw.linear, "y", 0.0))
        return self._body_to_world(vx_body, vy_body, yaw)

    def _channel_constraint_points(self):
        if not self.channel_constraints_enable:
            return []
        x_lo = min(self.channel_x_min, self.channel_x_max)
        x_hi = max(self.channel_x_min, self.channel_x_max)
        step = max(0.5, abs(self.channel_x_step))
        xs = []
        cur = x_lo
        while cur <= (x_hi + 1e-6):
            xs.append(cur)
            cur += step
        if not xs or abs(xs[-1] - x_hi) > 1e-6:
            xs.append(x_hi)

        out = []
        for idx, x in enumerate(xs, start=1):
            out.append(
                {
                    "id": "channel_left_%d" % idx,
                    "x": float(x),
                    "y": float(self.channel_left_y),
                    "vx": 0.0,
                    "vy": 0.0,
                    "range_m": math.hypot(float(x), float(self.channel_left_y)),
                    "obstacle_kind": "channel_constraint",
                    "semantic_candidate": False,
                    "size_x_m": 2.0 * float(self.channel_obstacle_radius_m),
                    "size_y_m": 2.0 * float(self.channel_obstacle_radius_m),
                }
            )
            out.append(
                {
                    "id": "channel_right_%d" % idx,
                    "x": float(x),
                    "y": float(self.channel_right_y),
                    "vx": 0.0,
                    "vy": 0.0,
                    "range_m": math.hypot(float(x), float(self.channel_right_y)),
                    "obstacle_kind": "channel_constraint",
                    "semantic_candidate": False,
                    "size_x_m": 2.0 * float(self.channel_obstacle_radius_m),
                    "size_y_m": 2.0 * float(self.channel_obstacle_radius_m),
                }
            )
        return out

    def _channel_boundary_cost(self, pts):
        if not self.channel_constraints_enable or not pts:
            return 1e9, 0, 0.0
        x_lo = min(self.channel_x_min, self.channel_x_max)
        x_hi = max(self.channel_x_min, self.channel_x_max)
        y_lo = min(self.channel_left_y, self.channel_right_y) + float(self.channel_boundary_margin_m)
        y_hi = max(self.channel_left_y, self.channel_right_y) - float(self.channel_boundary_margin_m)
        if y_hi <= y_lo:
            mid_y = 0.5 * (min(self.channel_left_y, self.channel_right_y) + max(self.channel_left_y, self.channel_right_y))
            y_lo = mid_y - 0.1
            y_hi = mid_y + 0.1

        min_margin = 1e9
        violations = 0
        deficit_cost = 0.0
        for px, py in pts:
            if px < (x_lo - 1e-6) or px > (x_hi + 1e-6):
                continue
            margin = min(py - y_lo, y_hi - py)
            if margin < min_margin:
                min_margin = margin
            if margin < 0.0:
                violations += 1
                deficit_cost += -margin
        return min_margin, violations, deficit_cost

    def _goal_retreat_cost(self, pts, start_x, start_y):
        if not pts:
            return 0.0, 0.0
        goal_dx = self.nav_goal_x - float(start_x)
        goal_dy = self.nav_goal_y - float(start_y)
        goal_norm = math.hypot(goal_dx, goal_dy)
        if goal_norm < 1e-6:
            return 0.0, 0.0
        goal_ux = goal_dx / goal_norm
        goal_uy = goal_dy / goal_norm
        margin = max(0.0, float(self.goal_retreat_margin_m))
        max_retreat = 0.0
        retreat_cost = 0.0
        for px, py in pts:
            along = (float(px) - float(start_x)) * goal_ux + (float(py) - float(start_y)) * goal_uy
            retreat = max(0.0, -(along + margin))
            if retreat > max_retreat:
                max_retreat = retreat
            retreat_cost += retreat
        return max_retreat, retreat_cost

    def _extract_vo_targets_world(self, ego_x, ego_y, ego_yaw):
        out = []
        ego_vx_w, ego_vy_w = self._ego_world_velocity(ego_yaw)
        raw_targets = self._perception_targets if isinstance(self._perception_targets, list) else []
        for t in raw_targets:
            if not isinstance(t, dict):
                continue
            try:
                rel_x = float(t.get("rel_x", 0.0))
                rel_y = float(t.get("rel_y", 0.0))
                rel_vx = float(t.get("rel_vx", 0.0))
                rel_vy = float(t.get("rel_vy", 0.0))
            except Exception:
                continue

            tar_x = ego_x + rel_x * math.cos(ego_yaw) - rel_y * math.sin(ego_yaw)
            tar_y = ego_y + rel_x * math.sin(ego_yaw) + rel_y * math.cos(ego_yaw)
            rel_vx_w, rel_vy_w = self._body_to_world(rel_vx, rel_vy, ego_yaw)
            tar_vx = ego_vx_w + rel_vx_w
            tar_vy = ego_vy_w + rel_vy_w

            out.append(
                {
                    "id": str(t.get("id", "unknown")),
                    "x": tar_x,
                    "y": tar_y,
                    "vx": tar_vx,
                    "vy": tar_vy,
                    "obstacle_kind": str(t.get("obstacle_kind", "vessel_candidate")),
                    "semantic_candidate": bool(t.get("semantic_candidate", True)),
                    "size_x_m": float(t.get("size_x_m", 0.0) or 0.0),
                    "size_y_m": float(t.get("size_y_m", 0.0) or 0.0),
                    "range_m": float(t.get("range_m", math.hypot(rel_x, rel_y))),
                    "rel_x": rel_x,
                    "rel_y": rel_y,
                }
            )
        for t in self._channel_constraint_points():
            if not isinstance(t, dict):
                continue
            item = dict(t)
            item["range_m"] = math.hypot(float(item["x"]) - ego_x, float(item["y"]) - ego_y)
            out.append(item)
        return out

    def _min_clearance_to_targets(self, pts, dt, targets):
        if not pts or not targets:
            return 1e9, 0
        safe_d = max(0.5, self.vo_ego_radius_m + self.vo_target_radius_m + self.vo_safety_margin_m)
        ego_speed = abs(self._ego_body_speed())
        min_d = 1e9
        violations = 0
        for tar in targets:
            local_min = 1e9
            tar_speed = math.hypot(float(tar.get("vx", 0.0)), float(tar.get("vy", 0.0)))
            latency_buffer = min(
                self.nav_latency_buffer_max_m,
                max(0.0, ego_speed + 0.5 * tar_speed) * max(0.0, self.nav_latency_margin_s),
            )
            safe_d_eff = safe_d + latency_buffer
            for i, p in enumerate(pts):
                tt = float(i) * dt
                tx = tar["x"] + tar["vx"] * tt
                ty = tar["y"] + tar["vy"] * tt
                d = math.hypot(tx - p[0], ty - p[1])
                if d < local_min:
                    local_min = d
            if local_min < min_d:
                min_d = local_min
            if local_min < safe_d_eff:
                violations += 1
        return min_d, violations

    # ------------------------------------------------------------------
    # Pure geometric VO helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _vo_cone_check(ego_x, ego_y, ego_vx, ego_vy, tar, combined_r, tau):
        """Check whether velocity (ego_vx, ego_vy) lies inside the VO cone of *tar*.

        Returns (in_vo: bool, penetration: float in [0, 1]).
        Geometry: VO_tau(A, B) = { v_A | exists t in [0,tau]:
            |p_B - p_A + (v_B - v_A)*t| < combined_r }.

        Args:
            ego_x, ego_y:  ego world position
            ego_vx, ego_vy: candidate ego world velocity
            tar:           target dict with keys x, y, vx, vy
            combined_r:    r_ego + r_obstacle + safety_margin
            tau:           time horizon [s]
        """
        dx = float(tar["x"]) - ego_x
        dy = float(tar["y"]) - ego_y
        r2 = combined_r * combined_r
        dist2 = dx * dx + dy * dy

        # Relative velocity of ego w.r.t. obstacle: w = v_A - v_B
        wx = ego_vx - float(tar.get("vx", 0.0))
        wy = ego_vy - float(tar.get("vy", 0.0))
        speed2 = wx * wx + wy * wy

        if dist2 < r2:
            return True, 1.0  # already inside combined footprint

        if speed2 < 1e-12:
            return False, 0.0  # no relative motion

        # Dot product p_rel . w  (p_rel = obstacle centre w.r.t. ego)
        dot = dx * wx + dy * wy
        if dot <= 0.0:
            return False, 0.0  # diverging; no future collision

        # Perpendicular distance from the velocity ray to the obstacle centre
        # cross = |p_rel x w| (scalar 2-D)
        cross = dx * wy - dy * wx
        min_dist2 = cross * cross / speed2

        if min_dist2 >= r2:
            return False, 0.0  # ray misses obstacle

        # Collision interval [t_entry, t_exit] via quadratic formula
        disc = dot * dot - speed2 * (dist2 - r2)
        if disc < 0.0:
            return False, 0.0
        sqrt_disc = math.sqrt(disc)
        t_entry = (dot - sqrt_disc) / speed2
        t_exit = (dot + sqrt_disc) / speed2

        # Overlap with [0, tau]?
        if t_exit < 0.0 or t_entry > tau:
            return False, 0.0

        penetration = (combined_r - math.sqrt(max(0.0, min_dist2))) / max(0.1, combined_r)
        return True, max(0.0, penetration)

    def _sim_arc(self, x0, y0, yaw0, v, w, n, dt):
        """Simple kinematic arc simulation (no slew-rate / delay model)."""
        pts = [(x0, y0)]
        x, y, yaw = x0, y0, yaw0
        for _ in range(n):
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            yaw = self._norm_pi(yaw + w * dt)
            pts.append((x, y))
        return pts

    def _pure_geometric_vo_path(self, x, y, yaw):
        """Pure geometric Velocity Obstacle velocity selection.

        Algorithm
        ---------
        1. For each candidate (v, omega):
           a. Compute instantaneous world velocity v_A = (v*cos(yaw), v*sin(yaw)).
           b. Check v_A geometrically against the VO cone of every obstacle.
           c. Score: -progress + heading_err + smoothness + VO_penalty.
        2. Select the lowest-cost (v, omega).
        3. Generate a short kinematic arc for pure-pursuit path following.

        No bspline, no n-step dynamic simulation for scoring.

        Returns (pts, best_v, best_w, best_cost, min_clearance, violations, n_targets).
        """
        tau = max(1.0, float(self.vo_time_horizon_s))
        dt_vis = max(0.2, float(self.vo_sample_dt_s))
        n_vis = max(20, int(tau / dt_vis))

        raw_targets = self._extract_vo_targets_world(x, y, yaw)

        # Build (target, combined_r) pairs
        targets_r = []
        for tar in raw_targets:
            if not tar.get("semantic_candidate", True):
                geom_r = 0.5 * math.hypot(
                    max(0.2, float(tar.get("size_x_m", 0.0) or 0.0)),
                    max(0.2, float(tar.get("size_y_m", 0.0) or 0.0)),
                )
                base_r = max(0.45, geom_r)
            else:
                base_r = self.vo_target_radius_m
            combined_r = max(0.5, self.vo_ego_radius_m + base_r + self.vo_safety_margin_m)
            targets_r.append((tar, combined_r))

        goal_dist = max(1.0, math.hypot(self.nav_goal_x - x, self.nav_goal_y - y))
        goal_yaw = math.atan2(self.nav_goal_y - y, self.nav_goal_x - x)

        v_lo = 0.0
        v_hi = self.max_speed
        w_lo = -abs(self.max_turn_rate)
        w_hi = abs(self.max_turn_rate)

        v_samples = self._candidate_samples(self.default_speed, v_lo, v_hi, self.opt_speed_samples)
        w_samples = self._candidate_samples(0.0, w_lo, w_hi, self.opt_turn_samples)

        # Shorter horizon for VO cone check: large tau makes forbidden set too large
        # and allows even distant slow-closing obstacles to block all v>0 choices.
        tau_check = min(tau, 4.0)
        # Cap heading/progress evaluation horizon: beyond 5 s, w*tau wraps
        # past 2*pi and makes heading_err non-monotone, rewarding large omega.
        tau_eval = min(tau, 5.0)
        # Effective collision penalty: scaled so that a full-penetration violation
        # costs ~4x the maximum reachable progress reward, not 1000x.
        eff_penalty = max(30.0, self.vo_collision_penalty * 0.04)
        # Bounded clearance band: replaces 1/clearance which blows up and
        # makes stopping look better than manoeuvring.
        max_cls = max(0.5, self.vo_safety_margin_m + 0.3)

        best_v = self.default_speed
        best_w = 0.0
        best_cost = 1e18
        best_violations = 0
        best_clearance = -1.0

        for v in v_samples:
            for w in w_samples:
                # --- Progress/heading cost: capped evaluation horizon ---
                if abs(w) > 1e-4:
                    final_yaw = yaw + w * tau_eval
                    end_x = x + (v / w) * (math.sin(final_yaw) - math.sin(yaw))
                    end_y = y + (v / w) * (math.cos(yaw) - math.cos(final_yaw))
                else:
                    end_x = x + v * tau_eval * math.cos(yaw)
                    end_y = y + v * tau_eval * math.sin(yaw)

                goal_dist_end = math.hypot(self.nav_goal_x - end_x, self.nav_goal_y - end_y)
                progress = goal_dist - goal_dist_end
                final_yaw_norm = self._norm_pi(yaw + w * tau_eval)
                heading_err = abs(self._norm_pi(goal_yaw - final_yaw_norm))

                # --- Geometric VO cone checks ---
                # Use 0.25*tau_check so the mid-arc heading stays within ~18°
                # of current heading at max turn rate (0.4 rad/s * 1s = 0.4 rad),
                # preventing large omega from escaping VO cones via heading wrap.
                mid_yaw = self._norm_pi(yaw + w * tau_check * 0.25)
                vx_a = v * math.cos(mid_yaw)
                vy_a = v * math.sin(mid_yaw)

                total_vo_penalty = 0.0
                violations = 0
                min_clearance = 1e9
                for tar, r in targets_r:
                    in_vo, penetration = self._vo_cone_check(
                        x, y, vx_a, vy_a, tar, r, tau_check
                    )
                    if in_vo:
                        violations += 1
                        total_vo_penalty += penetration
                    else:
                        dx_ = float(tar["x"]) - x
                        dy_ = float(tar["y"]) - y
                        clearance = math.hypot(dx_, dy_) - r
                        if clearance < min_clearance:
                            min_clearance = clearance

                # --- Composite cost ---
                cost = 0.0
                cost -= self.vo_progress_weight * progress
                cost += self.vo_heading_weight * heading_err
                cost += 0.8 * ((v - self._last_cmd_v) ** 2)
                # Stronger angular rate penalty: discourages large |w| even
                # when last_cmd_w == 0, and adds a direct |w|^2 preference
                # for straight-line motion to prevent unnecessary spin.
                cost += 1.2 * ((w - self._last_cmd_w) ** 2)
                cost += 0.6 * (w ** 2)
                # Penalise speeds below cruising: prevents v=0 from being a
                # "free pass" (static obstacles give speed2≈0 → no VO hit at v=0).
                cost += 1.5 * max(0.0, self.default_speed - v)
                if violations > 0:
                    cost += eff_penalty * total_vo_penalty
                elif min_clearance < 1e8:
                    # Bounded linear clearance cost: does not blow up for small gaps
                    cost += self.vo_clearance_weight * max(0.0, max_cls - min_clearance)

                if cost < best_cost:
                    best_cost = cost
                    best_v = v
                    best_w = w
                    best_violations = violations
                    best_clearance = -1.0 if violations > 0 else min_clearance

        # Generate visualization path (simple kinematic arc, no bspline)
        best_pts = self._sim_arc(x, y, yaw, best_v, best_w, n_vis, dt_vis)
        return best_pts, best_v, best_w, best_cost, best_clearance, best_violations, len(raw_targets)

    def _semantic_geometric_vo_path(self, x, y, yaw):
        """Semantic Velocity Obstacle velocity selection (pure geometric, no bspline).

        Extends ``_pure_geometric_vo_path`` with semantic-aware collision-body
        inflation.  Each obstacle's combined radius is derived from
        ``_extract_weighted_nav_targets_world`` which scales ego + obstacle
        radius and safety margin according to vessel type and VLM/association
        confidence.  A per-obstacle ``penalty_scale`` further weights the VO
        penetration cost so high-risk vessel types are avoided more aggressively.

        Algorithm
        ---------
        1. Retrieve semantically-weighted targets (radius_m, safety_margin_m,
           penalty_scale already computed via ``_vessel_profile``).
        2. For each candidate (v, omega):
           a. Compute instantaneous world velocity v_A = (v*cos(yaw), v*sin(yaw)).
           b. Check v_A against the VO cone of every obstacle using the
              semantically-inflated combined_r.
           c. Score: -progress + heading_err + smoothness
                     + penalty_scale * VO_penetration.
        3. Select the lowest-cost (v, omega).
        4. Generate a short kinematic arc via ``_sim_arc`` (no bspline).

        Returns (pts, best_v, best_w, best_cost, min_clearance, violations, n_targets).
        """
        tau = max(1.0, float(self.vo_time_horizon_s))
        dt_vis = max(0.2, float(self.vo_sample_dt_s))
        n_vis = max(20, int(tau / dt_vis))

        weighted_targets = self._extract_weighted_nav_targets_world(x, y, yaw)

        # Update mean VLM weight for diagnostics
        if weighted_targets:
            self._last_parallel_nav_mean_vlm_weight = sum(
                float(t.get("vlm_weight", 0.0)) for t in weighted_targets
            ) / len(weighted_targets)
        else:
            self._last_parallel_nav_mean_vlm_weight = 0.0

        # Build (target, combined_r, penalty_scale) triples
        targets_r = []
        for tar in weighted_targets:
            combined_r = max(
                0.5,
                self.vo_ego_radius_m
                + float(tar.get("radius_m", self.vo_target_radius_m))
                + float(tar.get("safety_margin_m", self.vo_safety_margin_m)),
            )
            p_scale = max(0.5, float(tar.get("penalty_scale", 1.0)))
            targets_r.append((tar, combined_r, p_scale))

        goal_dist = max(1.0, math.hypot(self.nav_goal_x - x, self.nav_goal_y - y))
        goal_yaw = math.atan2(self.nav_goal_y - y, self.nav_goal_x - x)

        v_lo = 0.0
        v_hi = self.max_speed
        w_lo = -abs(self.max_turn_rate)
        w_hi = abs(self.max_turn_rate)

        v_samples = self._candidate_samples(self.default_speed, v_lo, v_hi, self.opt_speed_samples)
        w_samples = self._candidate_samples(0.0, w_lo, w_hi, self.opt_turn_samples)

        tau_check = min(tau, 4.0)
        tau_eval = min(tau, 5.0)
        eff_penalty = max(30.0, self.vo_collision_penalty * 0.04)
        max_cls = max(0.5, self.vo_safety_margin_m + 0.3)

        best_v = self.default_speed
        best_w = 0.0
        best_cost = 1e18
        best_violations = 0
        best_clearance = -1.0

        for v in v_samples:
            for w in w_samples:
                # --- Progress/heading cost: capped evaluation horizon ---
                if abs(w) > 1e-4:
                    final_yaw = yaw + w * tau_eval
                    end_x = x + (v / w) * (math.sin(final_yaw) - math.sin(yaw))
                    end_y = y + (v / w) * (math.cos(yaw) - math.cos(final_yaw))
                else:
                    end_x = x + v * tau_eval * math.cos(yaw)
                    end_y = y + v * tau_eval * math.sin(yaw)

                goal_dist_end = math.hypot(self.nav_goal_x - end_x, self.nav_goal_y - end_y)
                progress = goal_dist - goal_dist_end
                final_yaw_norm = self._norm_pi(yaw + w * tau_eval)
                heading_err = abs(self._norm_pi(goal_yaw - final_yaw_norm))

                # --- Semantic geometric VO cone checks ---
                # Mid-arc heading makes omega-dependent: turning into a clear corridor
                # exits the VO cone and costs much less than stopping in place.
                mid_yaw = self._norm_pi(yaw + w * tau_check * 0.25)
                vx_a = v * math.cos(mid_yaw)
                vy_a = v * math.sin(mid_yaw)

                total_vo_penalty = 0.0
                violations = 0
                min_clearance = 1e9
                for tar, r, p_scale in targets_r:
                    in_vo, penetration = self._vo_cone_check(
                        x, y, vx_a, vy_a, tar, r, tau_check
                    )
                    if in_vo:
                        violations += 1
                        total_vo_penalty += p_scale * penetration
                    else:
                        dx_ = float(tar["x"]) - x
                        dy_ = float(tar["y"]) - y
                        clearance = math.hypot(dx_, dy_) - r
                        if clearance < min_clearance:
                            min_clearance = clearance

                # --- Composite cost ---
                cost = 0.0
                cost -= self.vo_progress_weight * progress
                cost += self.vo_heading_weight * heading_err
                cost += 0.8 * ((v - self._last_cmd_v) ** 2)
                cost += 1.2 * ((w - self._last_cmd_w) ** 2)
                cost += 0.6 * (w ** 2)
                cost += 1.5 * max(0.0, self.default_speed - v)
                if violations > 0:
                    cost += eff_penalty * total_vo_penalty
                elif min_clearance < 1e8:
                    cost += self.vo_clearance_weight * max(0.0, max_cls - min_clearance)

                if cost < best_cost:
                    best_cost = cost
                    best_v = v
                    best_w = w
                    best_violations = violations
                    best_clearance = -1.0 if violations > 0 else min_clearance

        # Generate visualization path (simple kinematic arc, no bspline)
        best_pts = self._sim_arc(x, y, yaw, best_v, best_w, n_vis, dt_vis)
        return best_pts, best_v, best_w, best_cost, best_clearance, best_violations, len(weighted_targets)

    def _optimize_vo_path(self, x, y, yaw, cset=None, weighted=False):
        dt = max(0.2, min(1.0, self.vo_sample_dt_s))
        horizon_s = self._clamp(self.vo_time_horizon_s, 3.0, 20.0)
        n = max(8, int(horizon_s / dt))
        if cset is None:
            cset = self._vo_constraint_set()
        if weighted:
            targets = self._extract_weighted_nav_targets_world(x, y, yaw)
            if targets:
                self._last_parallel_nav_mean_vlm_weight = sum(
                    float(t.get("vlm_weight", 0.0)) for t in targets
                ) / float(len(targets))
            else:
                self._last_parallel_nav_mean_vlm_weight = 0.0
        else:
            targets = self._extract_vo_targets_world(x, y, yaw)
            self._last_parallel_nav_mean_vlm_weight = 0.0

        v_lo = self._clamp(self._safe_float(cset.get("min_linear_x", 0.0), 0.0), 0.0, self.max_speed)
        preferred_speed = self._clamp(
            self._safe_float(cset.get("target_linear_x", self.default_speed), self.default_speed),
            v_lo,
            self.max_speed,
        )
        v_hi = self._clamp(max(preferred_speed, self.default_speed + 0.35), 0.2, self.max_speed)
        w_lo = -abs(self.max_turn_rate)
        w_hi = abs(self.max_turn_rate)
        preferred_turn = self._clamp(
            self._safe_float(cset.get("target_angular_z", 0.0), 0.0),
            w_lo,
            w_hi,
        )
        colreg_weights = cset.get("colreg_weights", {})
        if not isinstance(colreg_weights, dict):
            colreg_weights = {}
        colreg_strength = self._clamp(
            self._safe_float(colreg_weights.get("strength", 1.0), 1.0),
            0.0,
            2.5,
        )
        colreg_speed_scale = self._clamp(
            self._safe_float(colreg_weights.get("speed_scale", 1.0), 1.0),
            0.0,
            1.5,
        )

        v_samples = self._candidate_samples(preferred_speed, v_lo, v_hi, self.opt_speed_samples)
        w_samples = self._candidate_samples(preferred_turn, w_lo, w_hi, self.opt_turn_samples)

        best = None
        best_cost = 1e18
        best_v = 0.0
        best_w = 0.0
        best_min_clearance = -1.0
        best_violations = 0

        goal_dist0 = max(1.0, math.hypot(self.nav_goal_x - x, self.nav_goal_y - y))
        goal_yaw = math.atan2(self.nav_goal_y - y, self.nav_goal_x - x)
        safe_d = max(0.5, self.vo_ego_radius_m + self.vo_target_radius_m + self.vo_safety_margin_m)

        for v in v_samples:
            for w in w_samples:
                pts, yaw_end = self._simulate_constant_control(x, y, yaw, v, w, n, dt, cset)
                end = pts[-1]
                goal_dist = math.hypot(self.nav_goal_x - end[0], self.nav_goal_y - end[1])
                progress = goal_dist0 - goal_dist
                heading_err = abs(self._norm_pi(goal_yaw - yaw_end))

                if weighted:
                    min_clearance, violations, clearance_cost, deficit_cost = self._min_weighted_clearance_to_targets(
                        pts,
                        dt,
                        targets,
                    )
                    predictive_risk_cost, _ = self._predictive_target_risk_cost(pts, dt, targets, weighted=True)
                else:
                    min_clearance, violations = self._min_clearance_to_targets(pts, dt, targets)
                    predictive_risk_cost, _ = self._predictive_target_risk_cost(pts, dt, targets, weighted=False)
                _channel_margin, channel_violations, channel_deficit_cost = self._channel_boundary_cost(pts)
                _max_goal_retreat, goal_retreat_cost = self._goal_retreat_cost(pts, x, y)

                cost = 0.0
                cost -= self.vo_progress_weight * progress
                cost += self.vo_heading_weight * heading_err
                cost += 1.6 * ((v - self._last_cmd_v) ** 2)
                cost += 1.3 * ((w - self._last_cmd_w) ** 2)
                cost += 2.4 * max(0.6, colreg_strength) * ((w - preferred_turn) ** 2)
                if abs(colreg_speed_scale - 1.0) > 0.02 or preferred_speed <= self.default_speed * 0.8:
                    cost += 2.0 * max(0.6, colreg_strength) * ((v - preferred_speed) ** 2)
                cost += self.vo_predictive_risk_weight * predictive_risk_cost
                cost += self.goal_retreat_penalty * goal_retreat_cost

                if weighted:
                    cost += self.vo_clearance_weight * clearance_cost
                    cost += self.vo_collision_penalty * deficit_cost
                elif min_clearance < 1e8:
                    clearance_deficit = max(0.0, safe_d - min_clearance)
                    cost += self.vo_clearance_weight / max(0.1, min_clearance)
                    cost += self.vo_collision_penalty * clearance_deficit
                if violations > 0:
                    cost += self.vo_collision_penalty * float(violations)
                if channel_violations > 0:
                    cost += self.channel_boundary_penalty * (float(channel_violations) + float(channel_deficit_cost))

                if cost < best_cost:
                    best_cost = cost
                    best = pts
                    best_v = v
                    best_w = w
                    best_min_clearance = min_clearance
                    best_violations = int(violations) + int(channel_violations)

        if best is None:
            pts = self._make_straight_path(x, y, yaw)
            return pts, max(0.05, self.default_speed), 0.0, 1e9, -1.0, 0, len(targets)

        return best, best_v, best_w, best_cost, best_min_clearance, best_violations, len(targets)

    def _optimize_mpc_path(self, x, y, yaw):
        self._mpc_goal_rejoin_active = False
        cset = self._mpc_constraint_set()
        dt = self._clamp(self.mpc_sample_dt_s, 0.2, 0.8)
        horizon_s = self._clamp(self.mpc_horizon_s, 3.0, 20.0)
        n = max(8, int(horizon_s / dt))
        if not self.mpc_geometry_avoidance_enable:
            targets = []
            self._last_parallel_nav_mean_vlm_weight = 0.0
        elif self.parallel_nav_semantic_weight_enable:
            targets = self._extract_weighted_nav_targets_world(x, y, yaw)
            if targets:
                self._last_parallel_nav_mean_vlm_weight = sum(
                    float(t.get("vlm_weight", 0.0)) for t in targets
                ) / float(len(targets))
            else:
                self._last_parallel_nav_mean_vlm_weight = 0.0
        else:
            raw_targets = self._extract_vo_targets_world(x, y, yaw)
            targets = []
            for tar in raw_targets:
                out = dict(tar)
                out["radius_m"] = self.vo_target_radius_m
                out["safety_margin_m"] = self.vo_safety_margin_m
                out["penalty_scale"] = 1.0
                targets.append(out)
            self._last_parallel_nav_mean_vlm_weight = 0.0

        behavior_active = bool(
            isinstance(self._llm_decision, dict)
            and self._llm_decision.get("behavior_active", False)
        )
        if self.mpc_geometry_avoidance_enable and self.mpc_goal_rejoin_enable and not self._trigger and not behavior_active:
            targets = self._filter_goal_rejoin_targets(x, y, targets)
            dynamic_targets = [
                t
                for t in targets
                if not str(t.get("obstacle_kind", "")).startswith("channel_")
            ]
            if not dynamic_targets:
                self._mpc_goal_rejoin_active = True
                pts, v, w, cost, min_clearance, violations, _target_count = self._goal_rejoin_direct_plan(x, y, yaw)
                return pts, v, w, cost, min_clearance, violations, 0

        v_lo = self._clamp(float(cset.get("min_linear_x", 0.03)), 0.0, self.max_speed)
        v_hi = self._clamp(min(self.max_speed, self.mpc_max_speed), v_lo, self.max_speed)
        v_ref = self._clamp(float(cset.get("target_linear_x", self.default_speed)), v_lo, v_hi)
        w_lo = -abs(self.max_turn_rate)
        w_hi = abs(self.max_turn_rate)
        block_count = max(1, min(int(self.mpc_control_blocks), n))

        seed_v = self._clamp(max(v_lo, float(self._last_cmd_v)), v_lo, v_hi)
        if seed_v < v_lo + 1e-4:
            seed_v = v_ref
        seed_w = self._clamp(float(self._last_cmd_w), w_lo, w_hi)

        x0_decision = []
        bounds = []
        for _ in range(block_count):
            x0_decision.extend([seed_v, seed_w])
            bounds.append((v_lo, v_hi))
            bounds.append((w_lo, w_hi))

        def _objective_eval(decision):
            return self._evaluate_mpc_decision(decision, x, y, yaw, dt, n, cset, targets)

        initial_eval = _objective_eval(x0_decision)
        best_eval = initial_eval
        best_decision = list(x0_decision)

        try:
            opt = nlopt.opt(int(self.mpc_nlopt_algorithm), len(x0_decision))
            opt.set_lower_bounds([b[0] for b in bounds])
            opt.set_upper_bounds([b[1] for b in bounds])
            opt.set_maxeval(max(10, int(self.mpc_solver_maxiter)))
            opt.set_xtol_rel(max(1e-6, float(self.mpc_solver_ftol)))
            opt.set_maxtime(max(0.0, float(self.mpc_nlopt_maxtime_s)))

            def _nlopt_objective(x_var, grad):
                nonlocal best_eval, best_decision
                decision = [float(v) for v in x_var]
                cur_eval = _objective_eval(decision)
                cur_cost = float(cur_eval["cost"])
                if math.isfinite(cur_cost) and cur_cost < float(best_eval["cost"]):
                    best_eval = cur_eval
                    best_decision = list(decision)
                if grad.size > 0:
                    fd_grad = self._finite_difference_mpc_gradient(decision, cur_cost, _objective_eval, bounds)
                    for i, val in enumerate(fd_grad):
                        grad[i] = float(val)
                return cur_cost

            opt.set_min_objective(_nlopt_objective)
            result_x = opt.optimize(x0_decision)
            if result_x is not None and len(result_x) == len(x0_decision):
                cand_eval = _objective_eval(result_x)
                if math.isfinite(float(cand_eval["cost"])) and float(cand_eval["cost"]) < float(best_eval["cost"]):
                    best_eval = cand_eval
                    best_decision = [float(v) for v in result_x]
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0,
                "[collision_traj_planner] MPC NLopt solver exception: %s",
                str(exc),
            )

        controls = best_eval["controls"]
        if not controls:
            pts = self._make_straight_path(x, y, yaw)
            return pts, max(0.05, self.default_speed), 0.0, 1e9, -1.0, 0, len(targets)

        first_v, first_w = controls[0]
        return (
            best_eval["pts"],
            first_v,
            first_w,
            float(best_eval["cost"]),
            float(best_eval["min_clearance"]),
            int(best_eval["violations"]),
            int(best_eval["target_count"]),
        )

    def _optimize_path(self, x, y, yaw, cset):
        pts, best_v, best_w, best_cost, _min_clearance, _violations, _target_count = self._optimize_vo_path(
            x,
            y,
            yaw,
            cset=cset,
            weighted=True,
        )
        return pts, best_v, best_w, best_cost

    def run(self):
        rate = rospy.Rate(max(1.0, self.publish_rate))
        while not rospy.is_shutdown():
            if self._ego is None:
                rate.sleep()
                continue

            x = float(self._ego.pose.pose.position.x)
            y = float(self._ego.pose.pose.position.y)
            yaw = self._yaw_from_quat(self._ego.pose.pose.orientation)
            frame_id = self.path_frame or (self._ego.header.frame_id or "world")

            now_s = rospy.Time.now().to_sec()
            dt_cmd = now_s - self._last_cmd_t if self._last_cmd_t > 0.0 else max(0.1, self.opt_dt_s)
            dt_cmd = max(0.05, min(1.0, dt_cmd))

            is_vo_mode = self.planner_mode == "vo"
            is_mpc_mode = self.planner_mode == "mpc"
            hold_left = max(0.0, self._constraint_hold_until_t - now_s)
            behavior_active = bool(
                isinstance(self._llm_decision, dict)
                and self._llm_decision.get("behavior_active", False)
            )
            force_replan_due = (
                self._decision_force_replan_seq >= 0
                and self._decision_force_replan_seq != self._last_applied_force_replan_seq
            )
            decision_changed = (
                bool(self._decision_signature)
                and self._decision_signature != self._last_decision_signature
            )
            if is_mpc_mode:
                if not self._started:
                    active_avoid = False
                else:
                    active_avoid = True
            elif is_vo_mode:
                active_avoid = bool(self._trigger or behavior_active)
            else:
                active_avoid = bool(self._trigger or behavior_active) or (hold_left > 0.0)
            must_replan = (
                (not self._planned_xy)
                or (active_avoid != self._last_trigger_state)
                or force_replan_due
                or decision_changed
                or ((now_s - self._last_replan_t) >= max(0.1, self.opt_replan_interval_s))
            )

            cmd = Twist()
            if active_avoid:
                if is_vo_mode:
                    cset = self._vo_constraint_set()
                    if must_replan:
                        (
                            self._planned_xy,
                            best_v,
                            best_w,
                            best_cost,
                            min_clearance,
                            violations,
                            target_count,
                        ) = self._pure_geometric_vo_path(x, y, yaw)
                        self._plan_mode = "vo"
                        self._last_plan_cost = best_cost
                        self._last_best_v = best_v
                        self._last_best_w = best_w
                        self._last_vo_min_clearance = min_clearance
                        self._last_vo_violations = int(violations)
                        self._last_vo_target_count = int(target_count)
                        self._last_replan_t = now_s
                        self._last_applied_force_replan_seq = self._decision_force_replan_seq
                        self._last_decision_signature = self._decision_signature
                        self._last_parallel_nav_min_clearance = -1.0
                        self._last_parallel_nav_target_count = 0
                        self._last_parallel_nav_cost = 0.0
                        self._last_parallel_nav_blend = 0.0
                        self._last_parallel_nav_mean_vlm_weight = 0.0
                    else:
                        self._last_vo_target_count = len(self._perception_targets)
                    cset = self._vo_constraint_set()
                elif is_mpc_mode:
                    cset = self._mpc_constraint_set()
                    if must_replan:
                        (
                            self._planned_xy,
                            best_v,
                            best_w,
                            best_cost,
                            min_clearance,
                            violations,
                            target_count,
                        ) = self._optimize_mpc_path(x, y, yaw)
                        if self._mpc_goal_rejoin_active:
                            self._plan_mode = "goal_rejoin"
                        elif not self.mpc_geometry_avoidance_enable:
                            self._plan_mode = "mpc_action_only"
                        else:
                            self._plan_mode = "semantic_mpc" if self.parallel_nav_semantic_weight_enable else "mpc"
                        self._last_plan_cost = best_cost
                        self._last_best_v = best_v
                        self._last_best_w = best_w
                        self._last_vo_min_clearance = min_clearance
                        self._last_vo_violations = int(violations)
                        self._last_vo_target_count = int(target_count)
                        self._last_replan_t = now_s
                        self._last_applied_force_replan_seq = self._decision_force_replan_seq
                        self._last_decision_signature = self._decision_signature
                        self._last_parallel_nav_min_clearance = -1.0
                        self._last_parallel_nav_target_count = 0
                        self._last_parallel_nav_cost = 0.0
                        self._last_parallel_nav_blend = 0.0
                        self._last_parallel_nav_mean_vlm_weight = 0.0
                    else:
                        self._last_vo_target_count = 0 if not self.mpc_geometry_avoidance_enable else len(self._perception_targets)
                else:
                    needs_identification = bool(self._trigger) and self.identification_mode_enable and bool(self._pending_identification_target_ids())
                    if self._trigger and not needs_identification:
                        latest_cset = self._extract_constraint_set()
                        self._enter_constraint_hold(latest_cset, now_s)
                        if self._avoid_anchor_xyyaw is None:
                            self._avoid_anchor_xyyaw = (x, y, yaw)
                    cset = self._identification_constraint_set() if needs_identification else self._active_constraint_set(now_s)

                    if must_replan:
                        if needs_identification:
                            (
                                self._planned_xy,
                                self._last_best_v,
                                self._last_best_w,
                                self._last_plan_cost,
                                self._last_identification_pending_count,
                                self._last_identification_out_of_fov_count,
                            ) = self._optimize_identification_path(x, y, yaw)
                            self._plan_mode = "identify_tracks"
                            self._last_parallel_nav_min_clearance = -1.0
                            self._last_parallel_nav_target_count = 0
                            self._last_parallel_nav_cost = 0.0
                            self._last_parallel_nav_blend = 0.0
                        else:
                            self._last_identification_pending_count = 0
                            self._last_identification_out_of_fov_count = 0
                            (
                                self._planned_xy,
                                self._last_best_v,
                                self._last_best_w,
                                self._last_plan_cost,
                                self._last_vo_min_clearance,
                                self._last_vo_violations,
                                self._last_vo_target_count,
                            ) = self._semantic_geometric_vo_path(x, y, yaw)
                            self._plan_mode = "semantic_vo"
                            self._last_parallel_nav_min_clearance = -1.0
                            self._last_parallel_nav_target_count = 0
                            self._last_parallel_nav_cost = 0.0
                            self._last_parallel_nav_blend = 0.0
                        self._last_replan_t = now_s
                        self._last_applied_force_replan_seq = self._decision_force_replan_seq
                        self._last_decision_signature = self._decision_signature

                if is_mpc_mode:
                    target_v = min(getattr(self, "_last_best_v", self.default_speed), self.mpc_max_speed)
                    tgt = self._select_lookahead(x, y, lookahead_m=self.mpc_lookahead_m)
                    path_heading_w = 0.0
                    if tgt is not None:
                        des_yaw = math.atan2(tgt[1] - y, tgt[0] - x)
                        yaw_err = self._norm_pi(des_yaw - yaw)
                        path_heading_w = self.mpc_heading_gain * yaw_err
                    target_w = self._clamp(
                        self.mpc_tracking_blend * getattr(self, "_last_best_w", 0.0)
                        + (1.0 - self.mpc_tracking_blend) * path_heading_w,
                        -abs(self.max_turn_rate),
                        abs(self.max_turn_rate),
                    )
                    target_w = self._apply_colreg_turn_floor(target_w)
                else:
                    target_v = getattr(self, "_last_best_v", self.default_speed)
                    tgt = self._select_lookahead(x, y)
                    if tgt is not None:
                        des_yaw = math.atan2(tgt[1] - y, tgt[0] - x)
                        yaw_err = self._norm_pi(des_yaw - yaw)
                        target_w = self._clamp(self.kp_heading * yaw_err, -abs(self.max_turn_rate), abs(self.max_turn_rate))
                    else:
                        target_w = getattr(self, "_last_best_w", 0.0)
                cmd.linear.x, cmd.angular.z = self._apply_slew_limit(target_v, target_w, cset, dt_cmd)
                self._last_cmd_v = cmd.linear.x
                self._last_cmd_w = cmd.angular.z
                self._maybe_publish_mpc_near_miss_event(active_avoid, cmd, now_s)
            else:
                if must_replan:
                    self._planned_xy = self._make_straight_path(x, y, yaw)
                    self._plan_mode = "straight"
                    self._last_replan_t = now_s
                    self._last_applied_force_replan_seq = self._decision_force_replan_seq
                    self._last_decision_signature = self._decision_signature
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self._last_cmd_v = 0.0
                self._last_cmd_w = 0.0
                self._last_best_v = 0.0
                self._last_best_w = 0.0
                self._avoid_anchor_xyyaw = None
                self._last_vo_min_clearance = -1.0
                self._last_vo_violations = 0
                self._last_vo_target_count = 0
                self._last_parallel_nav_min_clearance = -1.0
                self._last_parallel_nav_target_count = 0
                self._last_parallel_nav_cost = 0.0
                self._last_parallel_nav_blend = 0.0
                self._last_parallel_nav_mean_vlm_weight = 0.0
                self._last_identification_pending_count = 0
                self._last_identification_out_of_fov_count = 0
                self._maybe_publish_mpc_near_miss_event(active_avoid, cmd, now_s)

            self._last_trigger_state = active_avoid
            self._last_cmd_t = now_s

            self._publish_path(self._planned_xy, frame_id)

            cur_wp = self._select_lookahead(x, y)
            if cur_wp is not None:
                wp_yaw = math.atan2(cur_wp[1] - y, cur_wp[0] - x)
                self._publish_current_waypoint(cur_wp, frame_id, wp_yaw)

            self._cmd_pub.publish(cmd)

            if self.debug_enable:
                colreg_action, colreg_weights = self._colreg_constraints()
                course_action = colreg_weights.get("course_action", "") if isinstance(colreg_weights, dict) else ""
                speed_action = colreg_weights.get("speed_action", "") if isinstance(colreg_weights, dict) else ""
                rospy.loginfo_throttle(
                    2.0,
                    "[collision_traj_planner] planner=%s trigger=%s active_avoid=%s hold_left=%.2fs mode=%s course_action=%s speed_action=%s colreg_action=%s path_pts=%d cmd(v=%.3f,w=%.3f) opt_cost=%.3f vo_targets=%d vo_min_clear=%.3f vo_viol=%d nav_targets=%d nav_min_clear=%.3f nav_blend=%.2f nav_sem=%s nav_mean_vlm_w=%.2f identify_pending=%d identify_oof=%d",
                    self.planner_mode,
                    str(self._trigger),
                    str(active_avoid),
                    0.0 if is_vo_mode else hold_left,
                    self._plan_mode,
                    course_action or "-",
                    speed_action or "-",
                    colreg_action or "-",
                    len(self._planned_xy),
                    cmd.linear.x,
                    cmd.angular.z,
                    self._last_plan_cost,
                    int(self._last_vo_target_count),
                    float(self._last_vo_min_clearance),
                    int(self._last_vo_violations),
                    int(self._last_parallel_nav_target_count),
                    float(self._last_parallel_nav_min_clearance),
                    float(self._last_parallel_nav_blend),
                    str(self.parallel_nav_semantic_weight_enable),
                    float(self._last_parallel_nav_mean_vlm_weight),
                    int(self._last_identification_pending_count),
                    int(self._last_identification_out_of_fov_count),
                )
            rate.sleep()


if __name__ == "__main__":
    CollisionTrajectoryPlannerNode().run()
