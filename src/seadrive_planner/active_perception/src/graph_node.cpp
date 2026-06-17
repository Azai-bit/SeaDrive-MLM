#include <active_perception/graph_node.h>
#include <path_searching/astar2.h>
#include <plan_env/sdf_map.h>
#include <plan_env/raycast.h>

namespace fast_planner {
// Static data
double ViewNode::vm_;
double ViewNode::am_;
double ViewNode::yd_;
double ViewNode::ydd_;
double ViewNode::w_dir_;
double ViewNode::w_hist_ = 0.5;  // 访问同一 viewpoint 的历史代价权重（适中偏小）
shared_ptr<Astar> ViewNode::astar_;
shared_ptr<RayCaster> ViewNode::caster_;
shared_ptr<SDFMap> ViewNode::map_;
unordered_map<int, int> ViewNode::visit_counts_;

// Graph node for viewpoints planning
ViewNode::ViewNode(const Vector3d& p, const double& y) {
  pos_ = p;
  yaw_ = y;
  parent_ = nullptr;
  vel_.setZero();  // vel is zero by default, should be set explicitly
}

double ViewNode::costTo(const ViewNode::Ptr& node) {
  vector<Vector3d> path;
  double base_cost = ViewNode::computeCost(pos_, node->pos_, yaw_, node->yaw_, vel_, yaw_dot_, path);

  // 对多次访问同一 viewpoint 增加历史代价，访问次数越多，代价越大，从而减少在局部来回“打转”。
  int count = 0;
  auto it = visit_counts_.find(node->id_);
  if (it != visit_counts_.end()) count = it->second;
  double hist_cost = w_hist_ * static_cast<double>(count);

  return base_cost + hist_cost;
}

double ViewNode::searchPath(const Vector3d& p1, const Vector3d& p2, vector<Vector3d>& path) {
  // 仅考虑 xy 平面上的连通性：将起终点在 z 轴上投影到同一高度（保持与 p1 的 z 相同），
  // 并在使用前将它们投影/裁剪回 sdf_map 的 box 内，避免目标点落在 ESDF/box 外导致
  // “Lower bound not satisfied” 一类的下界错误。
  Vector3d q1 = p1;
  Vector3d q2 = p2;
  q2.z() = q1.z();

  // 将起终点裁剪到 box 内（按世界坐标 clamp 到 box_min/box_max 范围）
  Vector3d box_min, box_max;
  map_->getBox(box_min, box_max);
  auto clampToBox = [&](const Vector3d& p) {
    Vector3d q = p;
    for (int i = 0; i < 3; ++i) {
      if (q[i] < box_min[i]) q[i] = box_min[i] + 1e-3;
      if (q[i] > box_max[i]) q[i] = box_max[i] - 1e-3;
    }
    return q;
  };
  q1 = clampToBox(q1);
  q2 = clampToBox(q2);

  // Try connect two points with straight line (in xy-plane)
  bool safe = true;
  Vector3i idx;
  //caster_->input(q1, q2);
  caster_->input(p1, p2);

  while (caster_->nextId(idx)) {
    if (map_->getInflateOccupancy(idx) == 1 || map_->getOccupancy(idx) == SDFMap::UNKNOWN ||
        !map_->isInBox(idx)) {
      safe = false;
      break;
    }
  }
  if (safe) {
    // path = { q1, q2 };
    // return (q1.head<2>() - q2.head<2>()).norm();
    path = { p1, p2 };
    return (p1 - p2).norm();
  }
  // Search a path using decreasing resolution
  vector<double> res = { 0.4 };
  for (int k = 0; k < res.size(); ++k) {
    astar_->reset();
    astar_->setResolution(res[k]);
    //if (astar_->search(q1, q2) == Astar::REACH_END) {
    if (astar_->search(p1, p2) == Astar::REACH_END) {

      path = astar_->getPath();
      return astar_->pathLength(path);
    }
  }
  // Use Astar early termination cost as an estimate
  path = { q1, q2 };
  return 1000;
}

void ViewNode::addVisit(const ViewNode::Ptr& node) {
  if (!node) return;
  ++visit_counts_[node->id_];
}

void ViewNode::addVisits(const vector<ViewNode::Ptr>& nodes) {
  for (const auto& n : nodes) {
    addVisit(n);
  }
}

double ViewNode::computeCost(const Vector3d& p1, const Vector3d& p2, const double& y1, const double& y2,
                             const Vector3d& v1, const double& yd1, vector<Vector3d>& path) {
  // Cost of position change
  double pos_cost = ViewNode::searchPath(p1, p2, path) / vm_;

  // Consider velocity change
  if (v1.norm() > 1e-3) {
    Vector3d dir = (p2 - p1).normalized();
    Vector3d vdir = v1.normalized();
    double diff = acos(vdir.dot(dir));
    pos_cost += w_dir_ * diff;
    // double vc = v1.dot(dir);
    // pos_cost += w_dir_ * pow(vm_ - fabs(vc), 2) / (2 * vm_ * am_);
    // if (vc < 0)
    //   pos_cost += w_dir_ * 2 * fabs(vc) / am_;
  }

  // Cost of yaw change
  double diff = fabs(y2 - y1);
  diff = min(diff, 2 * M_PI - diff);
  double yaw_cost = diff / yd_;
  return max(pos_cost, yaw_cost);

  // // Consider yaw rate change
  // if (fabs(yd1) > 1e-3)
  // {
  //   double diff1 = y2 - y1;
  //   while (diff1 < -M_PI)
  //     diff1 += 2 * M_PI;
  //   while (diff1 > M_PI)
  //     diff1 -= 2 * M_PI;
  //   double diff2 = diff1 > 0 ? diff1 - 2 * M_PI : 2 * M_PI + diff1;
  // }
  // else
  // {
  // }
}
}