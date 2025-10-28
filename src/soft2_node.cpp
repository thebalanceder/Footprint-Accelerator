// src/soft2_vo/src/soft2_node.cpp
#include "soft2_vo/soft2_node.hpp"
#include "soft2_vo/optimize_module.hpp"
#include <rclcpp/time.hpp>
#include <iomanip>
#include <algorithm>
#include <cmath>

/* ============================================================= */
/*                     SAFE ITERATION HELPER                    */
/* ============================================================= */
inline int safe_max_iter(int base, int fallback, int processed, double total_lat) {
  if (processed <= 0) return fallback;
  double avg = total_lat / processed;
  if (!std::isfinite(avg) || avg <= 0.0) return fallback;
  double denom = avg + 1.0;
  if (denom <= 0.0) return fallback;
  int val = static_cast<int>(base / denom);
  return std::max(1, val);
}

/* ============================================================= */
/*                     COST FUNCTORS                             */
/* ============================================================= */
struct StereoCost {
  StereoCost(double u_l, double v_l, double u_r, double v_r,
             double baseline, double focal, double cx, double cy)
      : u_l_(u_l), v_l_(v_l), u_r_(u_r), v_r_(v_r),
        baseline_(baseline), focal_(focal), cx_(cx), cy_(cy) {}

  template <typename T>
  bool operator()(const T* const t, const T* const q, T* residuals) const {
    T X = (T(u_l_) - T(cx_)) * T(baseline_) / (T(u_l_) - T(u_r_));
    T Y = (T(v_l_) - T(cy_)) * T(baseline_) / (T(u_l_) - T(u_r_));
    T Z = T(focal_) * T(baseline_) / (T(u_l_) - T(u_r_));

    T p[3] = {X, Y, Z};
    T p_cam[3];
    ceres::QuaternionRotatePoint(q, p, p_cam);
    p_cam[0] += t[0];
    p_cam[1] += t[1];
    p_cam[2] += t[2];

    T xp = p_cam[0] / p_cam[2];
    T yp = p_cam[1] / p_cam[2];
    T proj_u = T(focal_) * xp + T(cx_);
    T proj_v = T(focal_) * yp + T(cy_);

    residuals[0] = proj_u - T(u_r_);
    residuals[1] = proj_v - T(v_r_);
    return true;
  }

private:
  double u_l_, v_l_, u_r_, v_r_, baseline_, focal_, cx_, cy_;
};

struct TemporalCost {
  TemporalCost(const cv::Point2f& pt_prev, const cv::Point2f& pt_curr,
               double focal, double cx, double cy, double weight)
      : pt_prev_(pt_prev), pt_curr_(pt_curr), focal_(focal), cx_(cx), cy_(cy), weight_(weight) {}

  template <typename T>
  bool operator()(const T* const prev_t, const T* const prev_q,
                  const T* const curr_t, const T* const curr_q,
                  T* residuals) const {
    T X = (T(pt_prev_.x) - T(cx_)) / T(focal_);
    T Y = (T(pt_prev_.y) - T(cy_)) / T(focal_);
    T p_prev[3] = {X, Y, T(1.0)};

    T p_cam_prev[3];
    ceres::QuaternionRotatePoint(prev_q, p_prev, p_cam_prev);
    p_cam_prev[0] += prev_t[0];
    p_cam_prev[1] += prev_t[1];
    p_cam_prev[2] += prev_t[2];

    T inv_q_prev[4] = {prev_q[0], -prev_q[1], -prev_q[2], -prev_q[3]};
    T p_world[3];
    ceres::QuaternionRotatePoint(inv_q_prev, p_cam_prev, p_world);

    T p_cam_curr[3];
    ceres::QuaternionRotatePoint(curr_q, p_world, p_cam_curr);
    p_cam_curr[0] += curr_t[0];
    p_cam_curr[1] += curr_t[1];
    p_cam_curr[2] += curr_t[2];

    T rel_t[3], rel_q[4];
    T diff_t[3] = {curr_t[0] - prev_t[0], curr_t[1] - prev_t[1], curr_t[2] - prev_t[2]};
    ceres::QuaternionRotatePoint(inv_q_prev, diff_t, rel_t);
    T inv_q_curr[4] = {curr_q[0], -curr_q[1], -curr_q[2], -curr_q[3]};
    ceres::QuaternionProduct(inv_q_prev, curr_q, rel_q);

    T p_prev_norm[3] = {X, Y, T(1.0)};
    T epipolar_line[3];
    ComputeEpipolarLine(rel_t, rel_q, p_prev_norm, epipolar_line);

    T u_curr = T(pt_curr_.x);
    T v_curr = T(pt_curr_.y);
    T dist = (epipolar_line[0] * u_curr + epipolar_line[1] * v_curr + epipolar_line[2]) /
             ceres::sqrt(epipolar_line[0] * epipolar_line[0] + epipolar_line[1] * epipolar_line[1]);

    residuals[0] = dist * T(weight_);
    residuals[1] = T(0.0);
    return true;
  }

private:
  cv::Point2f pt_prev_, pt_curr_;
  double focal_, cx_, cy_, weight_;
};

/* ============================================================= */
/*                         NODE IMPLEMENTATION                  */
/* ============================================================= */
Soft2Node::Soft2Node() : Node("soft2_node") {
  declare_parameter("baseline", 0.54);
  declare_parameter("max_keypoints", 3000);
  declare_parameter("clear_trajectory", true);
  declare_parameter("min_inliers", 10);
  baseline_ = get_parameter("baseline").as_double();
  max_keypoints_ = get_parameter("max_keypoints").as_int();
  bool clear_traj = get_parameter("clear_trajectory").as_bool();
  min_inliers_ = get_parameter("min_inliers").as_int();

  orb_ = cv::ORB::create(max_keypoints_);
  matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, true);  // cross-check

  auto qos = rclcpp::QoS(rclcpp::KeepLast(1000))
                 .reliable()
                 .durability_volatile();
  sub_left_sync_.subscribe(this, "/camera/left/image_raw", qos.get_rmw_qos_profile());
  sub_right_sync_.subscribe(this, "/camera/right/image_raw", qos.get_rmw_qos_profile());

  sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(1000), sub_left_sync_, sub_right_sync_);
  sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.5));
  sync_->registerCallback(std::bind(&Soft2Node::syncedCallback, this,
                                    std::placeholders::_1, std::placeholders::_2));

  pub_odom_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
  timer_ = this->create_wall_timer(std::chrono::milliseconds(100),
                                   std::bind(&Soft2Node::timerCallback, this));

  // Clear file on start
  if (clear_traj) {
    std::ofstream clear_file("trajectory.txt", std::ios::out | std::ios::trunc);
    clear_file.close();
  }
  traj_file_.open("trajectory.txt", std::ios::out | std::ios::app);
  if (!traj_file_.is_open())
    RCLCPP_ERROR(get_logger(), "Failed to open trajectory.txt");

  processed_frames_ = 0;
  total_latency_ = 0.0;
}

Soft2Node::~Soft2Node() {
  if (traj_file_.is_open()) traj_file_.close();
}

/* ------------------------------------------------------------------ */
void Soft2Node::syncedCallback(const ImageMsg::ConstSharedPtr& left,
                               const ImageMsg::ConstSharedPtr& right) {
  auto start = std::chrono::high_resolution_clock::now();

  // ---- 1. Debug sync timing ---------------------------------------------
  rclcpp::Time left_time(left->header.stamp);
  rclcpp::Time right_time(right->header.stamp);
  double stamp_diff_ms = std::abs((left_time - right_time).nanoseconds()) / 1e6;

  RCLCPP_INFO(get_logger(), "SYNCED: KITTI frame %s (idx %d) | stamp diff: %.3f ms",
              left->header.frame_id.c_str(), frame_id_, stamp_diff_ms);

  // ---- 2. ROS to OpenCV ------------------------------------------------
  cv_bridge::CvImageConstPtr cv_left  = cv_bridge::toCvShare(left,  "mono8");
  cv_bridge::CvImageConstPtr cv_right = cv_bridge::toCvShare(right, "mono8");

  // ---- 3. 4x4 grid mask (detection only) -------------------------------
  static cv::Mat grid_mask;
  if (grid_mask.empty()) {
    grid_mask = cv::Mat::zeros(cv_left->image.size(), CV_8UC1);
    const int gs = 4;
    const int cw = cv_left->image.cols  / gs;
    const int ch = cv_left->image.rows  / gs;
    for (int i = 0; i < gs; ++i)
      for (int j = 0; j < gs; ++j)
        grid_mask(cv::Rect(j * cw, i * ch, cw, ch)).setTo(255);
  }

  // ---- 4. ORB detection + description ----------------------------------
  orb_->detectAndCompute(cv_left->image,  grid_mask, kp_left_,  desc_left_);
  orb_->detectAndCompute(cv_right->image, grid_mask, kp_right_, desc_right_);

  // ---- 5. Keep only the best N keypoints --------------------------------
  if (kp_left_.size() > static_cast<size_t>(max_keypoints_)) {
    cv::KeyPointsFilter::retainBest(kp_left_, max_keypoints_);
    desc_left_ = desc_left_(cv::Range(0, static_cast<int>(kp_left_.size())), cv::Range::all()).clone();
  }
  if (kp_right_.size() > static_cast<size_t>(max_keypoints_)) {
    cv::KeyPointsFilter::retainBest(kp_right_, max_keypoints_);
    desc_right_ = desc_right_(cv::Range(0, static_cast<int>(kp_right_.size())), cv::Range::all()).clone();
  }

  // ---- 6. Safety checks -------------------------------------------------
  assert(desc_left_.rows  == static_cast<int>(kp_left_.size()));
  assert(desc_right_.rows == static_cast<int>(kp_right_.size()));
  assert(desc_left_.type()  == CV_8U && desc_right_.type() == CV_8U);
  assert(desc_left_.isContinuous() && desc_right_.isContinuous());

  // ---- 7. Matching (cross-check) ----------------------------------------
  std::vector<cv::DMatch> cross_matches;
  matcher_->match(desc_left_, desc_right_, cross_matches);

  good_matches_.clear();
  for (const auto& m : cross_matches) {
    const cv::Point2f &pl = kp_left_[m.queryIdx].pt;
    const cv::Point2f &pr = kp_right_[m.trainIdx].pt;
    if (isValidMatch(pl, pr))
      good_matches_.push_back(m);
  }

  // Use frame_id_ as KITTI frame index (0 to 270)
  int kitti_frame = frame_id_;
  RCLCPP_INFO(get_logger(),
              "KITTI Frame %06d (idx %d): %zu good matches (need >=5)",
              kitti_frame, frame_id_, good_matches_.size());

  // ---- 8. REPEAT POSE HELPER (used for all failures) -------------------
  auto repeat_previous_pose = [&]() -> void {
    std::vector<double> pose = (frame_id_ > 0 && frame_poses_.count(frame_id_ - 1))
        ? frame_poses_.at(frame_id_ - 1)
        : std::vector<double>{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};

    frame_poses_[frame_id_] = pose;
    frame_stamps_[frame_id_] = left->header.stamp;
    frame_kps_[frame_id_]   = kp_left_;
    writePoseToFile(pose);

    ++processed_frames_;
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    if (ms < 0.0) ms = 0.0;
    total_latency_ += ms;

    RCLCPP_INFO(get_logger(),
                "KITTI Frame %06d (idx %d): Pose repeated | This: %.2f ms | Avg: %.2f ms",
                frame_id_, frame_id_, ms, total_latency_ / processed_frames_);

    ++frame_id_;
  };

  // ---- 9. If too few matches --------------------------------------------
  if (good_matches_.size() < 5) {
    RCLCPP_WARN(get_logger(), "KITTI Frame %06d: Too few matches (%zu), repeating previous pose.",
                frame_id_, good_matches_.size());
    repeat_previous_pose();
    return;
  }

  // ---- 10. Build point vectors for essential matrix --------------------
  std::vector<cv::Point2f> pts_l, pts_r;
  for (const auto& m : good_matches_) {
    pts_l.push_back(kp_left_[m.queryIdx].pt);
    pts_r.push_back(kp_right_[m.trainIdx].pt);
  }

  // ---- 11. Essential matrix --------------------------------------------
  cv::Mat E = cv::findEssentialMat(pts_l, pts_r, focal_,
                                   cv::Point2d(cx_, cy_),
                                   cv::RANSAC, 0.999, 1.0);
  if (E.empty()) {
    RCLCPP_WARN(get_logger(), "KITTI Frame %06d: findEssentialMat failed, repeating previous pose",
                frame_id_);
    repeat_previous_pose();
    return;
  }

  // ---- 12. Recover pose -------------------------------------------------
  cv::Mat R, t;
  int inliers = cv::recoverPose(E, pts_l, pts_r, R, t, focal_, cv::Point2d(cx_, cy_));

  if (inliers < min_inliers_) {
    RCLCPP_WARN(get_logger(), "KITTI Frame %06d: recoverPose failed: %d inliers < %d, repeating previous pose",
                frame_id_, inliers, min_inliers_);
    repeat_previous_pose();
    return;
  }

  // ---- 13. Convert to quaternion ----------------------------------------
  double q[4];
  ceres::RotationMatrixToQuaternion(R.ptr<double>(), q);

  opt_t_[0] = t.at<double>(0);
  opt_t_[1] = t.at<double>(1);
  opt_t_[2] = t.at<double>(2);
  opt_q_[0] = q[0];
  opt_q_[1] = q[1];
  opt_q_[2] = q[2];
  opt_q_[3] = q[3];

  // ---- 14. Store pose ---------------------------------------------------
  std::vector<double> pose(7);
  pose[0] = opt_t_[0]; pose[1] = opt_t_[1]; pose[2] = opt_t_[2];
  pose[3] = opt_q_[0]; pose[4] = opt_q_[1]; pose[5] = opt_q_[2]; pose[6] = opt_q_[3];

  frame_poses_[frame_id_] = pose;
  frame_stamps_[frame_id_] = left->header.stamp;
  frame_kps_[frame_id_]   = kp_left_;

  // ---- 15. WRITE POSE FIRST --------------------------------------------
  writePoseToFile(pose);

  // ---- 16. Update latency -----------------------------------------------
  ++processed_frames_;
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
  if (ms < 0.0) ms = 0.0;
  total_latency_ += ms;

  RCLCPP_INFO(get_logger(),
              "KITTI Frame %06d (idx %d): Pose written | Matches: %zu | Inliers: %d | This: %.2f ms | Avg: %.2f ms",
              frame_id_, frame_id_, good_matches_.size(), inliers, ms, total_latency_ / processed_frames_);

  // ---- 17. Optimize current pose ----------------------------------------
  optimizePose();

  // ---- 18. Bundle Adjustment --------------------------------------------
  addToBA();

  // ---- 19. Marginalization ----------------------------------------------
  if (frame_poses_.size() > static_cast<size_t>(window_size_ + 2)) {
    int oldest = frame_id_ - window_size_ - 2;
    frame_poses_.erase(oldest);
    frame_stamps_.erase(oldest);
    frame_kps_.erase(oldest);
  }

  ++frame_id_;
}

/* ------------------------------------------------------------------ */
bool Soft2Node::isValidMatch(const cv::Point2f& pl,
                             const cv::Point2f& pr) const {
  return std::abs(pl.y - pr.y) < 5.0 && (pl.x - pr.x) > 0.0;
}

/* ------------------------------------------------------------------ */
void Soft2Node::optimizePose() {
  ceres::Problem prob;
  double huber = 1.0 + 0.1 * (total_latency_ / std::max(1, processed_frames_));

  for (const auto& m : good_matches_) {
    const auto& pl = kp_left_[m.queryIdx].pt;
    const auto& pr = kp_right_[m.trainIdx].pt;
    auto* cost = new ceres::AutoDiffCostFunction<StereoCost, 2, 3, 4>(
        new StereoCost(pl.x, pl.y, pr.x, pr.y,
                       baseline_, focal_, cx_, cy_));
    prob.AddResidualBlock(cost, new ceres::HuberLoss(huber), opt_t_, opt_q_);
  }
  if (prob.NumResiduals() == 0) return;

  ceres::Solver::Options opt;
  opt.max_num_iterations = std::min(100,
      safe_max_iter(200, 50, processed_frames_, total_latency_));
  opt.linear_solver_type = ceres::DENSE_QR;
  ceres::Solver::Summary summary;
  ceres::Solve(opt, &prob, &summary);
}

/* ------------------------------------------------------------------ */
void Soft2Node::addToBA() {
  if (frame_id_ < 2) return;

  ceres::Problem prob;
  int ws = window_size_;

  for (int id = std::max(0, frame_id_ - ws); id <= frame_id_; ++id) {
    if (!frame_poses_.count(id) || !frame_kps_.count(id)) continue;

    if (id == frame_id_) {
      double huber = 1.0 + 0.1 * (total_latency_ / std::max(1, processed_frames_));
      for (const auto& m : good_matches_) {
        const auto& pl = kp_left_[m.queryIdx].pt;
        const auto& pr = kp_right_[m.trainIdx].pt;
        auto* cost = new ceres::AutoDiffCostFunction<StereoCost, 2, 3, 4>(
            new StereoCost(pl.x, pl.y, pr.x, pr.y,
                           baseline_, focal_, cx_, cy_));
        prob.AddResidualBlock(cost, new ceres::HuberLoss(huber),
                              &frame_poses_[id][0], &frame_poses_[id][3]);
      }
    }

    if (id > 0 && frame_kps_.count(id - 1)) {
      const auto& prev_kps = frame_kps_.at(id - 1);
      std::vector<double> weights;
      for (size_t i = 0; i < std::min(kp_left_.size(), prev_kps.size()); ++i) {
        cv::Point2f cur = kp_left_[i].pt;
        cv::Point2f pre = prev_kps[i].pt;
        double d = std::hypot(cur.x - pre.x, cur.y - pre.y);
        double w = (d < 50.0) ? 1.0 / (d + 1e-6) : 0.0;
        weights.push_back(w);
      }
      double max_w = weights.empty() ? 1.0 : *std::max_element(weights.begin(), weights.end());

      for (size_t i = 0; i < std::min(kp_left_.size(), prev_kps.size()); ++i) {
        if (weights[i] > 0.1 * max_w) {
          auto* cost = new ceres::AutoDiffCostFunction<TemporalCost, 2, 3, 4, 3, 4>(
              new TemporalCost(prev_kps[i].pt, kp_left_[i].pt,
                               focal_, cx_, cy_, weights[i]));
          prob.AddResidualBlock(cost, new ceres::HuberLoss(1.0),
                                &frame_poses_[id - 1][0], &frame_poses_[id - 1][3],
                                &frame_poses_[id][0],     &frame_poses_[id][3]);
        }
      }
    }
  }

  if (prob.NumResiduals() == 0) return;

  ceres::Solver::Options opt;
  opt.max_num_iterations = std::min(50,
      safe_max_iter(100, 20, processed_frames_, total_latency_));
  opt.linear_solver_type = ceres::SPARSE_SCHUR;
  ceres::Solver::Summary summary;
  ceres::Solve(opt, &prob, &summary);
}

/* ------------------------------------------------------------------ */
void Soft2Node::timerCallback() {
  if (frame_poses_.find(frame_id_ - 1) == frame_poses_.end()) return;

  auto msg = nav_msgs::msg::Odometry();
  msg.header.frame_id = "odom";
  msg.child_frame_id = "base_link";
  msg.header.stamp = frame_stamps_.at(frame_id_ - 1);

  const auto& p = frame_poses_.at(frame_id_ - 1);
  msg.pose.pose.position.x = p[0] * baseline_;
  msg.pose.pose.position.y = p[1] * baseline_;
  msg.pose.pose.position.z = p[2] * baseline_;
  msg.pose.pose.orientation.w = p[3];
  msg.pose.pose.orientation.x = p[4];
  msg.pose.pose.orientation.y = p[5];
  msg.pose.pose.orientation.z = p[6];

  pub_odom_->publish(msg);
}

/* ------------------------------------------------------------------ */
void Soft2Node::writePoseToFile(const std::vector<double>& pose) {
  double tx = pose[0] * baseline_;
  double ty = pose[1] * baseline_;
  double tz = pose[2] * baseline_;
  double w = pose[3], x = pose[4], y = pose[5], z = pose[6];

  double R[9] = {
      1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w),
      2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w),
      2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)
  };

  traj_file_ << std::fixed << std::setprecision(9)
             << R[0] << " " << R[1] << " " << R[2] << " " << tx << " "
             << R[3] << " " << R[4] << " " << R[5] << " " << ty << " "
             << R[6] << " " << R[7] << " " << R[8] << " " << tz << "\n";
}

/* ------------------------------------------------------------------ */
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Soft2Node>());
  rclcpp::shutdown();
  return 0;
}
