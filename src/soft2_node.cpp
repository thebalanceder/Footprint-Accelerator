// src/soft2_vo/src/soft2_node.cpp
#include "soft2_vo/soft2_node.hpp"
#include "soft2_vo/optimize_module.hpp"
#include <iomanip>
#include <algorithm> // For std::min_element

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

    T proj_x = p_cam_curr[0] / p_cam_curr[2];
    T proj_y = p_cam_curr[1] / p_cam_curr[2];

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
  declare_parameter("max_keypoints", 1000); // New parameter for keypoint limit
  baseline_ = get_parameter("baseline").as_double();
  max_keypoints_ = get_parameter("max_keypoints").as_int();

  orb_ = cv::ORB::create(max_keypoints_);
  matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, true);

  auto qos = rclcpp::QoS(rclcpp::KeepLast(1000)).reliable();
  sub_left_sync_.subscribe(this, "/camera/left/image_raw", qos.get_rmw_qos_profile());
  sub_right_sync_.subscribe(this, "/camera/right/image_raw", qos.get_rmw_qos_profile());

  sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(100), sub_left_sync_, sub_right_sync_);
  sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.1));

  sync_->registerCallback(std::bind(&Soft2Node::syncedCallback, this, std::placeholders::_1, std::placeholders::_2));

  pub_odom_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
  timer_ = this->create_wall_timer(std::chrono::milliseconds(100), std::bind(&Soft2Node::timerCallback, this));

  traj_file_.open("trajectory.txt");
  if (!traj_file_.is_open()) {
    RCLCPP_ERROR(get_logger(), "Failed to open trajectory file.");
  }
}

Soft2Node::~Soft2Node() {
  if (traj_file_.is_open()) traj_file_.close();
}

void Soft2Node::syncedCallback(const ImageMsg::ConstSharedPtr& left,
                               const ImageMsg::ConstSharedPtr& right)
{
  auto start_time = std::chrono::high_resolution_clock::now();

  // -----------------------------------------------------------------
  // 1. Convert ROS images to OpenCV (mono8)
  // -----------------------------------------------------------------
  cv_bridge::CvImageConstPtr cv_left  = cv_bridge::toCvShare(left,  "mono8");
  cv_bridge::CvImageConstPtr cv_right = cv_bridge::toCvShare(right, "mono8");

  // -----------------------------------------------------------------
  // 2. OPTIONAL: 4x4 grid mask (only for detection, NOT for matching)
  // -----------------------------------------------------------------
  static cv::Mat grid_mask;
  if (grid_mask.empty()) {
    grid_mask = cv::Mat::zeros(cv_left->image.size(), CV_8UC1);
    const int gs = 4;
    const int cw = cv_left->image.cols  / gs;
    const int ch = cv_left->image.rows  / gs;
    for (int i = 0; i < gs; ++i)
      for (int j = 0; j < gs; ++j)
        grid_mask(cv::Rect(j*cw, i*ch, cw, ch)).setTo(255);
  }

  // -----------------------------------------------------------------
  // 3. Detect + compute ORB features
  // -----------------------------------------------------------------
  orb_->detectAndCompute(cv_left->image,  grid_mask, kp_left_,  desc_left_);
  orb_->detectAndCompute(cv_right->image, grid_mask, kp_right_, desc_right_);

  // -----------------------------------------------------------------
  // 4. Limit to max_keypoints_ (retain best)
  // -----------------------------------------------------------------
  if (kp_left_.size() > max_keypoints_) {
    cv::KeyPointsFilter::retainBest(kp_left_, max_keypoints_);
    desc_left_ = cv::Mat(desc_left_,
                         cv::Range(0, static_cast<int>(kp_left_.size())),
                         cv::Range::all()).clone();
  }
  if (kp_right_.size() > max_keypoints_) {
    cv::KeyPointsFilter::retainBest(kp_right_, max_keypoints_);
    desc_right_ = cv::Mat(desc_right_,
                          cv::Range(0, static_cast<int>(kp_right_.size())),
                          cv::Range::all()).clone();
  }

  // -----------------------------------------------------------------
  // 5. Safety checks
  // -----------------------------------------------------------------
  assert(desc_left_.rows  == static_cast<int>(kp_left_.size()));
  assert(desc_right_.rows == static_cast<int>(kp_right_.size()));
  assert(desc_left_.type()  == CV_8U && desc_right_.type() == CV_8U);
  assert(desc_left_.isContinuous() && desc_right_.isContinuous());

  RCLCPP_INFO(get_logger(),
              "Left:  kp=%zu  desc=%dx%d  (cont=%d)",
              kp_left_.size(), desc_left_.rows, desc_left_.cols, desc_left_.isContinuous());
  RCLCPP_INFO(get_logger(),
              "Right: kp=%zu  desc=%dx%d  (cont=%d)",
              kp_right_.size(), desc_right_.rows, desc_right_.cols, desc_right_.isContinuous());

  // -----------------------------------------------------------------
  // 6. Matching – K=2, NO mask, NO cross-check
  // -----------------------------------------------------------------
  cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
  std::vector<std::vector<cv::DMatch>> knn_matches;
  matcher->knnMatch(desc_left_, desc_right_, knn_matches, 2);

  good_matches_.clear();
  for (const auto& pair : knn_matches) {
    if (pair.size() != 2) continue;
    if (pair[0].distance > 0.7 * pair[1].distance) continue;
    const cv::Point2f &pl = kp_left_[pair[0].queryIdx].pt;
    const cv::Point2f &pr = kp_right_[pair[0].trainIdx].pt;
    if (isValidMatch(pl, pr))
        good_matches_.push_back(pair[0]);
  }

  if (good_matches_.size() < 10) {
    RCLCPP_WARN(get_logger(),
                "Too few good matches (%zu), skipping frame.", good_matches_.size());
    return;
  }

  // -----------------------------------------------------------------
  // 7. Build point vectors
  // -----------------------------------------------------------------
  std::vector<cv::Point2f> pts_l, pts_r;
  for (const auto& m : good_matches_) {
    pts_l.push_back(kp_left_[m.queryIdx].pt);
    pts_r.push_back(kp_right_[m.trainIdx].pt);
  }

  // -----------------------------------------------------------------
  // 8. Estimate Essential matrix + recover pose
  // -----------------------------------------------------------------
  cv::Mat E = cv::findEssentialMat(pts_l, pts_r, focal_, cv::Point2d(cx_, cy_),
                                   cv::RANSAC, 0.999, 1.0);
  cv::Mat R, t;
  cv::recoverPose(E, pts_l, pts_r, R, t, focal_, cv::Point2d(cx_, cy_));  // FIXED LINE

  double q[4];
  ceres::RotationMatrixToQuaternion(R.ptr<double>(), q);

  opt_t_[0] = t.at<double>(0);
  opt_t_[1] = t.at<double>(1);
  opt_t_[2] = t.at<double>(2);
  opt_q_[0] = q[0];
  opt_q_[1] = q[1];
  opt_q_[2] = q[2];
  opt_q_[3] = q[3];

  // -----------------------------------------------------------------
  // 9. Refine current pose
  // -----------------------------------------------------------------
  optimizePose();

  // -----------------------------------------------------------------
  // 10. Store pose
  // -----------------------------------------------------------------
  std::vector<double> pose(7);
  pose[0] = opt_t_[0];
  pose[1] = opt_t_[1];
  pose[2] = opt_t_[2];
  pose[3] = opt_q_[0];
  pose[4] = opt_q_[1];
  pose[5] = opt_q_[2];
  pose[6] = opt_q_[3];

  frame_poses_[frame_id_] = pose;
  frame_stamps_[frame_id_] = left->header.stamp;
  frame_kps_[frame_id_]   = kp_left_;

  // -----------------------------------------------------------------
  // 11. Bundle Adjustment
  // -----------------------------------------------------------------
  addToBA();
  writePoseToFile(pose);

  // -----------------------------------------------------------------
  // 12. Latency
  // -----------------------------------------------------------------
  auto end_time = std::chrono::high_resolution_clock::now();
  double latency = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
  total_latency_ += latency;
  processed_frames_++;

  RCLCPP_INFO(get_logger(),
              "Processed frame %d, average latency: %.2f ms",
              frame_id_, total_latency_ / processed_frames_);

  // -----------------------------------------------------------------
  // 13. Marginalization
  // -----------------------------------------------------------------
  if (frame_poses_.size() > window_size_ + 2) {
    int oldest_id = frame_id_ - window_size_ - 2;
    frame_poses_.erase(oldest_id);
    frame_stamps_.erase(oldest_id);
    frame_kps_.erase(oldest_id);
  }

  frame_id_++;
}

bool Soft2Node::isValidMatch(const cv::Point2f& pl, const cv::Point2f& pr) const {
  return std::abs(pl.y - pr.y) < 5.0 && (pl.x - pr.x) > 0;
}

void Soft2Node::optimizePose() {
  ceres::Problem prob;

  double huber_threshold = 1.0 + 0.1 * (total_latency_ / processed_frames_);
  for (const auto& m : good_matches_) {
    const auto& pl = kp_left_[m.queryIdx].pt;
    const auto& pr = kp_right_[m.trainIdx].pt;
    auto* cost = new ceres::AutoDiffCostFunction<StereoCost, 2, 3, 4>(
        new StereoCost(pl.x, pl.y, pr.x, pr.y,
                       baseline_, focal_, cx_, cy_));
    prob.AddResidualBlock(cost, new ceres::HuberLoss(huber_threshold), opt_t_, opt_q_);
  }
  if (prob.NumResiduals() == 0) return;

  ceres::Solver::Options opt;
  opt.max_num_iterations = std::min(100, static_cast<int>(200 / (total_latency_ / processed_frames_ + 1)));
  opt.linear_solver_type = ceres::DENSE_QR;
  ceres::Solver::Summary summary;
  ceres::Solve(opt, &prob, &summary);
}

void Soft2Node::addToBA() {
  if (frame_id_ < 2) return;

  ceres::Problem prob;

  int window_size = window_size_;
  for (int id = std::max(0, frame_id_ - window_size); id <= frame_id_; ++id) {
    if (!frame_poses_.count(id) || !frame_kps_.count(id)) continue;

    if (id == frame_id_) {
      double huber_threshold = 1.0 + 0.1 * (total_latency_ / processed_frames_);
      for (const auto& m : good_matches_) {
        const auto& pl = kp_left_[m.queryIdx].pt;
        const auto& pr = kp_right_[m.trainIdx].pt;
        auto* cost = new ceres::AutoDiffCostFunction<StereoCost, 2, 3, 4>(
            new StereoCost(pl.x, pl.y, pr.x, pr.y,
                           baseline_, focal_, cx_, cy_));
        prob.AddResidualBlock(cost, new ceres::HuberLoss(huber_threshold),
                              &frame_poses_[id][0],
                              &frame_poses_[id][3]);
      }
    }

    if (id > 0 && frame_kps_.count(id - 1)) {
      const auto& prev = frame_kps_[id - 1];
      std::vector<double> weights;
      for (size_t i = 0; i < std::min(kp_left_.size(), prev.size()); ++i) {
        cv::Point2f pt_curr = kp_left_[i].pt;
        cv::Point2f pt_prev = prev[i].pt;
        double dx = pt_curr.x - pt_prev.x;
        double dy = pt_curr.y - pt_prev.y;
        double dist = std::sqrt(dx*dx + dy*dy);
        double weight = (dist < 50.0) ? 1.0 / (dist + 1e-6) : 0.0;
        weights.push_back(weight);
      }
      double max_weight = *std::max_element(weights.begin(), weights.end());
      for (size_t i = 0; i < std::min(kp_left_.size(), prev.size()); ++i) {
        if (weights[i] > 0.1 * max_weight) {
          auto* cost = new ceres::AutoDiffCostFunction<TemporalCost, 2, 3, 4, 3, 4>(
              new TemporalCost(prev[i].pt, kp_left_[i].pt,
                               focal_, cx_, cy_, weights[i]));
          prob.AddResidualBlock(cost, new ceres::HuberLoss(1.0),
                                &frame_poses_[id - 1][0],
                                &frame_poses_[id - 1][3],
                                &frame_poses_[id][0],
                                &frame_poses_[id][3]);
        }
      }
    }
  }

  if (prob.NumResiduals() == 0) return;

  ceres::Solver::Options opt;
  opt.linear_solver_type = ceres::SPARSE_SCHUR;
  opt.max_num_iterations = std::min(50, static_cast<int>(100 / (total_latency_ / processed_frames_ + 1)));
  ceres::Solver::Summary summary;
  ceres::Solve(opt, &prob, &summary);
}

void Soft2Node::timerCallback() {
  if (frame_poses_.find(frame_id_ - 1) == frame_poses_.end()) return;

  auto msg = nav_msgs::msg::Odometry();
  msg.header.frame_id = "odom";
  msg.child_frame_id = "base_link";
  msg.header.stamp = frame_stamps_[frame_id_ - 1];

  const auto& p = frame_poses_[frame_id_ - 1];
  msg.pose.pose.position.x = p[0] * baseline_;
  msg.pose.pose.position.y = p[1] * baseline_;
  msg.pose.pose.position.z = p[2] * baseline_;
  msg.pose.pose.orientation.w = p[3];
  msg.pose.pose.orientation.x = p[4];
  msg.pose.pose.orientation.y = p[5];
  msg.pose.pose.orientation.z = p[6];

  pub_odom_->publish(msg);
}

void Soft2Node::writePoseToFile(const std::vector<double>& pose) {
  double w = pose[3], x = pose[4], y = pose[5], z = pose[6];
  double R00 = 1 - 2*(y*y + z*z);
  double R01 = 2*(x*y - z*w);
  double R02 = 2*(x*z + y*w);
  double R10 = 2*(x*y + z*w);
  double R11 = 1 - 2*(x*x + z*z);
  double R12 = 2*(y*z - x*w);
  double R20 = 2*(x*z - y*w);
  double R21 = 2*(y*z + x*w);
  double R22 = 1 - 2*(x*x + y*y);

  traj_file_ << std::fixed << std::setprecision(9)
             << R00 << " " << R01 << " " << R02 << " " << pose[0] << " "
             << R10 << " " << R11 << " " << R12 << " " << pose[1] << " "
             << R20 << " " << R21 << " " << R22 << " " << pose[2]
             << "\n";
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Soft2Node>());
  rclcpp::shutdown();
  return 0;
}
