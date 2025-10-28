#ifndef SOFT2_VO_SOFT2_NODE_HPP_
#define SOFT2_VO_SOFT2_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <unordered_map>
#include <vector>
#include <fstream>
#include <chrono>

using ImageMsg = sensor_msgs::msg::Image;
using SyncPolicy = message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg>;

class Soft2Node : public rclcpp::Node {
public:
  Soft2Node();
  ~Soft2Node() override;

private:
  void syncedCallback(const ImageMsg::ConstSharedPtr& left,
                      const ImageMsg::ConstSharedPtr& right);
  void timerCallback();
  void optimizePose();
  void addToBA();
  void writePoseToFile(const std::vector<double>& pose);
  bool isValidMatch(const cv::Point2f& pl, const cv::Point2f& pr) const;

  message_filters::Subscriber<ImageMsg> sub_left_sync_;
  message_filters::Subscriber<ImageMsg> sub_right_sync_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::TimerBase::SharedPtr timer_;

  cv::Ptr<cv::ORB> orb_;
  cv::Ptr<cv::BFMatcher> matcher_;
  std::vector<cv::KeyPoint> kp_left_, kp_right_;
  cv::Mat desc_left_, desc_right_;

  int frame_id_{0};
  double baseline_{0.54};
  int max_keypoints_{1000}; // New member for keypoint limit
  const int window_size_{5}; // Moved to member variable
  std::unordered_map<int, std::vector<double>> frame_poses_;
  std::unordered_map<int, rclcpp::Time> frame_stamps_;
  std::unordered_map<int, std::vector<cv::KeyPoint>> frame_kps_;

  double opt_t_[3]{0.0, 0.0, 0.0};
  double opt_q_[4]{1.0, 0.0, 0.0, 0.0};
  std::vector<cv::DMatch> good_matches_;

  const double focal_ = 718.8560;
  const double cx_ = 607.1928;
  const double cy_ = 185.2157;

  std::ofstream traj_file_;

  int processed_frames_{0};
  double total_latency_{0.0};

  std::vector<double> current_pose_;
};

#endif  // SOFT2_VO_SOFT2_NODE_HPP_
