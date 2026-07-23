/*
 * Coco-LIC: Continuous-Time Tightly-Coupled LiDAR-Inertial-Camera Odometry
 * using Non-Uniform B-spline
 *
 * Read-only continuous-time LiDAR localizability diagnostics.
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <memory>
#include <string>

#include <lidar/lidar_feature.h>
#include <spline/trajectory.h>

namespace cocolic {

struct CtLidarObservabilityResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool valid = false;
  bool raw_degenerate = false;
  bool persistent_degenerate = false;
  int weak_rank = 0;
  int correspondence_count = 0;
  int evaluated_factor_count = 0;
  int skipped_nondifferentiable_line_count = 0;
  int free_knot_count = 0;
  int64_t scan_time_ns = 0;
  int64_t reference_time_ns = 0;
  double characteristic_length = 1.0;
  double numerical_floor = 0.0;
  double decision_threshold = 3.0e-3;
  Eigen::Matrix<double, 6, 1> eigenvalues =
      Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> relative_eigenvalues =
      Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, Eigen::Dynamic> weak_basis;
  std::string state = "invalid";
};

class CtLidarObservability {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<CtLidarObservability>;

  CtLidarObservability(const YAML::Node& node, Trajectory::Ptr trajectory,
                       const std::string& output_prefix);

  bool Enabled() const { return enabled_; }

  CtLidarObservabilityResult Analyze(
      int64_t scan_time_ns, int64_t reference_time_ns,
      const Eigen::aligned_vector<PointCorrespondence>& correspondences,
      int64_t opt_min_time_ns, int64_t opt_max_time_ns,
      bool use_correspondence_scale);

 private:
  bool IsTimeUsable(int64_t time_ns) const;

  void UpdatePersistentState(bool enter_condition, bool exit_condition);

  void WriteCsvHeader();

  void WriteCsv(const CtLidarObservabilityResult& result);

  Trajectory::Ptr trajectory_;
  bool enabled_ = false;
  bool output_csv_ = true;
  double weak_eigenvalue_ratio_ = 3.0e-3;
  int scan_count_ = 0;
  int enter_count_ = 0;
  int exit_count_ = 0;
  bool persistent_degenerate_ = false;
  std::ofstream csv_;
};

}  // namespace cocolic
