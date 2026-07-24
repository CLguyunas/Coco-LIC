/*
 * Coco-LIC: Continuous-Time Tightly-Coupled LiDAR-Inertial-Camera Odometry
 * using Non-Uniform B-spline
 *
 * Cause-specific visual complement selection for CT LiDAR degeneracy.
 */

#pragma once

#include <degeneracy/ct_lidar_observability.h>
#include <spline/trajectory.h>

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace cocolic {

struct CtVisualRecoveryObservation {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int source_index = -1;
  Eigen::Vector3d point = Eigen::Vector3d::Zero();
  Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
  double factor_weight = 1.0;
  bool baseline_selected = false;
};

struct CtVisualRecoveryResult {
  bool valid = false;
  bool eligible = false;
  bool apply_requested = false;
  bool applied = false;
  int weak_rank = 0;
  int visual_candidate_count = 0;
  int valid_factor_count = 0;
  int rejected_factor_count = 0;
  int baseline_selected_count = 0;
  int eligible_unselected_count = 0;
  int additional_budget = 0;
  int additional_selected_count = 0;
  int weak_policy_selected_count = 0;
  int64_t scan_time_ns = 0;
  int64_t image_time_ns = 0;
  double weak_d_efficiency_target = 0.95;
  double baseline_weak_d_efficiency = 0.0;
  double repaired_weak_d_efficiency = 0.0;
  double baseline_weak_min_retention = 0.0;
  double repaired_weak_min_retention = 0.0;
  double baseline_global_d_efficiency = 0.0;
  double repaired_global_d_efficiency = 0.0;
  double weak_policy_weak_d_efficiency = 0.0;
  double weak_policy_weak_min_retention = 0.0;
  double weak_policy_global_d_efficiency = 0.0;
  double random_policy_weak_d_efficiency = 0.0;
  double random_policy_weak_min_retention = 0.0;
  double random_policy_global_d_efficiency = 0.0;
  double global_d_policy_weak_d_efficiency = 0.0;
  double global_d_policy_weak_min_retention = 0.0;
  double global_d_policy_global_d_efficiency = 0.0;
  double median_factorization_error = 0.0;
  double max_factorization_error = 0.0;
  std::vector<int> additional_source_indices;
  std::string selection_policy = "weak_subspace";
  std::string state = "invalid";
};

class CtVisualRecovery {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<CtVisualRecovery>;

  CtVisualRecovery(const YAML::Node& node, Trajectory::Ptr trajectory,
                   const Eigen::Matrix3d& intrinsic,
                   const std::string& output_prefix);

  bool Enabled() const { return enabled_; }

  bool ApplyToEstimator() const {
    return enabled_ && apply_to_estimator_;
  }

  CtVisualRecoveryResult Select(
      int64_t scan_time_ns, int64_t image_time_ns,
      const CtLidarObservabilityResult& lidar_result,
      const Eigen::aligned_vector<CtVisualRecoveryObservation>& observations,
      double weak_d_efficiency_target, double numerical_prior_relative,
      double minimum_marginal_gain);

 private:
  struct PoseInformation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int source_index = -1;
    bool baseline_selected = false;
    Eigen::Matrix<double, 6, 6> information =
        Eigen::Matrix<double, 6, 6>::Zero();
    double factorization_error =
        std::numeric_limits<double>::quiet_NaN();
  };

  bool BuildPoseInformation(
      int64_t image_time_ns,
      const CtLidarObservabilityResult& lidar_result,
      const CtVisualRecoveryObservation& observation,
      PoseInformation& output) const;

  double LogDet(const Eigen::MatrixXd& matrix) const;

  double RelativeDEfficiency(const Eigen::MatrixXd& selected,
                             const Eigen::MatrixXd& full) const;

  double MinimumDirectionRetention(const Eigen::MatrixXd& selected,
                                   const Eigen::MatrixXd& full) const;

  void WriteCsvHeader();

  void WriteCsv(const CtVisualRecoveryResult& result);

  Trajectory::Ptr trajectory_;
  Eigen::Matrix3d intrinsic_ = Eigen::Matrix3d::Identity();
  bool enabled_ = false;
  bool apply_to_estimator_ = false;
  bool output_csv_ = true;
  int max_additional_visual_observations_ = 64;
  uint32_t random_seed_ = 42;
  std::string selection_policy_ = "weak_subspace";
  std::ofstream csv_;
};

}  // namespace cocolic
