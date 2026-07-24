/*
 * Coco-LIC: Continuous-Time Tightly-Coupled LiDAR-Inertial-Camera Odometry
 * using Non-Uniform B-spline
 *
 * Cause-specific visual complement selection for CT LiDAR degeneracy.
 */

#include <degeneracy/ct_visual_recovery.h>

#include <odom/factor/analytic_diff/image_feature_factor.h>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <random>
#include <utility>
#include <vector>

namespace cocolic {
namespace {

constexpr double kAbsoluteFloor = 1.0e-12;
constexpr double kMaximumFactorizationError = 1.0e-5;
constexpr double kCauchyLossScale = 10.0;

double Clamp(double value, double low, double high) {
  return std::max(low, std::min(high, value));
}

double Median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  const size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  const double upper = values[middle];
  if (values.size() % 2 != 0) {
    return upper;
  }
  std::nth_element(values.begin(), values.begin() + middle - 1,
                   values.begin() + middle);
  return 0.5 * (values[middle - 1] + upper);
}

struct LazyWeakGain {
  double gain = -std::numeric_limits<double>::infinity();
  int information_index = -1;
  int evaluated_round = -1;
};

struct LazyWeakGainCompare {
  bool operator()(const LazyWeakGain& lhs,
                  const LazyWeakGain& rhs) const {
    if (lhs.gain == rhs.gain) {
      return lhs.information_index > rhs.information_index;
    }
    return lhs.gain < rhs.gain;
  }
};

}  // namespace

CtVisualRecovery::CtVisualRecovery(
    const YAML::Node& node, Trajectory::Ptr trajectory,
    const Eigen::Matrix3d& intrinsic, const std::string& output_prefix)
    : trajectory_(std::move(trajectory)), intrinsic_(intrinsic) {
  if (!node || !node.IsMap()) {
    return;
  }

  enabled_ = node["enabled"] ? node["enabled"].as<bool>() : false;
  apply_to_estimator_ =
      node["apply_to_estimator"]
          ? node["apply_to_estimator"].as<bool>()
          : false;
  output_csv_ =
      node["output_csv"] ? node["output_csv"].as<bool>() : true;
  if (node["max_additional_visual_observations"]) {
    max_additional_visual_observations_ = std::max(
        node["max_additional_visual_observations"].as<int>(), 0);
  }
  if (node["selection_policy"]) {
    selection_policy_ = node["selection_policy"].as<std::string>();
  }
  if (selection_policy_ != "weak_subspace" &&
      selection_policy_ != "random" &&
      selection_policy_ != "global_d") {
    std::cerr << "[CT-Visual] Unknown selection_policy='"
              << selection_policy_
              << "'; using weak_subspace.\n";
    selection_policy_ = "weak_subspace";
  }
  if (node["random_seed"]) {
    random_seed_ = node["random_seed"].as<uint32_t>();
  }

  if (enabled_ && output_csv_) {
    csv_.open(output_prefix + "_ct_visual_recovery.csv",
              std::ios::out | std::ios::trunc);
    if (csv_.is_open()) {
      WriteCsvHeader();
    } else {
      std::cerr << "[CT-Visual] Unable to open recovery CSV at "
                << output_prefix << "_ct_visual_recovery.csv\n";
    }
  }
}

bool CtVisualRecovery::BuildPoseInformation(
    int64_t image_time_ns,
    const CtLidarObservabilityResult& lidar_result,
    const CtVisualRecoveryObservation& observation,
    PoseInformation& output) const {
  if (!trajectory_ || observation.source_index < 0 ||
      !observation.point.allFinite() || !observation.pixel.allFinite() ||
      !std::isfinite(observation.factor_weight) ||
      observation.factor_weight <= 0.0 ||
      lidar_result.free_knot_indices.empty() ||
      lidar_result.control_to_reference_pose.rows() != 6 ||
      lidar_result.reference_pose_lift.cols() != 6 ||
      !lidar_result.control_to_reference_pose.allFinite() ||
      !lidar_result.reference_pose_lift.allFinite()) {
    return false;
  }

  const int state_dimension =
      6 * static_cast<int>(lidar_result.free_knot_indices.size());
  if (lidar_result.control_to_reference_pose.cols() != state_dimension ||
      lidar_result.reference_pose_lift.rows() != state_dimension) {
    return false;
  }

  std::pair<int, double> su{-1, 0.0};
  trajectory_->GetIdxT(image_time_ns, su);
  if (su.first < 3) {
    return false;
  }
  const int support_start = su.first - 3;
  if (support_start + 3 >=
      static_cast<int>(trajectory_->numKnots())) {
    return false;
  }

  analytic_derivative::PnPFactorNURBS factor(
      image_time_ns, su, trajectory_->blending_mats[support_start],
      trajectory_->cumu_blending_mats[support_start],
      observation.point, observation.pixel,
      trajectory_->GetSensorEP(CameraSensor).so3,
      trajectory_->GetSensorEP(CameraSensor).p, intrinsic_,
      observation.factor_weight);

  std::array<const double*, 8> parameters{};
  for (int local = 0; local < 4; ++local) {
    parameters[local] =
        trajectory_->getKnotSO3(support_start + local).data();
    parameters[4 + local] =
        trajectory_->getKnotPos(support_start + local).data();
  }

  std::array<std::array<double, 8>, 4> rotation_jacobians{};
  std::array<std::array<double, 6>, 4> position_jacobians{};
  std::array<double*, 8> jacobians{};
  for (int local = 0; local < 4; ++local) {
    jacobians[local] = rotation_jacobians[local].data();
    jacobians[4 + local] = position_jacobians[local].data();
  }

  std::array<double, 2> residuals{};
  if (!factor.Evaluate(parameters.data(), residuals.data(),
                       jacobians.data()) ||
      !std::isfinite(residuals[0]) ||
      !std::isfinite(residuals[1])) {
    return false;
  }

  std::map<int, int> knot_to_block;
  for (size_t block = 0;
       block < lidar_result.free_knot_indices.size(); ++block) {
    knot_to_block.emplace(lidar_result.free_knot_indices[block],
                          static_cast<int>(block));
  }

  Eigen::MatrixXd control_jacobian =
      Eigen::MatrixXd::Zero(2, state_dimension);
  for (int local = 0; local < 4; ++local) {
    const int knot_index = support_start + local;
    const auto block_iter = knot_to_block.find(knot_index);
    if (block_iter == knot_to_block.end()) {
      continue;
    }
    const int offset = 6 * block_iter->second;
    const Eigen::Map<
        const Eigen::Matrix<double, 2, 4, Eigen::RowMajor>>
        rotation_block(rotation_jacobians[local].data());
    const Eigen::Map<
        const Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>
        position_block(position_jacobians[local].data());
    control_jacobian.block<2, 3>(0, offset) =
        rotation_block.leftCols<3>();
    control_jacobian.block<2, 3>(0, offset + 3) =
        position_block;
  }

  if (!control_jacobian.allFinite() ||
      control_jacobian.squaredNorm() <= kAbsoluteFloor) {
    return false;
  }

  // Match the Cauchy loss used by the existing PnP residual. This is not a
  // new weight: it is the local robust curvature of the real factor that
  // will be added to Ceres if the recovery is armed.
  const double squared_residual =
      residuals[0] * residuals[0] + residuals[1] * residuals[1];
  const double cauchy_scale_squared =
      kCauchyLossScale * kCauchyLossScale;
  const double robust_first_derivative =
      1.0 / (1.0 + squared_residual / cauchy_scale_squared);
  control_jacobian *=
      std::sqrt(std::max(robust_first_derivative, 0.0));

  const Eigen::Matrix<double, 2, 6> pose_jacobian =
      control_jacobian * lidar_result.reference_pose_lift;
  const Eigen::MatrixXd reconstructed_control_jacobian =
      pose_jacobian * lidar_result.control_to_reference_pose;
  output.factorization_error =
      (control_jacobian - reconstructed_control_jacobian).norm() /
      std::max(control_jacobian.norm(), kAbsoluteFloor);
  if (!std::isfinite(output.factorization_error) ||
      output.factorization_error > kMaximumFactorizationError ||
      !pose_jacobian.allFinite() ||
      pose_jacobian.squaredNorm() <= kAbsoluteFloor) {
    return false;
  }

  output.source_index = observation.source_index;
  output.baseline_selected = observation.baseline_selected;
  output.information.noalias() =
      pose_jacobian.transpose() * pose_jacobian;
  return output.information.allFinite() &&
         output.information.trace() > kAbsoluteFloor;
}

double CtVisualRecovery::LogDet(const Eigen::MatrixXd& matrix) const {
  if (matrix.rows() == 0 || matrix.rows() != matrix.cols()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
      0.5 * (matrix + matrix.transpose()));
  if (solver.info() != Eigen::Success ||
      solver.eigenvalues().minCoeff() <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return solver.eigenvalues().array().log().sum();
}

double CtVisualRecovery::RelativeDEfficiency(
    const Eigen::MatrixXd& selected,
    const Eigen::MatrixXd& full) const {
  if (selected.rows() == 0 || selected.rows() != selected.cols() ||
      selected.rows() != full.rows() || full.rows() != full.cols()) {
    return 0.0;
  }
  const double selected_logdet = LogDet(selected);
  const double full_logdet = LogDet(full);
  if (!std::isfinite(selected_logdet) ||
      !std::isfinite(full_logdet)) {
    return 0.0;
  }
  const double log_ratio =
      (selected_logdet - full_logdet) /
      static_cast<double>(selected.rows());
  return Clamp(std::exp(std::min(log_ratio, 0.0)), 0.0, 1.0);
}

double CtVisualRecovery::MinimumDirectionRetention(
    const Eigen::MatrixXd& selected,
    const Eigen::MatrixXd& full) const {
  if (selected.rows() == 0 || selected.rows() != selected.cols() ||
      selected.rows() != full.rows() || full.rows() != full.cols()) {
    return 0.0;
  }
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(
      0.5 * (selected + selected.transpose()),
      0.5 * (full + full.transpose()),
      Eigen::EigenvaluesOnly);
  if (solver.info() != Eigen::Success) {
    return 0.0;
  }
  return Clamp(solver.eigenvalues().minCoeff(), 0.0, 1.0);
}

CtVisualRecoveryResult CtVisualRecovery::Select(
    int64_t scan_time_ns, int64_t image_time_ns,
    const CtLidarObservabilityResult& lidar_result,
    const Eigen::aligned_vector<CtVisualRecoveryObservation>& observations,
    double weak_d_efficiency_target, double numerical_prior_relative,
    double minimum_marginal_gain) {
  CtVisualRecoveryResult result;
  result.scan_time_ns = scan_time_ns;
  result.image_time_ns = image_time_ns;
  result.apply_requested = ApplyToEstimator();
  result.selection_policy = selection_policy_;
  result.visual_candidate_count =
      static_cast<int>(observations.size());
  result.weak_rank = lidar_result.weak_rank;
  result.additional_budget = max_additional_visual_observations_;
  result.weak_d_efficiency_target =
      Clamp(weak_d_efficiency_target, 0.0, 1.0);
  for (const auto& observation : observations) {
    result.baseline_selected_count +=
        observation.baseline_selected ? 1 : 0;
  }

  if (!enabled_) {
    result.state = "disabled";
    WriteCsv(result);
    return result;
  }
  if (!lidar_result.valid) {
    result.state = "ct_invalid";
    WriteCsv(result);
    return result;
  }
  if (!lidar_result.persistent_degenerate) {
    result.state = "ct_not_persistent";
    WriteCsv(result);
    return result;
  }
  if (lidar_result.reference_time_ns != image_time_ns) {
    result.state = "reference_time_mismatch";
    WriteCsv(result);
    return result;
  }
  if (lidar_result.weak_rank <= 0 ||
      lidar_result.weak_basis.rows() != 6 ||
      lidar_result.weak_basis.cols() != lidar_result.weak_rank ||
      !lidar_result.weak_basis.allFinite()) {
    result.state = "no_weak_direction";
    WriteCsv(result);
    return result;
  }
  result.eligible = true;
  if (observations.empty()) {
    result.state = "no_visual_observation";
    WriteCsv(result);
    return result;
  }

  Eigen::aligned_vector<PoseInformation> pose_information;
  pose_information.reserve(observations.size());
  std::vector<double> factorization_errors;
  factorization_errors.reserve(observations.size());
  for (const auto& observation : observations) {
    PoseInformation information;
    const bool factor_valid =
        BuildPoseInformation(image_time_ns, lidar_result, observation,
                             information);
    if (std::isfinite(information.factorization_error)) {
      factorization_errors.push_back(information.factorization_error);
    }
    if (factor_valid) {
      pose_information.push_back(information);
    } else {
      ++result.rejected_factor_count;
    }
  }
  result.valid_factor_count =
      static_cast<int>(pose_information.size());
  result.median_factorization_error = Median(factorization_errors);
  result.max_factorization_error =
      factorization_errors.empty()
          ? 0.0
          : *std::max_element(factorization_errors.begin(),
                              factorization_errors.end());
  if (pose_information.empty()) {
    result.state = "no_factor_consistent_visual";
    WriteCsv(result);
    return result;
  }

  Eigen::Matrix<double, 6, 6> global_information_full =
      Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 6> global_information_baseline =
      Eigen::Matrix<double, 6, 6>::Zero();
  for (const auto& information : pose_information) {
    global_information_full += information.information;
    if (information.baseline_selected) {
      global_information_baseline += information.information;
    } else {
      ++result.eligible_unselected_count;
    }
  }

  const Eigen::MatrixXd weak_basis = lidar_result.weak_basis;
  const Eigen::MatrixXd weak_information_full_raw =
      weak_basis.transpose() * global_information_full * weak_basis;
  const Eigen::MatrixXd weak_information_baseline_raw =
      weak_basis.transpose() * global_information_baseline * weak_basis;
  const double weak_information_scale =
      weak_information_full_raw.trace() /
      static_cast<double>(lidar_result.weak_rank);
  if (!std::isfinite(weak_information_scale) ||
      weak_information_scale <= kAbsoluteFloor) {
    result.state = "no_weak_visual_information";
    WriteCsv(result);
    return result;
  }

  const double global_information_scale =
      std::max(global_information_full.trace() / 6.0, kAbsoluteFloor);
  const double relative_prior =
      std::max(numerical_prior_relative, kAbsoluteFloor);
  const Eigen::Matrix<double, 6, 6> global_prior =
      relative_prior * global_information_scale *
      Eigen::Matrix<double, 6, 6>::Identity();
  const Eigen::MatrixXd weak_prior =
      relative_prior * weak_information_scale *
      Eigen::MatrixXd::Identity(lidar_result.weak_rank,
                                lidar_result.weak_rank);

  const Eigen::Matrix<double, 6, 6> global_full =
      global_prior + global_information_full;
  const Eigen::Matrix<double, 6, 6> global_baseline =
      global_prior + global_information_baseline;
  const Eigen::MatrixXd weak_full =
      weak_prior + weak_information_full_raw;
  Eigen::MatrixXd weak_repaired =
      weak_prior + weak_information_baseline_raw;
  Eigen::Matrix<double, 6, 6> global_repaired = global_baseline;

  result.baseline_weak_d_efficiency =
      RelativeDEfficiency(weak_repaired, weak_full);
  result.baseline_weak_min_retention =
      MinimumDirectionRetention(weak_repaired, weak_full);
  result.baseline_global_d_efficiency =
      RelativeDEfficiency(global_baseline, global_full);
  result.repaired_weak_d_efficiency =
      result.baseline_weak_d_efficiency;
  result.repaired_weak_min_retention =
      result.baseline_weak_min_retention;
  result.repaired_global_d_efficiency =
      result.baseline_global_d_efficiency;
  result.valid = true;

  if (result.baseline_weak_d_efficiency >=
      result.weak_d_efficiency_target) {
    result.state = "already_sufficient";
    WriteCsv(result);
    return result;
  }
  if (result.eligible_unselected_count == 0 ||
      max_additional_visual_observations_ == 0) {
    result.state = "no_repair_budget";
    WriteCsv(result);
    return result;
  }

  std::priority_queue<LazyWeakGain, std::vector<LazyWeakGain>,
                      LazyWeakGainCompare>
      gain_queue;
  const double initial_logdet = LogDet(weak_repaired);
  for (size_t index = 0; index < pose_information.size(); ++index) {
    if (pose_information[index].baseline_selected) {
      continue;
    }
    const Eigen::MatrixXd candidate_weak_information =
        weak_basis.transpose() * pose_information[index].information *
        weak_basis;
    LazyWeakGain entry;
    entry.gain =
        LogDet(weak_repaired + candidate_weak_information) -
        initial_logdet;
    entry.information_index = static_cast<int>(index);
    entry.evaluated_round = 0;
    gain_queue.push(entry);
  }

  const int repair_budget = std::min(
      max_additional_visual_observations_,
      result.eligible_unselected_count);
  std::vector<int> weak_policy_indices;
  result.state = "candidates_exhausted";
  for (int selected_count = 0;
       selected_count < repair_budget; ++selected_count) {
    const double current_logdet = LogDet(weak_repaired);
    double best_gain = -std::numeric_limits<double>::infinity();
    int best_index = -1;
    while (!gain_queue.empty()) {
      LazyWeakGain entry = gain_queue.top();
      gain_queue.pop();
      if (entry.information_index < 0) {
        continue;
      }
      if (entry.evaluated_round != selected_count) {
        const auto& information =
            pose_information[static_cast<size_t>(
                entry.information_index)];
        const Eigen::MatrixXd candidate_weak_information =
            weak_basis.transpose() * information.information * weak_basis;
        entry.gain =
            LogDet(weak_repaired + candidate_weak_information) -
            current_logdet;
        entry.evaluated_round = selected_count;
      }
      if (gain_queue.empty() ||
          entry.gain >= gain_queue.top().gain - 1.0e-12) {
        best_gain = entry.gain;
        best_index = entry.information_index;
        break;
      }
      gain_queue.push(entry);
    }

    if (best_index < 0) {
      result.state = "candidates_exhausted";
      break;
    }
    if (!std::isfinite(best_gain) ||
        best_gain < std::max(minimum_marginal_gain, 0.0)) {
      result.state = "gain_exhausted";
      break;
    }

    const auto& selected_information =
        pose_information[static_cast<size_t>(best_index)];
    weak_repaired +=
        weak_basis.transpose() * selected_information.information *
        weak_basis;
    global_repaired += selected_information.information;
    weak_policy_indices.push_back(best_index);
    result.additional_selected_count =
        static_cast<int>(weak_policy_indices.size());
    result.repaired_weak_d_efficiency =
        RelativeDEfficiency(weak_repaired, weak_full);
    if (result.repaired_weak_d_efficiency >=
        result.weak_d_efficiency_target) {
      result.state = "target_reached";
      break;
    }
    if (selected_count + 1 == repair_budget) {
      result.state = "budget_exhausted";
    }
  }

  result.repaired_weak_min_retention =
      MinimumDirectionRetention(weak_repaired, weak_full);
  result.repaired_global_d_efficiency =
      RelativeDEfficiency(global_repaired, global_full);
  result.weak_policy_selected_count =
      static_cast<int>(weak_policy_indices.size());
  result.weak_policy_weak_d_efficiency =
      result.repaired_weak_d_efficiency;
  result.weak_policy_weak_min_retention =
      result.repaired_weak_min_retention;
  result.weak_policy_global_d_efficiency =
      result.repaired_global_d_efficiency;

  std::vector<int> eligible_indices;
  eligible_indices.reserve(
      static_cast<size_t>(result.eligible_unselected_count));
  for (size_t index = 0; index < pose_information.size(); ++index) {
    if (!pose_information[index].baseline_selected) {
      eligible_indices.push_back(static_cast<int>(index));
    }
  }

  // All comparison policies receive exactly the number of observations that
  // the proposed weak-subspace policy requested on this frame. This removes
  // observation count as a confounder in both shadow audits and armed
  // trajectory comparisons.
  std::vector<int> random_policy_indices = eligible_indices;
  const uint64_t scan_bits = static_cast<uint64_t>(scan_time_ns);
  const uint64_t image_bits = static_cast<uint64_t>(image_time_ns);
  const uint32_t mixed_seed =
      random_seed_ ^ static_cast<uint32_t>(scan_bits) ^
      static_cast<uint32_t>(scan_bits >> 32) ^
      static_cast<uint32_t>(image_bits) ^
      static_cast<uint32_t>(image_bits >> 32);
  std::mt19937 random_generator(mixed_seed);
  std::shuffle(random_policy_indices.begin(), random_policy_indices.end(),
               random_generator);
  random_policy_indices.resize(
      std::min(random_policy_indices.size(),
               weak_policy_indices.size()));

  std::vector<int> global_d_policy_indices;
  Eigen::Matrix<double, 6, 6> global_d_information = global_baseline;
  std::vector<char> global_d_used(pose_information.size(), 0);
  for (size_t round = 0; round < weak_policy_indices.size(); ++round) {
    const double current_global_logdet = LogDet(global_d_information);
    double best_global_gain =
        -std::numeric_limits<double>::infinity();
    int best_global_index = -1;
    for (const int candidate_index : eligible_indices) {
      if (global_d_used[static_cast<size_t>(candidate_index)]) {
        continue;
      }
      const double gain =
          LogDet(global_d_information +
                 pose_information[static_cast<size_t>(candidate_index)]
                     .information) -
          current_global_logdet;
      if (gain > best_global_gain + 1.0e-12 ||
          (std::abs(gain - best_global_gain) <= 1.0e-12 &&
           (best_global_index < 0 ||
            candidate_index < best_global_index))) {
        best_global_gain = gain;
        best_global_index = candidate_index;
      }
    }
    if (best_global_index < 0 ||
        !std::isfinite(best_global_gain)) {
      break;
    }
    global_d_used[static_cast<size_t>(best_global_index)] = 1;
    global_d_policy_indices.push_back(best_global_index);
    global_d_information +=
        pose_information[static_cast<size_t>(best_global_index)]
            .information;
  }

  const auto evaluate_policy =
      [&](const std::vector<int>& indices, double& weak_d,
          double& weak_min, double& global_d) {
        Eigen::MatrixXd policy_weak =
            weak_prior + weak_information_baseline_raw;
        Eigen::Matrix<double, 6, 6> policy_global = global_baseline;
        for (const int information_index : indices) {
          if (information_index < 0 ||
              information_index >=
                  static_cast<int>(pose_information.size())) {
            continue;
          }
          const auto& information =
              pose_information[static_cast<size_t>(information_index)]
                  .information;
          policy_weak +=
              weak_basis.transpose() * information * weak_basis;
          policy_global += information;
        }
        weak_d = RelativeDEfficiency(policy_weak, weak_full);
        weak_min =
            MinimumDirectionRetention(policy_weak, weak_full);
        global_d =
            RelativeDEfficiency(policy_global, global_full);
      };

  evaluate_policy(
      random_policy_indices,
      result.random_policy_weak_d_efficiency,
      result.random_policy_weak_min_retention,
      result.random_policy_global_d_efficiency);
  evaluate_policy(
      global_d_policy_indices,
      result.global_d_policy_weak_d_efficiency,
      result.global_d_policy_weak_min_retention,
      result.global_d_policy_global_d_efficiency);

  const std::vector<int>* selected_policy_indices =
      &weak_policy_indices;
  if (selection_policy_ == "random") {
    selected_policy_indices = &random_policy_indices;
    result.state = "random_weak_count_matched";
  } else if (selection_policy_ == "global_d") {
    selected_policy_indices = &global_d_policy_indices;
    result.state = "global_d_weak_count_matched";
  }

  result.additional_source_indices.clear();
  for (const int information_index : *selected_policy_indices) {
    result.additional_source_indices.push_back(
        pose_information[static_cast<size_t>(information_index)]
            .source_index);
  }
  result.additional_selected_count =
      static_cast<int>(result.additional_source_indices.size());
  if (selection_policy_ != "weak_subspace") {
    evaluate_policy(*selected_policy_indices,
                    result.repaired_weak_d_efficiency,
                    result.repaired_weak_min_retention,
                    result.repaired_global_d_efficiency);
  }
  result.applied =
      result.apply_requested && result.additional_selected_count > 0;
  WriteCsv(result);
  return result;
}

void CtVisualRecovery::WriteCsvHeader() {
  csv_ << "scan_time_ns,image_time_ns,selection_policy,random_seed,"
          "valid,eligible,state,"
          "apply_requested,applied,weak_rank,visual_candidate_count,"
          "valid_factor_count,rejected_factor_count,"
          "baseline_selected_count,eligible_unselected_count,"
          "additional_budget,additional_selected_count,"
          "weak_policy_selected_count,"
          "weak_d_efficiency_target,baseline_weak_d_efficiency,"
          "repaired_weak_d_efficiency,baseline_weak_min_retention,"
          "repaired_weak_min_retention,baseline_global_d_efficiency,"
          "repaired_global_d_efficiency,median_factorization_error,"
          "max_factorization_error,"
          "weak_policy_weak_d_efficiency,"
          "weak_policy_weak_min_retention,"
          "weak_policy_global_d_efficiency,"
          "random_policy_weak_d_efficiency,"
          "random_policy_weak_min_retention,"
          "random_policy_global_d_efficiency,"
          "global_d_policy_weak_d_efficiency,"
          "global_d_policy_weak_min_retention,"
          "global_d_policy_global_d_efficiency\n";
  csv_ << std::setprecision(17);
}

void CtVisualRecovery::WriteCsv(
    const CtVisualRecoveryResult& result) {
  if (!csv_.is_open()) {
    return;
  }
  csv_ << result.scan_time_ns << ',' << result.image_time_ns << ','
       << result.selection_policy << ',' << random_seed_ << ','
       << static_cast<int>(result.valid) << ','
       << static_cast<int>(result.eligible) << ',' << result.state << ','
       << static_cast<int>(result.apply_requested) << ','
       << static_cast<int>(result.applied) << ',' << result.weak_rank << ','
       << result.visual_candidate_count << ','
       << result.valid_factor_count << ','
       << result.rejected_factor_count << ','
       << result.baseline_selected_count << ','
       << result.eligible_unselected_count << ','
       << result.additional_budget << ','
       << result.additional_selected_count << ','
       << result.weak_policy_selected_count << ','
       << result.weak_d_efficiency_target << ','
       << result.baseline_weak_d_efficiency << ','
       << result.repaired_weak_d_efficiency << ','
       << result.baseline_weak_min_retention << ','
       << result.repaired_weak_min_retention << ','
       << result.baseline_global_d_efficiency << ','
       << result.repaired_global_d_efficiency << ','
       << result.median_factorization_error << ','
       << result.max_factorization_error << ','
       << result.weak_policy_weak_d_efficiency << ','
       << result.weak_policy_weak_min_retention << ','
       << result.weak_policy_global_d_efficiency << ','
       << result.random_policy_weak_d_efficiency << ','
       << result.random_policy_weak_min_retention << ','
       << result.random_policy_global_d_efficiency << ','
       << result.global_d_policy_weak_d_efficiency << ','
       << result.global_d_policy_weak_min_retention << ','
       << result.global_d_policy_global_d_efficiency << '\n';
  csv_.flush();
}

}  // namespace cocolic
