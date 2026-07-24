/*
 * Coco-LIC: Continuous-Time Tightly-Coupled LiDAR-Inertial-Camera Odometry
 * using Non-Uniform B-spline
 *
 * Read-only continuous-time LiDAR localizability diagnostics.
 */

#include <degeneracy/ct_lidar_observability.h>

#include <odom/factor/analytic_diff/lidar_feature_factor.h>
#include <odom/factor/analytic_diff/rd_spline_view.h>
#include <odom/factor/analytic_diff/so3_spline_view.h>

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

namespace cocolic {
namespace {

constexpr int kFixedControlPointIndex = 3;
constexpr int kEnterConsecutiveScans = 3;
constexpr int kExitConsecutiveScans = 3;
constexpr int kPrintEveryNScans = 20;
constexpr double kExitRatioMultiplier = 2.0;
constexpr double kLineDerivativeEpsilon = 1.0e-8;
constexpr double kMinimumCharacteristicLength = 1.0;
constexpr double kMaximumCharacteristicLength = 100.0;
constexpr double kRelativeNumericalFloor = 1.0e-9;
constexpr double kAbsoluteNumericalFloor = 1.0e-12;

double Median(std::vector<double> values) {
  if (values.empty()) {
    return 1.0;
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

}  // namespace

CtLidarObservability::CtLidarObservability(
    const YAML::Node& node, Trajectory::Ptr trajectory,
    const std::string& output_prefix)
    : trajectory_(std::move(trajectory)) {
  if (!node || !node.IsMap()) {
    return;
  }

  enabled_ = node["enabled"] ? node["enabled"].as<bool>() : false;
  output_csv_ = node["output_csv"] ? node["output_csv"].as<bool>() : true;
  if (node["weak_eigenvalue_ratio"]) {
    weak_eigenvalue_ratio_ =
        std::max(node["weak_eigenvalue_ratio"].as<double>(), 1.0e-12);
  }

  // Stage 1 is deliberately read-only. Refuse an accidental estimator-write
  // request rather than silently implying that recovery has been implemented.
  if (node["apply"] && node["apply"].as<bool>()) {
    std::cerr << "[CT-LiDAR] apply=true is not available in the shadow "
                 "diagnostic stage; forcing read-only operation.\n";
  }

  if (enabled_ && output_csv_) {
    csv_.open(output_prefix + "_ct_lidar_observability.csv",
              std::ios::out | std::ios::trunc);
    if (csv_.is_open()) {
      WriteCsvHeader();
    } else {
      std::cerr << "[CT-LiDAR] Unable to open observability CSV at "
                << output_prefix << "_ct_lidar_observability.csv\n";
    }
  }
}

bool CtLidarObservability::IsTimeUsable(int64_t time_ns) const {
  if (!trajectory_ || trajectory_->knts.size() < 7) {
    return false;
  }
  return time_ns >= trajectory_->knts[trajectory_->startIdx] &&
         time_ns < trajectory_->knts.back();
}

CtLidarObservabilityResult CtLidarObservability::Analyze(
    int64_t scan_time_ns, int64_t reference_time_ns,
    const Eigen::aligned_vector<PointCorrespondence>& correspondences,
    int64_t opt_min_time_ns, int64_t opt_max_time_ns,
    bool use_correspondence_scale) {
  CtLidarObservabilityResult result;
  result.scan_time_ns = scan_time_ns;
  result.reference_time_ns = reference_time_ns;
  result.correspondence_count = static_cast<int>(correspondences.size());
  result.decision_threshold = weak_eigenvalue_ratio_;

  if (!enabled_ || !trajectory_ || correspondences.empty() ||
      !IsTimeUsable(reference_time_ns)) {
    WriteCsv(result);
    return result;
  }

  std::map<int, int> knot_to_block;
  std::vector<double> point_ranges;
  point_ranges.reserve(correspondences.size());

  const auto add_support_knots = [&](int64_t time_ns) {
    if (!IsTimeUsable(time_ns)) {
      return;
    }
    std::pair<int, double> su{-1, 0.0};
    trajectory_->GetIdxT(time_ns, su);
    if (su.first < 3) {
      return;
    }
    const int start_index = su.first - 3;
    for (int local = 0; local < 4; ++local) {
      const int knot_index = start_index + local;
      if (knot_index > kFixedControlPointIndex &&
          knot_index < static_cast<int>(trajectory_->numKnots())) {
        knot_to_block.emplace(knot_index,
                              static_cast<int>(knot_to_block.size()));
      }
    }
  };

  add_support_knots(reference_time_ns);
  for (const auto& correspondence : correspondences) {
    if (correspondence.t_point < opt_min_time_ns ||
        correspondence.t_point >= opt_max_time_ns ||
        !IsTimeUsable(correspondence.t_point)) {
      continue;
    }
    add_support_knots(correspondence.t_point);
    if (std::isfinite(correspondence.point.norm())) {
      point_ranges.push_back(correspondence.point.norm());
    }
  }

  result.free_knot_count = static_cast<int>(knot_to_block.size());
  if (knot_to_block.empty()) {
    WriteCsv(result);
    return result;
  }

  result.characteristic_length =
      std::min(kMaximumCharacteristicLength,
               std::max(kMinimumCharacteristicLength, Median(point_ranges)));

  const int state_dimension = 6 * result.free_knot_count;
  Eigen::MatrixXd lidar_information =
      Eigen::MatrixXd::Zero(state_dimension, state_dimension);

  const SO3d S_LtoI = trajectory_->GetSensorEP(LiDARSensor).so3;
  const Eigen::Vector3d p_LinI =
      trajectory_->GetSensorEP(LiDARSensor).p;
  const SO3d S_GtoM(Eigen::Quaterniond::Identity());
  const Eigen::Vector3d p_GinM = Eigen::Vector3d::Zero();

  for (const auto& correspondence : correspondences) {
    if (correspondence.t_point < opt_min_time_ns ||
        correspondence.t_point >= opt_max_time_ns ||
        !IsTimeUsable(correspondence.t_point)) {
      continue;
    }

    std::pair<int, double> su{-1, 0.0};
    trajectory_->GetIdxT(correspondence.t_point, su);
    if (su.first < 3) {
      continue;
    }
    const int start_index = su.first - 3;
    if (start_index + 3 >= static_cast<int>(trajectory_->numKnots())) {
      continue;
    }

    if (correspondence.geo_type != Plane) {
      const SE3d T_lidar =
          trajectory_->GetLidarPoseNURBS(correspondence.t_point);
      const Eigen::Vector3d point_map = T_lidar * correspondence.point;
      const double line_distance =
          ((point_map - correspondence.geo_point)
               .cross(correspondence.geo_normal))
              .norm();
      if (!std::isfinite(line_distance) ||
          line_distance <= kLineDerivativeEpsilon) {
        ++result.skipped_nondifferentiable_line_count;
        continue;
      }
    }

    const Eigen::Matrix4d blending_matrix =
        trajectory_->blending_mats[start_index];
    const Eigen::Matrix4d cumulative_blending_matrix =
        trajectory_->cumu_blending_mats[start_index];
    const double factor_weight =
        use_correspondence_scale ? correspondence.scale : 1.0;

    analytic_derivative::LoamFeatureFactorNURBS factor(
        correspondence.t_point, correspondence, su, blending_matrix,
        cumulative_blending_matrix, S_GtoM, p_GinM, S_LtoI, p_LinI,
        factor_weight);

    std::array<const double*, 8> parameters{};
    for (int local = 0; local < 4; ++local) {
      parameters[local] =
          trajectory_->getKnotSO3(start_index + local).data();
      parameters[4 + local] =
          trajectory_->getKnotPos(start_index + local).data();
    }

    std::array<std::array<double, 4>, 4> rotation_jacobians{};
    std::array<std::array<double, 3>, 4> position_jacobians{};
    std::array<double*, 8> jacobians{};
    for (int local = 0; local < 4; ++local) {
      jacobians[local] = rotation_jacobians[local].data();
      jacobians[4 + local] = position_jacobians[local].data();
    }

    double residual = 0.0;
    if (!factor.Evaluate(parameters.data(), &residual, jacobians.data()) ||
        !std::isfinite(residual)) {
      continue;
    }

    Eigen::RowVectorXd row = Eigen::RowVectorXd::Zero(state_dimension);
    for (int local = 0; local < 4; ++local) {
      const int knot_index = start_index + local;
      const auto block_iter = knot_to_block.find(knot_index);
      if (block_iter == knot_to_block.end()) {
        continue;
      }
      const int offset = 6 * block_iter->second;
      for (int axis = 0; axis < 3; ++axis) {
        row[offset + axis] = rotation_jacobians[local][axis];
        row[offset + 3 + axis] = position_jacobians[local][axis];
      }
    }
    if (!row.allFinite() || row.squaredNorm() <= kAbsoluteNumericalFloor) {
      continue;
    }
    lidar_information.noalias() += row.transpose() * row;
    ++result.evaluated_factor_count;
  }

  if (result.evaluated_factor_count < 30 ||
      lidar_information.trace() <= kAbsoluteNumericalFloor) {
    WriteCsv(result);
    return result;
  }

  std::pair<int, double> reference_su{-1, 0.0};
  trajectory_->GetIdxT(reference_time_ns, reference_su);
  if (reference_su.first < 3) {
    WriteCsv(result);
    return result;
  }
  const int reference_start = reference_su.first - 3;
  if (reference_start + 3 >= static_cast<int>(trajectory_->numKnots())) {
    WriteCsv(result);
    return result;
  }

  std::array<const double*, 4> rotation_parameters{};
  std::array<const double*, 4> position_parameters{};
  for (int local = 0; local < 4; ++local) {
    rotation_parameters[local] =
        trajectory_->getKnotSO3(reference_start + local).data();
    position_parameters[local] =
        trajectory_->getKnotPos(reference_start + local).data();
  }

  analytic_derivative::So3SplineView::JacobianStruct rotation_pose_jacobian;
  analytic_derivative::RdSplineView::JacobianStruct position_pose_jacobian;
  analytic_derivative::So3SplineView::EvaluateRpNURBS(
      reference_su, trajectory_->cumu_blending_mats[reference_start],
      rotation_parameters.data(), &rotation_pose_jacobian);
  analytic_derivative::RdSplineView::evaluateNURBS(
      reference_su, trajectory_->blending_mats[reference_start],
      position_parameters.data(), &position_pose_jacobian);

  Eigen::MatrixXd pose_jacobian =
      Eigen::MatrixXd::Zero(6, state_dimension);
  for (int local = 0; local < 4; ++local) {
    const int knot_index = reference_start + local;
    const auto block_iter = knot_to_block.find(knot_index);
    if (block_iter == knot_to_block.end()) {
      continue;
    }
    const int offset = 6 * block_iter->second;
    pose_jacobian.block<3, 3>(0, offset) =
        rotation_pose_jacobian.d_val_d_knot[local];
    pose_jacobian.block<3, 3>(3, offset + 3) =
        position_pose_jacobian.d_val_d_knot[local] *
        Eigen::Matrix3d::Identity();
  }

  Eigen::Matrix<double, 6, 6> scale =
      Eigen::Matrix<double, 6, 6>::Identity();
  scale.diagonal().head<3>().setConstant(result.characteristic_length);
  const Eigen::MatrixXd scaled_pose_jacobian = scale * pose_jacobian;

  Eigen::JacobiSVD<Eigen::MatrixXd> pose_svd(
      scaled_pose_jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const double pose_rank_floor =
      pose_svd.singularValues().size() > 0
          ? std::max(kAbsoluteNumericalFloor,
                     kRelativeNumericalFloor *
                         pose_svd.singularValues()[0])
          : kAbsoluteNumericalFloor;
  if (pose_svd.singularValues().size() < 6 ||
      !pose_svd.singularValues().allFinite() ||
      pose_svd.singularValues()[5] <= pose_rank_floor) {
    WriteCsv(result);
    return result;
  }

  result.free_knot_indices.resize(
      static_cast<size_t>(result.free_knot_count), -1);
  for (const auto& knot_block : knot_to_block) {
    result.free_knot_indices[static_cast<size_t>(knot_block.second)] =
        knot_block.first;
  }
  result.control_to_reference_pose = scaled_pose_jacobian;
  const Eigen::VectorXd inverse_pose_singular_values =
      pose_svd.singularValues().unaryExpr([](double value) {
        return 1.0 / value;
      });
  result.reference_pose_lift =
      pose_svd.matrixV() *
      inverse_pose_singular_values.asDiagonal() *
      pose_svd.matrixU().transpose();

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> knot_solver(
      lidar_information);
  if (knot_solver.info() != Eigen::Success) {
    WriteCsv(result);
    return result;
  }
  const double maximum_knot_eigenvalue =
      knot_solver.eigenvalues().maxCoeff();
  result.numerical_floor =
      std::max(kAbsoluteNumericalFloor,
               kRelativeNumericalFloor * maximum_knot_eigenvalue);
  Eigen::VectorXd inverse_knot_eigenvalues =
      knot_solver.eigenvalues().unaryExpr([&](double value) {
        return 1.0 / std::max(value, result.numerical_floor);
      });
  const Eigen::MatrixXd knot_covariance =
      knot_solver.eigenvectors() * inverse_knot_eigenvalues.asDiagonal() *
      knot_solver.eigenvectors().transpose();

  Eigen::Matrix<double, 6, 6> reference_covariance =
      pose_jacobian * knot_covariance * pose_jacobian.transpose();
  reference_covariance =
      scale * reference_covariance * scale.transpose();
  reference_covariance =
      0.5 * (reference_covariance + reference_covariance.transpose());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>>
      covariance_solver(reference_covariance);
  if (covariance_solver.info() != Eigen::Success ||
      covariance_solver.eigenvalues().minCoeff() <= 0.0) {
    WriteCsv(result);
    return result;
  }
  const double covariance_floor =
      std::max(kAbsoluteNumericalFloor,
               kRelativeNumericalFloor *
                   covariance_solver.eigenvalues().maxCoeff());
  const Eigen::Matrix<double, 6, 1> inverse_covariance_eigenvalues =
      covariance_solver.eigenvalues().unaryExpr([&](double value) {
        return 1.0 / std::max(value, covariance_floor);
      });
  const Eigen::Matrix<double, 6, 6> reference_information =
      covariance_solver.eigenvectors() *
      inverse_covariance_eigenvalues.asDiagonal() *
      covariance_solver.eigenvectors().transpose();

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>>
      reference_solver(reference_information);
  if (reference_solver.info() != Eigen::Success) {
    WriteCsv(result);
    return result;
  }

  result.eigenvalues = reference_solver.eigenvalues();
  const double maximum_reference_eigenvalue =
      result.eigenvalues.maxCoeff();
  if (!std::isfinite(maximum_reference_eigenvalue) ||
      maximum_reference_eigenvalue <= kAbsoluteNumericalFloor) {
    WriteCsv(result);
    return result;
  }
  result.relative_eigenvalues =
      result.eigenvalues / maximum_reference_eigenvalue;
  result.weak_rank =
      static_cast<int>((result.relative_eigenvalues.array() <
                        weak_eigenvalue_ratio_)
                           .count());
  result.raw_degenerate = result.weak_rank > 0;
  result.weak_basis.resize(6, result.weak_rank);
  if (result.weak_rank > 0) {
    result.weak_basis =
        reference_solver.eigenvectors().leftCols(result.weak_rank);
  }

  const bool exit_condition =
      result.relative_eigenvalues.minCoeff() >=
      kExitRatioMultiplier * weak_eigenvalue_ratio_;
  UpdatePersistentState(result.raw_degenerate, exit_condition);
  result.persistent_degenerate = persistent_degenerate_;
  result.valid = true;
  result.state = persistent_degenerate_
                     ? "degenerate_shadow"
                     : (result.raw_degenerate ? "entering" : "healthy");

  ++scan_count_;
  if (scan_count_ % kPrintEveryNScans == 0) {
    std::cout << "[CT-LiDAR] scan=" << scan_count_
              << " factors=" << result.evaluated_factor_count
              << " free_knots=" << result.free_knot_count
              << " weak_rank=" << result.weak_rank
              << " min_ratio=" << result.relative_eigenvalues.minCoeff()
              << " state=" << result.state << '\n';
  }
  WriteCsv(result);
  return result;
}

void CtLidarObservability::UpdatePersistentState(bool enter_condition,
                                                 bool exit_condition) {
  if (!persistent_degenerate_) {
    enter_count_ = enter_condition ? enter_count_ + 1 : 0;
    exit_count_ = 0;
    if (enter_count_ >= kEnterConsecutiveScans) {
      persistent_degenerate_ = true;
      enter_count_ = 0;
    }
    return;
  }

  exit_count_ = exit_condition ? exit_count_ + 1 : 0;
  enter_count_ = 0;
  if (exit_count_ >= kExitConsecutiveScans) {
    persistent_degenerate_ = false;
    exit_count_ = 0;
  }
}

void CtLidarObservability::WriteCsvHeader() {
  csv_ << "scan_time_ns,reference_time_ns,valid,state,raw_degenerate,"
          "persistent_degenerate,weak_rank,correspondence_count,"
          "evaluated_factor_count,skipped_nondifferentiable_line_count,"
          "free_knot_count,characteristic_length,numerical_floor,"
          "decision_threshold,"
          "lambda_0,lambda_1,lambda_2,lambda_3,lambda_4,lambda_5,"
          "relative_0,relative_1,relative_2,relative_3,relative_4,"
          "relative_5\n";
}

void CtLidarObservability::WriteCsv(
    const CtLidarObservabilityResult& result) {
  if (!csv_.is_open()) {
    return;
  }
  csv_ << result.scan_time_ns << ',' << result.reference_time_ns << ','
       << static_cast<int>(result.valid) << ',' << result.state << ','
       << static_cast<int>(result.raw_degenerate) << ','
       << static_cast<int>(result.persistent_degenerate) << ','
       << result.weak_rank << ',' << result.correspondence_count << ','
       << result.evaluated_factor_count << ','
       << result.skipped_nondifferentiable_line_count << ','
       << result.free_knot_count << ',' << std::setprecision(17)
       << result.characteristic_length << ',' << result.numerical_floor << ','
       << result.decision_threshold;
  for (int index = 0; index < 6; ++index) {
    csv_ << ',' << result.eigenvalues[index];
  }
  for (int index = 0; index < 6; ++index) {
    csv_ << ',' << result.relative_eigenvalues[index];
  }
  csv_ << '\n';
  csv_.flush();
}

}  // namespace cocolic
