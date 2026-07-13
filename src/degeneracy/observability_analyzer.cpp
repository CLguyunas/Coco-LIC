/*
 * Coco-LIC detector-only LiDAR observability analysis.
 */

#include <degeneracy/observability_analyzer.h>

#include <utils/sophus_utils.hpp>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

namespace cocolic
{
  namespace
  {
    template <typename T>
    T ReadValue(const YAML::Node &node, const std::string &key,
                const T &default_value)
    {
      if (node && node[key])
      {
        return node[key].as<T>();
      }
      return default_value;
    }
  } // namespace

  ObservabilityAnalyzer::ObservabilityAnalyzer(
      const YAML::Node &node, Trajectory::Ptr trajectory,
      const std::string &output_prefix)
      : trajectory_(std::move(trajectory))
  {
    enabled_ = ReadValue<bool>(node, "enabled", false);
    output_csv_ = ReadValue<bool>(node, "output_csv", true);
    use_correspondence_scale_ =
        ReadValue<bool>(node, "use_correspondence_scale", true);
    min_correspondences_ =
        std::max(1, ReadValue<int>(node, "min_correspondences", 30));
    analyze_every_n_scans_ =
        std::max(1, ReadValue<int>(node, "analyze_every_n_scans", 1));
    print_every_n_scans_ =
        std::max(1, ReadValue<int>(node, "print_every_n_scans", 20));
    relative_eigenvalue_threshold_ =
        std::max(0.0, ReadValue<double>(
                          node, "relative_eigenvalue_threshold", 1e-3));
    enter_relative_eigenvalue_threshold_ =
        std::max(0.0, ReadValue<double>(
                          node, "enter_relative_eigenvalue_threshold", 3e-3));
    exit_relative_eigenvalue_threshold_ =
        std::max(enter_relative_eigenvalue_threshold_,
                 ReadValue<double>(
                     node, "exit_relative_eigenvalue_threshold", 6e-3));
    enter_consecutive_scans_ =
        std::max(1, ReadValue<int>(node, "enter_consecutive_scans", 10));
    exit_consecutive_scans_ =
        std::max(1, ReadValue<int>(node, "exit_consecutive_scans", 10));
    min_characteristic_range_ =
        std::max(1e-3, ReadValue<double>(
                           node, "min_characteristic_range", 1.0));
    max_characteristic_range_ =
        std::max(min_characteristic_range_,
                 ReadValue<double>(node, "max_characteristic_range", 100.0));

    degeneracy_hysteresis_.Configure(
        enter_relative_eigenvalue_threshold_,
        exit_relative_eigenvalue_threshold_, enter_consecutive_scans_,
        exit_consecutive_scans_);

    if (!enabled_)
    {
      return;
    }

    csv_path_ = output_prefix + "_dso_observability.csv";
    if (output_csv_)
    {
      csv_stream_.open(csv_path_, std::ios::out | std::ios::trunc);
      if (!csv_stream_.is_open())
      {
        std::cerr << "[DSO-DetectOnly] Cannot open CSV: " << csv_path_
                  << ". Console diagnostics remain enabled.\n";
      }
      else
      {
        WriteCsvHeader();
      }
    }

    std::cout << "[DSO-DetectOnly] enabled | min_corr="
              << min_correspondences_ << " | hard_threshold="
              << relative_eigenvalue_threshold_ << " | enter/exit="
              << enter_relative_eigenvalue_threshold_ << "/"
              << exit_relative_eigenvalue_threshold_ << " | persistence="
              << enter_consecutive_scans_ << "/" << exit_consecutive_scans_
              << " | csv="
              << (csv_stream_.is_open() ? csv_path_ : std::string("disabled"))
              << std::endl;
  }

  ObservabilityAnalyzer::~ObservabilityAnalyzer()
  {
    if (csv_stream_.is_open())
    {
      csv_stream_.flush();
      csv_stream_.close();
    }
  }

  ObservabilityResult ObservabilityAnalyzer::AnalyzeAndLog(
      int64_t scan_timestamp_ns,
      const Eigen::aligned_vector<PointCorrespondence> &point_corrs)
  {
    if (!enabled_)
    {
      return last_result_;
    }

    ++scan_counter_;
    if ((scan_counter_ - 1) % static_cast<size_t>(analyze_every_n_scans_) != 0)
    {
      return last_result_;
    }

    last_result_ = Analyze(scan_timestamp_ns, point_corrs);
    std::array<double, 6> relative_eigenvalues;
    for (int i = 0; i < 6; ++i)
    {
      relative_eigenvalues[static_cast<size_t>(i)] =
          last_result_.relative_eigenvalues[i];
    }
    const DegeneracyDecision decision = degeneracy_hysteresis_.Update(
        last_result_.valid, relative_eigenvalues);
    last_result_.candidate_weak_direction_num =
        decision.candidate_weak_direction_num;
    last_result_.degeneracy_score = decision.degeneracy_score;
    last_result_.degenerate_state = decision.degenerate_state;
    last_result_.enter_counter = decision.enter_counter;
    last_result_.exit_counter = decision.exit_counter;
    WriteCsvRow(last_result_);

    if (scan_counter_ == 1 ||
        scan_counter_ % static_cast<size_t>(print_every_n_scans_) == 0)
    {
      PrintSummary(last_result_);
    }
    return last_result_;
  }

  ObservabilityResult ObservabilityAnalyzer::Analyze(
      int64_t scan_timestamp_ns,
      const Eigen::aligned_vector<PointCorrespondence> &point_corrs) const
  {
    ObservabilityResult result;
    result.scan_timestamp_ns = scan_timestamp_ns;
    result.characteristic_range = ComputeCharacteristicRange(point_corrs);

    Eigen::Matrix<double, 6, 6> information =
        Eigen::Matrix<double, 6, 6>::Zero();

    for (const auto &pc : point_corrs)
    {
      Eigen::Matrix<double, 1, 6> jacobian;
      if (!BuildPoseJacobian(pc, result.characteristic_range, jacobian))
      {
        continue;
      }

      double factor_weight = 1.0;
      if (use_correspondence_scale_)
      {
        factor_weight = std::max(0.0, pc.scale);
      }
      const Eigen::Matrix<double, 1, 6> weighted_jacobian =
          factor_weight * jacobian;
      information.noalias() +=
          weighted_jacobian.transpose() * weighted_jacobian;

      ++result.correspondence_num;
      if (pc.geo_type == GeometryType::Plane)
      {
        ++result.plane_num;
      }
      else
      {
        ++result.line_num;
      }
    }

    if (result.correspondence_num <
        static_cast<size_t>(min_correspondences_))
    {
      return result;
    }

    information = 0.5 * (information + information.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(
        information);
    if (solver.info() != Eigen::Success)
    {
      return result;
    }

    result.eigenvalues = solver.eigenvalues().cwiseMax(0.0);
    result.eigenvectors = solver.eigenvectors();

    const double max_eigenvalue = result.eigenvalues.maxCoeff();
    if (!std::isfinite(max_eigenvalue) || max_eigenvalue <= 1e-12)
    {
      return result;
    }

    result.relative_eigenvalues = result.eigenvalues / max_eigenvalue;
    result.weak_direction_num = 0;
    for (int i = 0; i < 6; ++i)
    {
      if (result.relative_eigenvalues[i] < relative_eigenvalue_threshold_)
      {
        ++result.weak_direction_num;
      }
    }

    const double min_eigenvalue = result.eigenvalues.minCoeff();
    result.condition_number =
        max_eigenvalue / std::max(min_eigenvalue, 1e-12);
    result.valid = true;
    return result;
  }

  double ObservabilityAnalyzer::ComputeCharacteristicRange(
      const Eigen::aligned_vector<PointCorrespondence> &point_corrs) const
  {
    std::vector<double> ranges;
    ranges.reserve(point_corrs.size());
    for (const auto &pc : point_corrs)
    {
      const double range = pc.point.norm();
      if (std::isfinite(range) && range > 1e-3)
      {
        ranges.push_back(range);
      }
    }

    if (ranges.empty())
    {
      return min_characteristic_range_;
    }

    const size_t mid = ranges.size() / 2;
    std::nth_element(ranges.begin(), ranges.begin() + mid, ranges.end());
    double median = ranges[mid];
    if (ranges.size() % 2 == 0 && mid > 0)
    {
      const auto lower_max =
          std::max_element(ranges.begin(), ranges.begin() + mid);
      median = 0.5 * (median + *lower_max);
    }

    return std::max(min_characteristic_range_,
                    std::min(max_characteristic_range_, median));
  }

  bool ObservabilityAnalyzer::BuildPoseJacobian(
      const PointCorrespondence &pc, double characteristic_range,
      Eigen::Matrix<double, 1, 6> &jacobian) const
  {
    jacobian.setZero();
    if (!trajectory_ || pc.t_point < 0 || characteristic_range <= 1e-12)
    {
      return false;
    }

    const SE3d lidar_pose = trajectory_->GetLidarPoseNURBS(pc.t_point);
    const Eigen::Matrix3d rotation = lidar_pose.so3().matrix();
    const Eigen::Vector3d point_in_map =
        rotation * pc.point + lidar_pose.translation();

    Eigen::Matrix<double, 1, 3> position_jacobian;
    if (pc.geo_type == GeometryType::Plane)
    {
      const Eigen::Vector3d normal = pc.geo_plane.head<3>();
      const double normal_norm = normal.norm();
      if (!std::isfinite(normal_norm) || normal_norm <= 1e-12)
      {
        return false;
      }
      position_jacobian = (normal / normal_norm).transpose();
    }
    else
    {
      const double direction_norm = pc.geo_normal.norm();
      if (!std::isfinite(direction_norm) || direction_norm <= 1e-12)
      {
        return false;
      }
      const Eigen::Vector3d direction = pc.geo_normal / direction_norm;
      const Eigen::Vector3d distance_vector =
          (point_in_map - pc.geo_point).cross(direction);
      const double distance = distance_vector.norm();
      if (!std::isfinite(distance) || distance <= 1e-12)
      {
        return false;
      }
      position_jacobian =
          -distance_vector.transpose() / distance * SO3d::hat(direction);
    }

    // Use a common map-frame perturbation for every point timestamp:
    // p' = Exp(dtheta_map) R p + t + dt_map. This makes it valid to accumulate
    // all per-point Jacobians in one information matrix even when the sensor
    // rotates during a scan. Rotation columns are divided by a representative
    // scene range so the eigenspectrum does not compare radians with metres.
    const Eigen::Vector3d rotated_point = rotation * pc.point;
    jacobian.block<1, 3>(0, 0) =
        position_jacobian * (-SO3d::hat(rotated_point)) /
        characteristic_range;
    jacobian.block<1, 3>(0, 3) = position_jacobian;

    return jacobian.allFinite();
  }

  void ObservabilityAnalyzer::WriteCsvHeader()
  {
    if (!csv_stream_.is_open())
    {
      return;
    }

    csv_stream_ << "scan_timestamp_s,valid,correspondence_num,plane_num,line_num,"
                   "characteristic_range,condition_number,weak_direction_num,"
                   "candidate_weak_direction_num,degeneracy_score,"
                   "degenerate_state,enter_counter,exit_counter";
    for (int i = 0; i < 6; ++i)
    {
      csv_stream_ << ",lambda_" << i;
    }
    for (int i = 0; i < 6; ++i)
    {
      csv_stream_ << ",relative_lambda_" << i;
    }
    const char *state_names[6] = {"rx_scaled", "ry_scaled", "rz_scaled",
                                  "tx", "ty", "tz"};
    for (int eigen_idx = 0; eigen_idx < 6; ++eigen_idx)
    {
      for (int state_idx = 0; state_idx < 6; ++state_idx)
      {
        csv_stream_ << ",v" << eigen_idx << "_" << state_names[state_idx];
      }
    }
    csv_stream_ << '\n';
    csv_stream_.flush();
  }

  void ObservabilityAnalyzer::WriteCsvRow(
      const ObservabilityResult &result)
  {
    if (!csv_stream_.is_open())
    {
      return;
    }

    csv_stream_ << std::setprecision(12)
                << result.scan_timestamp_ns * Trajectory::NS_TO_S << ','
                << static_cast<int>(result.valid) << ','
                << result.correspondence_num << ',' << result.plane_num << ','
                << result.line_num << ',' << result.characteristic_range << ','
                << result.condition_number << ',' << result.weak_direction_num
                << ',' << result.candidate_weak_direction_num << ','
                << result.degeneracy_score << ','
                << static_cast<int>(result.degenerate_state) << ','
                << result.enter_counter << ',' << result.exit_counter;
    for (int i = 0; i < 6; ++i)
    {
      csv_stream_ << ',' << result.eigenvalues[i];
    }
    for (int i = 0; i < 6; ++i)
    {
      csv_stream_ << ',' << result.relative_eigenvalues[i];
    }
    for (int eigen_idx = 0; eigen_idx < 6; ++eigen_idx)
    {
      for (int state_idx = 0; state_idx < 6; ++state_idx)
      {
        csv_stream_ << ',' << result.eigenvectors(state_idx, eigen_idx);
      }
    }
    csv_stream_ << '\n';
    csv_stream_.flush();
  }

  void ObservabilityAnalyzer::PrintSummary(
      const ObservabilityResult &result) const
  {
    std::cout << "[DSO-DetectOnly] t="
              << result.scan_timestamp_ns * Trajectory::NS_TO_S
              << " s | valid=" << result.valid
              << " | corr=" << result.correspondence_num
              << " | weak=" << result.weak_direction_num
              << " | candidate=" << result.candidate_weak_direction_num
              << " | score=" << result.degeneracy_score
              << " | state=" << static_cast<int>(result.degenerate_state)
              << " | enter/exit=" << result.enter_counter << "/"
              << result.exit_counter
              << " | rel_eigs="
              << result.relative_eigenvalues.transpose() << std::endl;
  }

} // namespace cocolic
