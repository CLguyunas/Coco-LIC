/*
 * Coco-LIC detector-only LiDAR observability analysis.
 */

#include <degeneracy/observability_analyzer.h>

#include <odom/factor/analytic_diff/so3_spline_view.h>
#include <utils/sophus_utils.hpp>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
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

    const char *DegeneracyCauseName(int cause)
    {
      switch (cause)
      {
      case 0:
        return "healthy";
      case 1:
        return "environment";
      case 2:
        return "spline_support";
      case 3:
        return "coupled";
      default:
        return "invalid";
      }
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

    support_enabled_ =
        ReadValue<bool>(node, "support_enabled", true);
    support_reference_samples_per_interval_ = std::max(
        8, ReadValue<int>(node,
                          "support_reference_samples_per_interval", 32));
    support_max_control_points_ = std::max(
        4, ReadValue<int>(node, "support_max_control_points", 32));
    support_enter_quality_threshold_ = std::max(
        0.0, ReadValue<double>(
                 node, "support_enter_quality_threshold", 2e-2));
    support_exit_quality_threshold_ = std::max(
        support_enter_quality_threshold_,
        ReadValue<double>(node, "support_exit_quality_threshold", 5e-2));
    support_enter_consecutive_scans_ = std::max(
        1, ReadValue<int>(node, "support_enter_consecutive_scans", 10));
    support_exit_consecutive_scans_ = std::max(
        1, ReadValue<int>(node, "support_exit_consecutive_scans", 10));

    const YAML::Node injection_node =
        node ? node["support_injection"] : YAML::Node();
    const bool injection_requested =
        ReadValue<bool>(injection_node, "enabled", false);
    const bool diagnostics_only =
        ReadValue<bool>(injection_node, "diagnostics_only", true);
    support_injection_output_csv_ =
        ReadValue<bool>(injection_node, "output_csv", true);
    const std::string injection_mode_name = ReadValue<std::string>(
        injection_node, "mode", "timestamp_compression");
    SupportInjectionMode injection_mode = SupportInjectionMode::Disabled;
    const bool injection_mode_valid =
        SupportDegradationInjector::ParseMode(injection_mode_name,
                                               injection_mode);
    SupportInjectionConfig injection_config;
    injection_config.enabled = injection_requested && diagnostics_only &&
                               support_enabled_ && injection_mode_valid;
    injection_config.mode = injection_mode;
    injection_config.severity =
        ReadValue<double>(injection_node, "severity", 0.5);
    injection_config.phase_start =
        ReadValue<double>(injection_node, "phase_start", 0.0);
    injection_config.phase_end =
        ReadValue<double>(injection_node, "phase_end", 1.0);
    injection_config.random_seed =
        ReadValue<uint64_t>(injection_node, "random_seed", 42);
    support_injector_.Configure(injection_config);
    support_injection_enabled_ = support_injector_.Enabled();

    degeneracy_hysteresis_.Configure(
        enter_relative_eigenvalue_threshold_,
        exit_relative_eigenvalue_threshold_, enter_consecutive_scans_,
        exit_consecutive_scans_);
    support_hysteresis_.Configure(
        support_enter_quality_threshold_, support_exit_quality_threshold_,
        support_enter_consecutive_scans_, support_exit_consecutive_scans_);
    injected_support_hysteresis_.Configure(
        support_enter_quality_threshold_, support_exit_quality_threshold_,
        support_enter_consecutive_scans_, support_exit_consecutive_scans_);

    if (!enabled_)
    {
      return;
    }

    if (injection_requested && !diagnostics_only)
    {
      std::cerr << "[DSO-DetectOnly] support_injection refused: "
                   "diagnostics_only must remain true.\n";
    }
    if (injection_requested && !support_enabled_)
    {
      std::cerr << "[DSO-DetectOnly] support_injection disabled because "
                   "support_enabled is false.\n";
    }
    if (injection_requested && !injection_mode_valid)
    {
      std::cerr << "[DSO-DetectOnly] support_injection disabled: unknown "
                << "mode '" << injection_mode_name << "'.\n";
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

    injection_csv_path_ = output_prefix + "_dso_support_injection.csv";
    if (support_injection_enabled_ && support_injection_output_csv_)
    {
      injection_csv_stream_.open(injection_csv_path_,
                                 std::ios::out | std::ios::trunc);
      if (!injection_csv_stream_.is_open())
      {
        std::cerr << "[DSO-DetectOnly] Cannot open injection CSV: "
                  << injection_csv_path_
                  << ". Console injection diagnostics remain enabled.\n";
      }
      else
      {
        WriteInjectionCsvHeader();
      }
    }

    std::cout << "[DSO-DetectOnly] enabled | min_corr="
              << min_correspondences_ << " | hard_threshold="
              << relative_eigenvalue_threshold_ << " | enter/exit="
              << enter_relative_eigenvalue_threshold_ << "/"
              << exit_relative_eigenvalue_threshold_ << " | persistence="
              << enter_consecutive_scans_ << "/" << exit_consecutive_scans_
              << " | spline_support=" << support_enabled_
              << " | support_enter/exit="
              << support_enter_quality_threshold_ << "/"
              << support_exit_quality_threshold_
              << " | support_injection=" << support_injection_enabled_;
    if (support_injection_enabled_)
    {
      const SupportInjectionConfig &config = support_injector_.Config();
      std::cout << "(" << SupportDegradationInjector::ModeName(config.mode)
                << ",severity=" << config.severity << ",phase="
                << config.phase_start << "-" << config.phase_end << ")"
                << " | injection_csv="
                << (injection_csv_stream_.is_open()
                        ? injection_csv_path_
                        : std::string("disabled"));
    }
    std::cout << " | csv="
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
    if (injection_csv_stream_.is_open())
    {
      injection_csv_stream_.flush();
      injection_csv_stream_.close();
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

    std::vector<WeightedTimestamp> weighted_timestamps;
    last_result_ =
        Analyze(scan_timestamp_ns, point_corrs, &weighted_timestamps);
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

    if (support_enabled_)
    {
      const DegeneracyDecision support_decision = support_hysteresis_.Update(
          last_result_.support_valid, last_result_.support_quality_min,
          last_result_.support_weak_direction_num);
      last_result_.support_weak_direction_num =
          support_decision.candidate_weak_direction_num;
      last_result_.support_score = support_decision.degeneracy_score;
      last_result_.support_degenerate_state =
          support_decision.degenerate_state;
      last_result_.support_enter_counter = support_decision.enter_counter;
      last_result_.support_exit_counter = support_decision.exit_counter;
    }

    if (last_result_.valid &&
        (!support_enabled_ || last_result_.support_valid))
    {
      last_result_.degeneracy_cause =
          (last_result_.degenerate_state ? 1 : 0) +
          (support_enabled_ && last_result_.support_degenerate_state ? 2 : 0);
    }
    WriteCsvRow(last_result_);

    if (support_injection_enabled_)
    {
      AnalyzeInjectedSupport(weighted_timestamps, last_result_);
      WriteInjectionCsvRow(scan_timestamp_ns, last_result_,
                           last_injection_result_);
    }

    if (scan_counter_ == 1 ||
        scan_counter_ % static_cast<size_t>(print_every_n_scans_) == 0)
    {
      PrintSummary(last_result_);
    }
    return last_result_;
  }

  ObservabilityResult ObservabilityAnalyzer::Analyze(
      int64_t scan_timestamp_ns,
      const Eigen::aligned_vector<PointCorrespondence> &point_corrs,
      std::vector<WeightedTimestamp> *weighted_timestamps) const
  {
    ObservabilityResult result;
    result.scan_timestamp_ns = scan_timestamp_ns;
    result.characteristic_range = ComputeCharacteristicRange(point_corrs);

    Eigen::Matrix<double, 6, 6> information =
        Eigen::Matrix<double, 6, 6>::Zero();
    std::vector<WeightedTimestamp> local_weighted_timestamps;
    std::vector<WeightedTimestamp> &support_timestamps =
        weighted_timestamps ? *weighted_timestamps
                            : local_weighted_timestamps;
    support_timestamps.clear();
    support_timestamps.reserve(point_corrs.size());

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
      if (factor_weight > 0.0)
      {
        support_timestamps.emplace_back(pc.t_point, factor_weight);
      }

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
    if (support_enabled_)
    {
      AnalyzeSplineSupport(support_timestamps, result);
    }
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

  void ObservabilityAnalyzer::AnalyzeSplineSupport(
      const std::vector<WeightedTimestamp> &weighted_timestamps,
      ObservabilityResult &result) const
  {
    if (!trajectory_ || weighted_timestamps.size() <
                            static_cast<size_t>(min_correspondences_))
    {
      return;
    }

    struct SupportSample
    {
      int64_t timestamp_ns = 0;
      int interval_index = -1;
      int control_start_index = -1;
      double u = 0.0;
      double squared_weight = 0.0;
    };

    const auto &knot_times = trajectory_->knts;
    if (knot_times.size() < 7)
    {
      return;
    }

    std::vector<SupportSample> samples;
    samples.reserve(weighted_timestamps.size());
    std::set<int> active_intervals;
    int min_interval_index = std::numeric_limits<int>::max();
    int max_interval_index = -1;
    int min_control_index = std::numeric_limits<int>::max();
    int max_control_index = -1;
    int64_t min_timestamp_ns = std::numeric_limits<int64_t>::max();
    int64_t max_timestamp_ns = std::numeric_limits<int64_t>::min();

    for (const auto &weighted_timestamp : weighted_timestamps)
    {
      const int64_t timestamp_ns = weighted_timestamp.first;
      const double weight = weighted_timestamp.second;
      if (!std::isfinite(weight) || weight <= 0.0 ||
          timestamp_ns < knot_times.front() ||
          timestamp_ns >= knot_times.back())
      {
        continue;
      }

      const auto upper =
          std::upper_bound(knot_times.begin(), knot_times.end(), timestamp_ns);
      if (upper == knot_times.begin() || upper == knot_times.end())
      {
        continue;
      }

      const int interval_index =
          static_cast<int>(std::distance(knot_times.begin(), upper)) - 1;
      const int control_start_index = interval_index - 3;
      if (control_start_index < 0 ||
          control_start_index + 3 >=
              static_cast<int>(trajectory_->numKnots()) ||
          control_start_index >=
              static_cast<int>(trajectory_->blending_mats.size()) ||
          control_start_index >=
              static_cast<int>(trajectory_->cumu_blending_mats.size()))
      {
        continue;
      }

      const int64_t interval_ns =
          knot_times[static_cast<size_t>(interval_index + 1)] -
          knot_times[static_cast<size_t>(interval_index)];
      if (interval_ns <= 0)
      {
        continue;
      }

      SupportSample sample;
      sample.timestamp_ns = timestamp_ns;
      sample.interval_index = interval_index;
      sample.control_start_index = control_start_index;
      sample.u = static_cast<double>(
                     timestamp_ns -
                     knot_times[static_cast<size_t>(interval_index)]) /
                 static_cast<double>(interval_ns);
      sample.squared_weight = weight * weight;
      samples.push_back(sample);

      active_intervals.insert(interval_index);
      min_interval_index = std::min(min_interval_index, interval_index);
      max_interval_index = std::max(max_interval_index, interval_index);
      min_control_index = std::min(min_control_index, control_start_index);
      max_control_index = std::max(max_control_index,
                                   control_start_index + SplineOrder - 1);
      min_timestamp_ns = std::min(min_timestamp_ns, timestamp_ns);
      max_timestamp_ns = std::max(max_timestamp_ns, timestamp_ns);
    }

    if (samples.size() < static_cast<size_t>(min_correspondences_) ||
        active_intervals.empty() || min_control_index > max_control_index)
    {
      return;
    }

    const int control_point_num = max_control_index - min_control_index + 1;
    if (control_point_num < SplineOrder ||
        control_point_num > support_max_control_points_)
    {
      return;
    }

    const int dimension = 6 * control_point_num;
    Eigen::MatrixXd observed_information =
        Eigen::MatrixXd::Zero(dimension, dimension);
    Eigen::MatrixXd reference_information =
        Eigen::MatrixXd::Zero(dimension, dimension);

    using SO3View = analytic_derivative::So3SplineView;
    const auto accumulate_mapping =
        [&](int interval_index, int control_start_index, double u,
            double sample_weight, Eigen::MatrixXd &information) -> bool
    {
      if (!std::isfinite(u) || u < 0.0 || u > 1.0 ||
          !std::isfinite(sample_weight) || sample_weight <= 0.0)
      {
        return false;
      }

      std::array<const double *, SplineOrder> rotation_knots;
      for (int i = 0; i < SplineOrder; ++i)
      {
        rotation_knots[static_cast<size_t>(i)] =
            trajectory_->getKnotSO3(
                static_cast<size_t>(control_start_index + i))
                .data();
      }

      typename SO3View::JacobianStruct rotation_jacobian;
      SO3View::EvaluateRpNURBS(
          std::make_pair(interval_index, u),
          trajectory_->cumu_blending_mats[
              static_cast<size_t>(control_start_index)],
          rotation_knots.data(), &rotation_jacobian);

      Eigen::Vector4d polynomial;
      polynomial << 1.0, u, u * u, u * u * u;
      const Eigen::Vector4d position_coefficients =
          trajectory_->blending_mats[
              static_cast<size_t>(control_start_index)] *
          polynomial;
      if (!position_coefficients.allFinite())
      {
        return false;
      }

      const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
      for (int i = 0; i < SplineOrder; ++i)
      {
        if (!rotation_jacobian.d_val_d_knot[static_cast<size_t>(i)]
                 .allFinite())
        {
          return false;
        }
        const int local_i = control_start_index + i - min_control_index;
        const int rotation_i = 6 * local_i;
        const int position_i = rotation_i + 3;
        for (int j = 0; j < SplineOrder; ++j)
        {
          const int local_j = control_start_index + j - min_control_index;
          const int rotation_j = 6 * local_j;
          const int position_j = rotation_j + 3;
          information.block<3, 3>(rotation_i, rotation_j).noalias() +=
              sample_weight *
              rotation_jacobian.d_val_d_knot[static_cast<size_t>(i)]
                  .transpose() *
              rotation_jacobian.d_val_d_knot[static_cast<size_t>(j)];
          information.block<3, 3>(position_i, position_j) +=
              sample_weight * position_coefficients[i] *
              position_coefficients[j] * identity;
        }
      }
      return true;
    };

    double observed_weight_sum = 0.0;
    for (const auto &sample : samples)
    {
      if (accumulate_mapping(sample.interval_index,
                             sample.control_start_index, sample.u,
                             sample.squared_weight, observed_information))
      {
        observed_weight_sum += sample.squared_weight;
      }
    }

    double reference_weight_sum = 0.0;
    double min_knot_dt_s = std::numeric_limits<double>::infinity();
    double max_knot_dt_s = 0.0;
    for (int interval_index = min_interval_index;
         interval_index <= max_interval_index; ++interval_index)
    {
      const int control_start_index = interval_index - 3;
      if (control_start_index < 0 ||
          control_start_index >=
              static_cast<int>(trajectory_->blending_mats.size()) ||
          control_start_index >=
              static_cast<int>(trajectory_->cumu_blending_mats.size()))
      {
        return;
      }
      const double interval_dt_s =
          static_cast<double>(
              knot_times[static_cast<size_t>(interval_index + 1)] -
              knot_times[static_cast<size_t>(interval_index)]) *
          Trajectory::NS_TO_S;
      if (!std::isfinite(interval_dt_s) || interval_dt_s <= 0.0)
      {
        return;
      }
      min_knot_dt_s = std::min(min_knot_dt_s, interval_dt_s);
      max_knot_dt_s = std::max(max_knot_dt_s, interval_dt_s);

      const double reference_sample_weight =
          interval_dt_s /
          static_cast<double>(support_reference_samples_per_interval_);
      for (int sample_index = 0;
           sample_index < support_reference_samples_per_interval_;
           ++sample_index)
      {
        const double u =
            (static_cast<double>(sample_index) + 0.5) /
            static_cast<double>(support_reference_samples_per_interval_);
        if (accumulate_mapping(interval_index, control_start_index, u,
                               reference_sample_weight,
                               reference_information))
        {
          reference_weight_sum += reference_sample_weight;
        }
      }
    }

    if (!std::isfinite(observed_weight_sum) || observed_weight_sum <= 0.0 ||
        !std::isfinite(reference_weight_sum) || reference_weight_sum <= 0.0)
    {
      return;
    }

    observed_information /= observed_weight_sum;
    reference_information /= reference_weight_sum;
    observed_information =
        0.5 * (observed_information + observed_information.transpose());
    reference_information =
        0.5 * (reference_information + reference_information.transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> reference_solver(
        reference_information);
    if (reference_solver.info() != Eigen::Success)
    {
      return;
    }
    const Eigen::VectorXd reference_eigenvalues =
        reference_solver.eigenvalues();
    const double reference_max = reference_eigenvalues.maxCoeff();
    const double reference_floor = reference_max * 1e-10;
    if (!std::isfinite(reference_max) || reference_max <= 1e-12 ||
        reference_eigenvalues.minCoeff() <= reference_floor)
    {
      return;
    }

    const Eigen::VectorXd inverse_sqrt_reference =
        reference_eigenvalues.array().sqrt().inverse();
    const Eigen::MatrixXd whitening =
        reference_solver.eigenvectors() *
        inverse_sqrt_reference.asDiagonal() *
        reference_solver.eigenvectors().transpose();
    Eigen::MatrixXd quality_information =
        whitening * observed_information * whitening.transpose();
    quality_information =
        0.5 * (quality_information + quality_information.transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> quality_solver(
        quality_information);
    if (quality_solver.info() != Eigen::Success)
    {
      return;
    }

    const Eigen::VectorXd quality_eigenvalues =
        quality_solver.eigenvalues().cwiseMax(0.0);
    const double min_quality = quality_eigenvalues.minCoeff();
    const double max_quality = quality_eigenvalues.maxCoeff();
    if (!std::isfinite(min_quality) || !std::isfinite(max_quality) ||
        max_quality <= 1e-12)
    {
      return;
    }

    int weak_direction_num = 0;
    for (int i = 0; i < quality_eigenvalues.size(); ++i)
    {
      if (quality_eigenvalues[i] < support_enter_quality_threshold_)
      {
        ++weak_direction_num;
      }
    }

    Eigen::VectorXd weakest_mode =
        whitening * quality_solver.eigenvectors().col(0);
    const double weakest_mode_norm = weakest_mode.norm();
    if (!std::isfinite(weakest_mode_norm) || weakest_mode_norm <= 1e-12)
    {
      return;
    }
    weakest_mode /= weakest_mode_norm;

    double rotation_energy = 0.0;
    double boundary_energy = 0.0;
    double max_knot_energy = -1.0;
    int weakest_knot_index = -1;
    for (int local_index = 0; local_index < control_point_num; ++local_index)
    {
      const int offset = 6 * local_index;
      const double knot_rotation_energy =
          weakest_mode.segment<3>(offset).squaredNorm();
      const double knot_position_energy =
          weakest_mode.segment<3>(offset + 3).squaredNorm();
      const double knot_energy =
          knot_rotation_energy + knot_position_energy;
      rotation_energy += knot_rotation_energy;
      if (local_index == 0 || local_index == control_point_num - 1)
      {
        boundary_energy += knot_energy;
      }
      if (knot_energy > max_knot_energy)
      {
        max_knot_energy = knot_energy;
        weakest_knot_index = min_control_index + local_index;
      }
    }

    result.support_valid = true;
    result.support_control_point_num = control_point_num;
    result.support_interval_num =
        max_interval_index - min_interval_index + 1;
    result.support_dimension = dimension;
    result.support_effective_rank = dimension - weak_direction_num;
    result.support_weak_direction_num = weak_direction_num;
    result.support_time_span_s =
        static_cast<double>(max_timestamp_ns - min_timestamp_ns) *
        Trajectory::NS_TO_S;
    result.support_min_knot_dt_s = min_knot_dt_s;
    result.support_max_knot_dt_s = max_knot_dt_s;
    result.support_quality_min = min_quality;
    result.support_condition_number =
        max_quality / std::max(min_quality, 1e-12);
    result.support_weakest_knot_index = weakest_knot_index;
    result.support_weakest_knot_energy_ratio =
        std::max(0.0, max_knot_energy);
    result.support_weakest_rotation_ratio = rotation_energy;
    result.support_boundary_energy_ratio = boundary_energy;
  }

  void ObservabilityAnalyzer::AnalyzeInjectedSupport(
      const std::vector<WeightedTimestamp> &weighted_timestamps,
      const ObservabilityResult &original_result)
  {
    last_injection_result_ = SupportInjectionDiagnostic();
    std::vector<WeightedTimestamp> injected_timestamps;
    last_injection_result_.metadata =
        support_injector_.Apply(weighted_timestamps, injected_timestamps);

    ObservabilityResult &injected =
        last_injection_result_.injected_support;
    injected.scan_timestamp_ns = original_result.scan_timestamp_ns;
    AnalyzeSplineSupport(injected_timestamps, injected);

    const DegeneracyDecision decision = injected_support_hysteresis_.Update(
        injected.support_valid, injected.support_quality_min,
        injected.support_weak_direction_num);
    injected.support_weak_direction_num =
        decision.candidate_weak_direction_num;
    injected.support_score = decision.degeneracy_score;
    injected.support_degenerate_state = decision.degenerate_state;
    injected.support_enter_counter = decision.enter_counter;
    injected.support_exit_counter = decision.exit_counter;

    if (original_result.valid && injected.support_valid)
    {
      last_injection_result_.degeneracy_cause =
          (original_result.degenerate_state ? 1 : 0) +
          (injected.support_degenerate_state ? 2 : 0);
    }
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
    // New stage-2 fields are appended so every stage-1 column retains its
    // original index as well as its name.
    csv_stream_ << ",support_valid,support_control_point_num,"
                   "support_interval_num,support_dimension,"
                   "support_effective_rank,support_weak_direction_num,"
                   "support_time_span_s,support_min_knot_dt_s,"
                   "support_max_knot_dt_s,support_quality_min,"
                   "support_condition_number,support_score,"
                   "support_degenerate_state,support_enter_counter,"
                   "support_exit_counter,support_weakest_knot_index,"
                   "support_weakest_knot_energy_ratio,"
                   "support_weakest_rotation_ratio,"
                   "support_boundary_energy_ratio,degeneracy_cause";
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
    csv_stream_ << ',' << static_cast<int>(result.support_valid) << ','
                << result.support_control_point_num << ','
                << result.support_interval_num << ','
                << result.support_dimension << ','
                << result.support_effective_rank << ','
                << result.support_weak_direction_num << ','
                << result.support_time_span_s << ','
                << result.support_min_knot_dt_s << ','
                << result.support_max_knot_dt_s << ','
                << result.support_quality_min << ','
                << result.support_condition_number << ','
                << result.support_score << ','
                << static_cast<int>(result.support_degenerate_state) << ','
                << result.support_enter_counter << ','
                << result.support_exit_counter << ','
                << result.support_weakest_knot_index << ','
                << result.support_weakest_knot_energy_ratio << ','
                << result.support_weakest_rotation_ratio << ','
                << result.support_boundary_energy_ratio << ','
                << result.degeneracy_cause;
    csv_stream_ << '\n';
    csv_stream_.flush();
  }

  void ObservabilityAnalyzer::WriteInjectionCsvHeader()
  {
    if (!injection_csv_stream_.is_open())
    {
      return;
    }

    injection_csv_stream_
        << "scan_timestamp_s,injection_mode_code,injection_mode,"
           "injection_severity,phase_start,phase_end,random_seed,"
           "injection_applied,input_sample_num,selected_sample_num,"
           "modified_sample_num,removed_sample_num,output_sample_num,"
           "retained_ratio,timestamp_span_ratio,environment_valid,"
           "environment_relative_lambda_0,environment_score,"
           "environment_state,original_support_valid,"
           "original_support_quality_min,original_support_weak_direction_num,"
           "original_support_state,original_degeneracy_cause,"
           "injected_support_valid,injected_support_control_point_num,"
           "injected_support_interval_num,injected_support_dimension,"
           "injected_support_effective_rank,"
           "injected_support_weak_direction_num,"
           "injected_support_time_span_s,injected_support_min_knot_dt_s,"
           "injected_support_max_knot_dt_s,"
           "injected_support_quality_min,"
           "injected_support_condition_number,injected_support_score,"
           "injected_support_state,injected_support_enter_counter,"
           "injected_support_exit_counter,"
           "injected_support_weakest_knot_index,"
           "injected_support_weakest_knot_energy_ratio,"
           "injected_support_weakest_rotation_ratio,"
           "injected_support_boundary_energy_ratio,"
           "injected_degeneracy_cause\n";
    injection_csv_stream_.flush();
  }

  void ObservabilityAnalyzer::WriteInjectionCsvRow(
      int64_t scan_timestamp_ns,
      const ObservabilityResult &original_result,
      const SupportInjectionDiagnostic &injection_result)
  {
    if (!injection_csv_stream_.is_open())
    {
      return;
    }

    const SupportInjectionMetadata &metadata = injection_result.metadata;
    const ObservabilityResult &injected = injection_result.injected_support;
    injection_csv_stream_
        << std::setprecision(12)
        << scan_timestamp_ns * Trajectory::NS_TO_S << ','
        << static_cast<int>(metadata.mode) << ','
        << SupportDegradationInjector::ModeName(metadata.mode) << ','
        << metadata.severity << ',' << metadata.phase_start << ','
        << metadata.phase_end << ',' << metadata.random_seed << ','
        << static_cast<int>(metadata.applied) << ','
        << metadata.input_sample_num << ',' << metadata.selected_sample_num
        << ',' << metadata.modified_sample_num << ','
        << metadata.removed_sample_num << ',' << metadata.output_sample_num
        << ',' << metadata.retained_ratio << ','
        << metadata.timestamp_span_ratio << ','
        << static_cast<int>(original_result.valid) << ','
        << original_result.relative_eigenvalues[0] << ','
        << original_result.degeneracy_score << ','
        << static_cast<int>(original_result.degenerate_state) << ','
        << static_cast<int>(original_result.support_valid) << ','
        << original_result.support_quality_min << ','
        << original_result.support_weak_direction_num << ','
        << static_cast<int>(original_result.support_degenerate_state) << ','
        << original_result.degeneracy_cause << ','
        << static_cast<int>(injected.support_valid) << ','
        << injected.support_control_point_num << ','
        << injected.support_interval_num << ','
        << injected.support_dimension << ','
        << injected.support_effective_rank << ','
        << injected.support_weak_direction_num << ','
        << injected.support_time_span_s << ','
        << injected.support_min_knot_dt_s << ','
        << injected.support_max_knot_dt_s << ','
        << injected.support_quality_min << ','
        << injected.support_condition_number << ','
        << injected.support_score << ','
        << static_cast<int>(injected.support_degenerate_state) << ','
        << injected.support_enter_counter << ','
        << injected.support_exit_counter << ','
        << injected.support_weakest_knot_index << ','
        << injected.support_weakest_knot_energy_ratio << ','
        << injected.support_weakest_rotation_ratio << ','
        << injected.support_boundary_energy_ratio << ','
        << injection_result.degeneracy_cause << '\n';
    injection_csv_stream_.flush();
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
              << " | support_valid=" << result.support_valid
              << " | support_q=" << result.support_quality_min
              << " | support_weak=" << result.support_weak_direction_num
              << " | support_state="
              << static_cast<int>(result.support_degenerate_state)
              << " | cause="
              << DegeneracyCauseName(result.degeneracy_cause)
              << " | rel_eigs="
              << result.relative_eigenvalues.transpose();
    if (support_injection_enabled_)
    {
      const ObservabilityResult &injected =
          last_injection_result_.injected_support;
      std::cout << " | injection="
                << SupportDegradationInjector::ModeName(
                       last_injection_result_.metadata.mode)
                << " | injected_q=" << injected.support_quality_min
                << " | injected_support_state="
                << static_cast<int>(injected.support_degenerate_state)
                << " | injected_cause="
                << DegeneracyCauseName(
                       last_injection_result_.degeneracy_cause);
    }
    std::cout << std::endl;
  }

} // namespace cocolic
