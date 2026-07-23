/*
 * Coco-LIC detector-only LiDAR observability analysis.
 */

#include <degeneracy/observability_analyzer.h>

#include <degeneracy/dso_fixed_config.h>

#include <odom/factor/analytic_diff/so3_spline_view.h>
#include <utils/sophus_utils.hpp>

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <initializer_list>
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

    void WarnIgnoredFixedKeys(
        const YAML::Node &node, const char *scope,
        std::initializer_list<const char *> keys)
    {
      if (!node)
      {
        return;
      }
      bool wrote_prefix = false;
      for (const char *key : keys)
      {
        if (node[key])
        {
          if (!wrote_prefix)
          {
            std::cerr << "[DSO-Config] Ignoring fixed YAML keys under '"
                      << scope << "': ";
            wrote_prefix = true;
          }
          else
          {
            std::cerr << ", ";
          }
          std::cerr << key;
        }
      }
      if (wrote_prefix)
      {
        std::cerr << ". Remove them from the configuration.\n";
      }
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
    output_csv_ = dso_fixed::kOutputCsv;
    use_correspondence_scale_ = dso_fixed::kUseCorrespondenceScale;
    min_correspondences_ = dso_fixed::kMinCorrespondences;
    analyze_every_n_scans_ = dso_fixed::kAnalyzeEveryNScans;
    print_every_n_scans_ = dso_fixed::kPrintEveryNScans;
    relative_eigenvalue_threshold_ =
        dso_fixed::kLegacyRelativeEigenvalueThreshold;
    enter_relative_eigenvalue_threshold_ =
        std::max(0.0, ReadValue<double>(
                          node, "enter_relative_eigenvalue_threshold", 3e-3));
    exit_relative_eigenvalue_threshold_ =
        std::max(enter_relative_eigenvalue_threshold_,
                 ReadValue<double>(
                     node, "exit_relative_eigenvalue_threshold", 6e-3));
    enter_consecutive_scans_ =
        dso_fixed::kDetectorEnterConsecutiveScans;
    exit_consecutive_scans_ = dso_fixed::kDetectorExitConsecutiveScans;
    min_characteristic_range_ = dso_fixed::kMinCharacteristicRange;
    max_characteristic_range_ = dso_fixed::kMaxCharacteristicRange;

    support_enabled_ = dso_fixed::kSupportEnabled;
    support_reference_samples_per_interval_ =
        dso_fixed::kSupportReferenceSamplesPerInterval;
    support_max_control_points_ = dso_fixed::kSupportMaxControlPoints;
    support_enter_quality_threshold_ = std::max(
        0.0, ReadValue<double>(
                 node, "support_enter_quality_threshold", 2e-2));
    support_exit_quality_threshold_ = std::max(
        support_enter_quality_threshold_,
        ReadValue<double>(node, "support_exit_quality_threshold", 5e-2));
    support_enter_consecutive_scans_ =
        dso_fixed::kSupportEnterConsecutiveScans;
    support_exit_consecutive_scans_ =
        dso_fixed::kSupportExitConsecutiveScans;

    const YAML::Node injection_node =
        node ? node["support_injection"] : YAML::Node();
    const bool injection_requested =
        ReadValue<bool>(injection_node, "enabled", false);
    support_injection_output_csv_ = dso_fixed::kInjectionOutputCsv;
    const std::string injection_mode_name = ReadValue<std::string>(
        injection_node, "mode", "timestamp_compression");
    SupportInjectionMode injection_mode = SupportInjectionMode::Disabled;
    const bool injection_mode_valid =
        SupportDegradationInjector::ParseMode(injection_mode_name,
                                               injection_mode);
    SupportInjectionConfig injection_config;
    // This object has no estimator write path; injection is diagnostic-only
    // by construction rather than by a user-settable promise.
    injection_config.enabled = enabled_ && injection_requested &&
                               support_enabled_ && injection_mode_valid;
    injection_config.mode = injection_mode;
    injection_config.severity =
        ReadValue<double>(injection_node, "severity",
                          dso_fixed::kInjectionDefaultSeverity);
    injection_config.phase_start =
        ReadValue<double>(injection_node, "phase_start",
                          dso_fixed::kInjectionDefaultPhaseStart);
    injection_config.phase_end =
        ReadValue<double>(injection_node, "phase_end",
                          dso_fixed::kInjectionDefaultPhaseEnd);
    injection_config.random_seed =
        ReadValue<uint64_t>(injection_node, "random_seed",
                            dso_fixed::kInjectionDefaultRandomSeed);
    support_injector_.Configure(injection_config);
    support_injection_enabled_ = support_injector_.Enabled();

    const YAML::Node casr_node =
        node ? node["casr_shadow"] : YAML::Node();
    casr_shadow_output_csv_ = dso_fixed::kCasrShadowOutputCsv;
    CasrShadowConfig casr_config;
    casr_config.enabled = enabled_ && support_enabled_;
    // Reuse the detector exit band instead of exposing a duplicated
    // environment threshold.
    casr_config.environment_relative_threshold =
        exit_relative_eigenvalue_threshold_;
    casr_config.support_basis_relative_singular_threshold =
        dso_fixed::kSupportBasisRelativeSingularThreshold;
    casr_config.lift_regularization = dso_fixed::kLiftRegularization;
    casr_config.principal_cosine_threshold = ReadValue<double>(
        casr_node, "principal_cosine_threshold", 7e-1);
    casr_config.route_consecutive_scans =
        dso_fixed::kRouteConsecutiveScans;
    casr_config.projector_consecutive_scans =
        dso_fixed::kProjectorConsecutiveScans;
    casr_config.projector_similarity_threshold = ReadValue<double>(
        casr_node, "projector_similarity_threshold", 8e-1);
    casr_config.scheduler_enabled = casr_config.enabled;
    casr_config.scheduler_environment_full_confidence_threshold =
        enter_relative_eigenvalue_threshold_;
    casr_config.scheduler_environment_zero_confidence_threshold =
        exit_relative_eigenvalue_threshold_;
    casr_config.scheduler_support_full_confidence_threshold =
        support_enter_quality_threshold_;
    casr_config.scheduler_support_zero_confidence_threshold =
        support_exit_quality_threshold_;
    casr_config.scheduler_projector_full_confidence =
        dso_fixed::kSchedulerProjectorFullConfidence;
    casr_config.scheduler_principal_full_confidence =
        dso_fixed::kSchedulerPrincipalFullConfidence;
    casr_config.scheduler_persistence_full_scans =
        dso_fixed::kSchedulerPersistenceFullScans;
    casr_config.scheduler_enter_confidence =
        dso_fixed::kSchedulerEnterConfidence;
    casr_config.scheduler_exit_confidence =
        dso_fixed::kSchedulerExitConfidence;
    casr_config.scheduler_rise_time_s =
        dso_fixed::kSchedulerRiseTimeSeconds;
    casr_config.scheduler_fall_time_s =
        dso_fixed::kSchedulerFallTimeSeconds;
    casr_config.scheduler_max_dt_s = dso_fixed::kSchedulerMaxDtSeconds;
    casr_shadow_evaluator_.Configure(casr_config);
    casr_shadow_enabled_ = casr_shadow_evaluator_.Enabled();

    const YAML::Node intervention_node =
        node ? node["casr_intervention"] : YAML::Node();
    const CasrInterventionConfig requested_intervention_config =
        ReadCasrInterventionConfig(intervention_node);
    casr_intervention_config_ = requested_intervention_config;
    casr_intervention_config_.enabled =
        requested_intervention_config.enabled && enabled_ &&
        casr_shadow_enabled_ && casr_config.scheduler_enabled;

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

    WarnIgnoredFixedKeys(
        node, "dso_detect_only",
        {"output_csv", "use_correspondence_scale", "min_correspondences",
         "analyze_every_n_scans", "print_every_n_scans",
         "relative_eigenvalue_threshold", "enter_consecutive_scans",
         "exit_consecutive_scans", "min_characteristic_range",
         "max_characteristic_range", "support_enabled",
         "support_reference_samples_per_interval",
         "support_max_control_points", "support_enter_consecutive_scans",
         "support_exit_consecutive_scans"});
    WarnIgnoredFixedKeys(injection_node, "support_injection",
                         {"diagnostics_only", "output_csv"});
    WarnIgnoredFixedKeys(
        casr_node, "casr_shadow",
        {"enabled", "shadow_only", "output_csv",
         "environment_relative_threshold",
         "support_basis_relative_singular_threshold", "lift_regularization",
         "route_consecutive_scans", "projector_consecutive_scans",
         "support_pose_relative_threshold"});
    const YAML::Node scheduler_node =
        casr_node ? casr_node["activation_scheduler"] : YAML::Node();
    WarnIgnoredFixedKeys(
        scheduler_node, "casr_shadow.activation_scheduler",
        {"enabled", "environment_full_confidence_threshold",
         "environment_zero_confidence_threshold",
         "support_full_confidence_threshold",
         "support_zero_confidence_threshold", "projector_full_confidence",
         "principal_full_confidence", "persistence_full_scans",
         "enter_confidence", "exit_confidence", "rise_time_s",
         "fall_time_s", "max_dt_s"});
    WarnIgnoredFixedKeys(
        intervention_node, "casr_intervention",
        {"output_csv", "curvature_matching_enabled",
         "curvature_target_relative_to_max",
         "curvature_max_added_relative_to_max", "curvature_min_reference",
         "counterfactual_validation",
         "counterfactual_ratio_denominator_floor",
         "base_information_weight", "max_effective_information_weight",
         "min_activation_strength", "max_activation_strength",
         "max_control_points", "max_recovery_rank",
         "max_basis_orthogonality_error"});

    if (injection_requested && !injection_mode_valid)
    {
      std::cerr << "[DSO-DetectOnly] support_injection disabled: unknown "
                << "mode '" << injection_mode_name << "'.\n";
    }
    if (requested_intervention_config.enabled &&
        !casr_intervention_config_.enabled)
    {
      std::cerr << "[CASR-Intervention] disabled: DSO diagnostics, "
                   "CASR shadow, support analysis, and the activation "
                   "scheduler must all be enabled.\n";
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

    casr_csv_path_ = output_prefix + "_casr_shadow.csv";
    if (casr_shadow_enabled_ && casr_shadow_output_csv_)
    {
      casr_csv_stream_.open(casr_csv_path_,
                            std::ios::out | std::ios::trunc);
      if (!casr_csv_stream_.is_open())
      {
        std::cerr << "[CASR-Shadow] Cannot open CSV: " << casr_csv_path_
                  << ". Console diagnostics remain enabled.\n";
      }
      else
      {
        WriteCasrCsvHeader();
      }
    }

    intervention_csv_path_ = output_prefix + "_casr_intervention.csv";
    if (casr_intervention_config_.enabled &&
        casr_intervention_config_.output_csv)
    {
      intervention_csv_stream_.open(intervention_csv_path_,
                                    std::ios::out | std::ios::trunc);
      if (!intervention_csv_stream_.is_open())
      {
        std::cerr << "[CASR-Intervention] Cannot open CSV: "
                  << intervention_csv_path_ << ".\n";
      }
      else
      {
        WriteInterventionCsvHeader();
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
    std::cout << " | casr_shadow=" << casr_shadow_enabled_;
    if (casr_shadow_enabled_)
    {
      const CasrShadowConfig &config = casr_shadow_evaluator_.Config();
      std::cout << "(" << kCasrShadowMethodVersion << ",principal_cos="
                << config.principal_cosine_threshold
                << ",route_persistence="
                << config.route_consecutive_scans
                << ",projector_persistence="
                << config.projector_consecutive_scans
                << ",projector_similarity="
                << config.projector_similarity_threshold
                << ",scheduler=" << config.scheduler_enabled;
      if (config.scheduler_enabled)
      {
        std::cout << ",scheduler_enter/exit="
                  << config.scheduler_enter_confidence << "/"
                  << config.scheduler_exit_confidence
                  << ",scheduler_rise/fall_s="
                  << config.scheduler_rise_time_s << "/"
                  << config.scheduler_fall_time_s;
      }
      std::cout << ")";
    }
    std::cout << " | casr_intervention="
              << casr_intervention_config_.enabled;
    if (casr_intervention_config_.enabled)
    {
      std::cout << "(" << kCasrInterventionMethodVersion
                << ",apply="
                << casr_intervention_config_.apply_to_estimator
                << ",curvature_matching="
                << casr_intervention_config_.curvature_matching_enabled
                << ",curvature_target/max="
                << casr_intervention_config_
                       .curvature_target_relative_to_max
                << "/"
                << casr_intervention_config_
                       .curvature_max_added_relative_to_max
                << ",curvature_gain="
                << casr_intervention_config_.curvature_gain
                << ",counterfactual="
                << casr_intervention_config_.counterfactual_validation
                << ",activation="
                << casr_intervention_config_.min_activation_strength
                << "-"
                << casr_intervention_config_.max_activation_strength
                << ")";
    }
    std::cout << " | csv="
              << (csv_stream_.is_open() ? csv_path_ : std::string("disabled"))
              << " | casr_csv="
              << (casr_csv_stream_.is_open()
                      ? casr_csv_path_
                      : std::string("disabled"))
              << " | intervention_csv="
              << (intervention_csv_stream_.is_open()
                      ? intervention_csv_path_
                      : std::string("disabled"))
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
    if (casr_csv_stream_.is_open())
    {
      casr_csv_stream_.flush();
      casr_csv_stream_.close();
    }
    if (intervention_csv_stream_.is_open())
    {
      intervention_csv_stream_.flush();
      intervention_csv_stream_.close();
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

    if (casr_shadow_enabled_)
    {
      last_casr_result_ = AnalyzeCasr(
          last_result_, last_result_, &real_casr_temporal_state_,
          CasrDataSource::RealMeasurements);
      last_injected_casr_result_ = CasrShadowResult();
      const CasrShadowResult *injected_casr = nullptr;
      if (support_injection_enabled_)
      {
        last_injected_casr_result_ = AnalyzeCasr(
            last_result_, last_injection_result_.injected_support,
            &injected_casr_temporal_state_,
            CasrDataSource::DiagnosticsCopy);
        injected_casr = &last_injected_casr_result_;
      }
      WriteCasrCsvRow(scan_timestamp_ns, last_result_, last_casr_result_,
                      injected_casr);
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
    Eigen::MatrixXd reference_pose_information =
        Eigen::MatrixXd::Zero(dimension, dimension);
    Eigen::MatrixXd reference_pose_cross =
        Eigen::MatrixXd::Zero(dimension, 6);

    const int support_interval_num =
        max_interval_index - min_interval_index + 1;
    const int temporal_bin_num =
        support_interval_num * support_reference_samples_per_interval_;
    std::vector<int> interval_sample_counts(
        static_cast<size_t>(support_interval_num), 0);
    std::vector<double> observed_temporal_mass(
        static_cast<size_t>(temporal_bin_num), 0.0);
    std::vector<double> reference_temporal_mass(
        static_cast<size_t>(temporal_bin_num), 0.0);

    using SO3View = analytic_derivative::So3SplineView;
    using LocalMapping =
        Eigen::Matrix<double, 6, 6 * SplineOrder>;
    const Eigen::Vector3d lidar_position_in_imu =
        trajectory_->GetSensorEP(LiDARSensor).p;
    const double characteristic_range =
        std::max(result.characteristic_range, 1e-3);

    const auto build_local_mappings =
        [&](int interval_index, int control_start_index, double u,
            LocalMapping &support_mapping,
            LocalMapping &pose_mapping) -> bool
    {
      if (!std::isfinite(u) || u < 0.0 || u > 1.0)
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
      const SO3d imu_rotation = SO3View::EvaluateRpNURBS(
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

      support_mapping.setZero();
      pose_mapping.setZero();
      const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
      const Eigen::Matrix3d rotation_matrix = imu_rotation.matrix();
      const Eigen::Matrix3d lever_arm_mapping =
          -rotation_matrix * SO3d::hat(lidar_position_in_imu) /
          characteristic_range;
      for (int i = 0; i < SplineOrder; ++i)
      {
        const Eigen::Matrix3d &rotation_knot_jacobian =
            rotation_jacobian.d_val_d_knot[static_cast<size_t>(i)];
        if (!rotation_knot_jacobian.allFinite())
        {
          return false;
        }

        const int offset = 6 * i;
        support_mapping.block<3, 3>(0, offset) =
            rotation_knot_jacobian;
        support_mapping.block<3, 3>(3, offset + 3) =
            position_coefficients[i] * identity;

        // The environment detector uses a map-frame left perturbation of the
        // LiDAR pose. NURBS rotation knots use right perturbations of the IMU
        // attitude, hence R_I*J_i. The lower-left block is the exact LiDAR
        // lever-arm translation induced by the scaled rotation knot.
        pose_mapping.block<3, 3>(0, offset) =
            rotation_matrix * rotation_knot_jacobian;
        pose_mapping.block<3, 3>(3, offset) =
            lever_arm_mapping * rotation_knot_jacobian;
        pose_mapping.block<3, 3>(3, offset + 3) =
            position_coefficients[i] * identity;
      }
      return support_mapping.allFinite() && pose_mapping.allFinite();
    };

    const auto accumulate_mapping =
        [&](int interval_index, int control_start_index, double u,
            double sample_weight, Eigen::MatrixXd &information,
            Eigen::MatrixXd *pose_information,
            Eigen::MatrixXd *pose_cross) -> bool
    {
      if (!std::isfinite(u) || u < 0.0 || u > 1.0 ||
          !std::isfinite(sample_weight) || sample_weight <= 0.0)
      {
        return false;
      }

      LocalMapping support_mapping;
      LocalMapping local_pose_mapping;
      if (!build_local_mappings(interval_index, control_start_index, u,
                                support_mapping, local_pose_mapping))
      {
        return false;
      }

      for (int i = 0; i < SplineOrder; ++i)
      {
        const int local_i = control_start_index + i - min_control_index;
        const int global_i = 6 * local_i;
        const int mapping_i = 6 * i;
        if (pose_cross)
        {
          pose_cross->block(global_i, 0, 6, 6).noalias() +=
              sample_weight *
              local_pose_mapping.block(0, mapping_i, 6, 6).transpose();
        }
        for (int j = 0; j < SplineOrder; ++j)
        {
          const int local_j = control_start_index + j - min_control_index;
          const int global_j = 6 * local_j;
          const int mapping_j = 6 * j;
          information.block(global_i, global_j, 6, 6).noalias() +=
              sample_weight *
              support_mapping.block(0, mapping_i, 6, 6).transpose() *
              support_mapping.block(0, mapping_j, 6, 6);
          if (pose_information)
          {
            pose_information->block(global_i, global_j, 6, 6).noalias() +=
                sample_weight *
                local_pose_mapping.block(0, mapping_i, 6, 6).transpose() *
                local_pose_mapping.block(0, mapping_j, 6, 6);
          }
        }
      }
      return true;
    };

    double observed_weight_sum = 0.0;
    for (const auto &sample : samples)
    {
      if (accumulate_mapping(sample.interval_index,
                             sample.control_start_index, sample.u,
                             sample.squared_weight, observed_information,
                             nullptr, nullptr))
      {
        observed_weight_sum += sample.squared_weight;
        const int local_interval =
            sample.interval_index - min_interval_index;
        const int local_bin = std::min(
            support_reference_samples_per_interval_ - 1,
            std::max(0, static_cast<int>(
                sample.u * support_reference_samples_per_interval_)));
        const int temporal_bin =
            local_interval * support_reference_samples_per_interval_ +
            local_bin;
        ++interval_sample_counts[static_cast<size_t>(local_interval)];
        observed_temporal_mass[static_cast<size_t>(temporal_bin)] +=
            sample.squared_weight;
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
                               reference_information,
                               &reference_pose_information,
                               &reference_pose_cross))
        {
          reference_weight_sum += reference_sample_weight;
          const int temporal_bin =
              (interval_index - min_interval_index) *
                  support_reference_samples_per_interval_ +
              sample_index;
          reference_temporal_mass[static_cast<size_t>(temporal_bin)] +=
              reference_sample_weight;
        }
      }
    }

    if (!std::isfinite(observed_weight_sum) || observed_weight_sum <= 0.0 ||
        !std::isfinite(reference_weight_sum) || reference_weight_sum <= 0.0)
    {
      return;
    }

    int empty_interval_num = 0;
    int min_interval_sample_num = std::numeric_limits<int>::max();
    int max_interval_sample_num = 0;
    for (const int count : interval_sample_counts)
    {
      empty_interval_num += count == 0 ? 1 : 0;
      min_interval_sample_num = std::min(min_interval_sample_num, count);
      max_interval_sample_num = std::max(max_interval_sample_num, count);
    }
    int occupied_temporal_bin_num = 0;
    double temporal_mass_total_variation = 0.0;
    for (int bin_index = 0; bin_index < temporal_bin_num; ++bin_index)
    {
      const double observed_mass =
          observed_temporal_mass[static_cast<size_t>(bin_index)] /
          observed_weight_sum;
      const double reference_mass =
          reference_temporal_mass[static_cast<size_t>(bin_index)] /
          reference_weight_sum;
      occupied_temporal_bin_num += observed_mass > 0.0 ? 1 : 0;
      temporal_mass_total_variation += std::abs(observed_mass - reference_mass);
    }
    temporal_mass_total_variation *= 0.5;

    observed_information /= observed_weight_sum;
    reference_information /= reference_weight_sum;
    reference_pose_information /= reference_weight_sum;
    reference_pose_cross /= reference_weight_sum;
    observed_information =
        0.5 * (observed_information + observed_information.transpose());
    reference_information =
        0.5 * (reference_information + reference_information.transpose());
    reference_pose_information =
        0.5 * (reference_pose_information +
               reference_pose_information.transpose());

    // A representative-time mapping is logged only as a compact 6DoF view of
    // each knot-space candidate. It is not used for principal angles or route
    // selection. The actual environment lift above uses all reference times.
    Eigen::MatrixXd representative_pose_mapping =
        Eigen::MatrixXd::Zero(6, dimension);
    const int64_t representative_timestamp_ns =
        min_timestamp_ns + (max_timestamp_ns - min_timestamp_ns) / 2;
    const auto representative_upper = std::upper_bound(
        knot_times.begin(), knot_times.end(), representative_timestamp_ns);
    if (representative_upper == knot_times.begin() ||
        representative_upper == knot_times.end())
    {
      return;
    }
    const int representative_interval_index =
        static_cast<int>(
            std::distance(knot_times.begin(), representative_upper)) -
        1;
    const int representative_control_start_index =
        representative_interval_index - 3;
    const int64_t representative_interval_ns =
        knot_times[static_cast<size_t>(representative_interval_index + 1)] -
        knot_times[static_cast<size_t>(representative_interval_index)];
    if (representative_control_start_index < min_control_index ||
        representative_control_start_index + SplineOrder - 1 >
            max_control_index ||
        representative_interval_ns <= 0)
    {
      return;
    }
    const double representative_u =
        static_cast<double>(
            representative_timestamp_ns -
            knot_times[static_cast<size_t>(representative_interval_index)]) /
        static_cast<double>(representative_interval_ns);
    LocalMapping representative_support_mapping;
    LocalMapping representative_local_pose_mapping;
    if (!build_local_mappings(representative_interval_index,
                              representative_control_start_index,
                              representative_u,
                              representative_support_mapping,
                              representative_local_pose_mapping))
    {
      return;
    }
    for (int i = 0; i < SplineOrder; ++i)
    {
      const int local_index =
          representative_control_start_index + i - min_control_index;
      representative_pose_mapping.block(0, 6 * local_index, 6, 6) =
          representative_local_pose_mapping.block(0, 6 * i, 6, 6);
    }

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

    // CASR-v2 keeps the weak generalized modes in the full active knot space.
    // A physical rotation perturbation dtheta corresponds to r*dtheta in the
    // environment detector's scaled state, so every rotation knot block is
    // scaled by the same characteristic range before Euclidean
    // orthonormalization. No per-knot outer-product folding is performed.
    const int candidate_knot_mode_num = std::max(1, weak_direction_num);
    Eigen::MatrixXd candidate_knot_modes =
        Eigen::MatrixXd::Zero(dimension, candidate_knot_mode_num);
    int accepted_knot_mode_num = 0;
    for (int mode_index = 0; mode_index < candidate_knot_mode_num;
         ++mode_index)
    {
      Eigen::VectorXd scaled_mode =
          whitening * quality_solver.eigenvectors().col(mode_index);
      for (int local_index = 0; local_index < control_point_num;
           ++local_index)
      {
        scaled_mode.segment<3>(6 * local_index) *= characteristic_range;
      }
      const double scaled_norm = scaled_mode.norm();
      if (!std::isfinite(scaled_norm) || scaled_norm <= 1e-12)
      {
        continue;
      }
      candidate_knot_modes.col(accepted_knot_mode_num) =
          scaled_mode / scaled_norm;
      ++accepted_knot_mode_num;
    }
    if (accepted_knot_mode_num <= 0)
    {
      return;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> knot_mode_svd(
        candidate_knot_modes.leftCols(accepted_knot_mode_num),
        Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd knot_mode_singular_values =
        knot_mode_svd.singularValues();
    if (knot_mode_singular_values.size() == 0 ||
        !knot_mode_singular_values.allFinite() ||
        knot_mode_singular_values[0] <= 1e-12)
    {
      return;
    }
    const double knot_mode_threshold =
        std::max(1e-12, 1e-8 * knot_mode_singular_values[0]);
    int support_knot_mode_num = 0;
    for (int i = 0; i < knot_mode_singular_values.size(); ++i)
    {
      if (knot_mode_singular_values[i] >= knot_mode_threshold)
      {
        ++support_knot_mode_num;
      }
    }
    if (support_knot_mode_num <= 0)
    {
      return;
    }
    const Eigen::MatrixXd support_knot_weak_basis =
        knot_mode_svd.matrixU().leftCols(support_knot_mode_num);

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
    size_t accepted_sample_num = 0;
    for (const int count : interval_sample_counts)
    {
      accepted_sample_num += static_cast<size_t>(count);
    }
    result.support_sample_num = accepted_sample_num;
    result.support_empty_interval_num = empty_interval_num;
    result.support_min_interval_sample_num = min_interval_sample_num;
    result.support_max_interval_sample_num = max_interval_sample_num;
    result.support_temporal_bin_num = temporal_bin_num;
    result.support_occupied_temporal_bin_num =
        occupied_temporal_bin_num;
    result.support_occupied_temporal_bin_ratio =
        static_cast<double>(occupied_temporal_bin_num) /
        static_cast<double>(temporal_bin_num);
    result.support_temporal_mass_total_variation = temporal_mass_total_variation;
    result.support_control_point_start_index = min_control_index;
    result.support_knot_mode_num = support_knot_mode_num;
    result.support_knot_weak_basis = support_knot_weak_basis;
    result.support_reference_pose_information =
        reference_pose_information;
    result.support_reference_pose_cross = reference_pose_cross;
    result.support_representative_pose_mapping =
        representative_pose_mapping;
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
    injected.characteristic_range = original_result.characteristic_range;
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

  CasrShadowResult ObservabilityAnalyzer::AnalyzeCasr(
      const ObservabilityResult &environment_result,
      const ObservabilityResult &support_result,
      CasrTemporalState *temporal_state,
      CasrDataSource data_source) const
  {
    CasrShadowInput input;
    input.data_source = data_source;
    input.environment_valid = environment_result.valid;
    input.support_valid = support_result.support_valid;
    input.environment_state = environment_result.degenerate_state;
    input.support_state = support_result.support_degenerate_state;
    input.scan_timestamp_s =
        environment_result.scan_timestamp_ns * Trajectory::NS_TO_S;
    input.support_quality_min = support_result.support_quality_min;
    input.environment_relative_eigenvalues =
        environment_result.relative_eigenvalues;
    input.environment_eigenvectors = environment_result.eigenvectors;
    input.support_control_point_start_index =
        support_result.support_control_point_start_index;
    input.support_knot_weak_basis =
        &support_result.support_knot_weak_basis;
    input.reference_pose_information =
        &support_result.support_reference_pose_information;
    input.reference_pose_cross =
        &support_result.support_reference_pose_cross;
    input.representative_pose_mapping =
        &support_result.support_representative_pose_mapping;
    return casr_shadow_evaluator_.Evaluate(input, temporal_state);
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
                   "support_boundary_energy_ratio,degeneracy_cause,"
                   "support_sample_num,support_empty_interval_num,"
                   "support_min_interval_sample_num,"
                   "support_max_interval_sample_num,"
                   "support_temporal_bin_num,"
                   "support_occupied_temporal_bin_num,"
                   "support_occupied_temporal_bin_ratio,"
                   "support_temporal_mass_total_variation";
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
                << result.degeneracy_cause << ','
                << result.support_sample_num << ','
                << result.support_empty_interval_num << ','
                << result.support_min_interval_sample_num << ','
                << result.support_max_interval_sample_num << ','
                << result.support_temporal_bin_num << ','
                << result.support_occupied_temporal_bin_num << ','
                << result.support_occupied_temporal_bin_ratio << ','
                << result.support_temporal_mass_total_variation;
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
           "injection_applied,input_sample_num,boundary_anchor_sample_num,"
           "selected_sample_num,"
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
        << metadata.input_sample_num << ','
        << metadata.boundary_anchor_sample_num << ','
        << metadata.selected_sample_num
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

  void ObservabilityAnalyzer::WriteCasrCsvHeader()
  {
    if (!casr_csv_stream_.is_open())
    {
      return;
    }

    casr_csv_stream_
        << "scan_timestamp_s,environment_valid,environment_state,"
           "environment_relative_lambda_0";
    const auto write_block_header = [&](const std::string &prefix)
    {
      casr_csv_stream_
          << ',' << prefix << "available"
          << ',' << prefix << "valid"
          << ',' << prefix << "support_valid"
          << ',' << prefix << "support_state"
          << ',' << prefix << "support_quality_min"
          << ',' << prefix << "support_knot_mode_num"
          << ',' << prefix << "cause"
          << ',' << prefix << "route_code"
          << ',' << prefix << "route"
          << ',' << prefix << "environment_rank"
          << ',' << prefix << "support_rank"
          << ',' << prefix << "common_rank"
          << ',' << prefix << "recovery_rank"
          << ',' << prefix << "overlap_score"
          << ',' << prefix << "principal_cosine_min"
          << ',' << prefix << "principal_cosine_max"
          << ',' << prefix << "environment_exclusive_ratio"
          << ',' << prefix << "support_exclusive_ratio";
      for (int i = 0; i < 6; ++i)
      {
        casr_csv_stream_ << ',' << prefix
                         << "support_pose_eigenvalue_" << i;
      }
      for (int row = 0; row < 6; ++row)
      {
        for (int col = row; col < 6; ++col)
        {
          casr_csv_stream_ << ',' << prefix << "recovery_p" << row << col;
        }
      }
      casr_csv_stream_
          << ',' << prefix << "method_version"
          << ',' << prefix << "support_control_point_start_index"
          << ',' << prefix << "support_knot_dimension"
          << ',' << prefix << "representative_pose_rank"
          << ',' << prefix << "environment_lift_residual"
          << ',' << prefix << "recovery_basis_orthogonality_error"
          << ',' << prefix << "stable_route_code"
          << ',' << prefix << "stable_route"
          << ',' << prefix << "route_candidate_count"
          << ',' << prefix << "temporal_projector_similarity"
          << ',' << prefix << "temporal_overlap_control_point_num"
          << ',' << prefix << "temporal_previous_overlap_rank"
          << ',' << prefix << "temporal_current_overlap_rank"
          << ',' << prefix << "temporal_forced_overlap_rank"
          << ',' << prefix << "temporal_projector_affinity"
          << ',' << prefix << "projector_consistency_count"
          << ',' << prefix << "recovery_ready"
          << ',' << prefix << "scheduler_state_code"
          << ',' << prefix << "scheduler_state"
          << ',' << prefix << "scheduler_eligible"
          << ',' << prefix << "scheduler_active"
          << ',' << prefix << "scheduler_environment_confidence"
          << ',' << prefix << "scheduler_support_confidence"
          << ',' << prefix << "scheduler_cause_confidence"
          << ',' << prefix << "scheduler_temporal_confidence"
          << ',' << prefix << "scheduler_persistence_confidence"
          << ',' << prefix << "scheduler_principal_confidence"
          << ',' << prefix << "scheduler_raw_confidence"
          << ',' << prefix << "scheduler_target_strength"
          << ',' << prefix << "scheduler_activation_strength"
          << ',' << prefix << "scheduler_dt_s";
    };
    write_block_header("real_");
    write_block_header("injected_");
    casr_csv_stream_ << '\n';
    casr_csv_stream_.flush();
  }

  void ObservabilityAnalyzer::WriteInterventionCsvHeader()
  {
    if (!intervention_csv_stream_.is_open())
    {
      return;
    }
    intervention_csv_stream_
        << "scan_timestamp_s,method_version,state_code,state,enabled,"
           "apply_to_estimator,eligible,factor_added,applied,"
           "data_source_code,route_code,route,"
           "recovery_mechanism_code,recovery_mechanism,"
           "recovery_reference_code,recovery_reference,"
           "control_point_start_index,control_point_num,recovery_rank,"
           "requested_activation_strength,used_activation_strength,"
           "characteristic_range,base_information_weight,"
           "max_effective_information_weight,"
           "effective_information_weight,sqrt_information_weight,"
           "pre_total_increment_norm,pre_projected_increment_norm,"
           "pre_orthogonal_increment_norm,pre_factor_residual_norm,"
           "post_total_increment_norm,post_projected_increment_norm,"
           "post_orthogonal_increment_norm,post_factor_residual_norm,"
           "max_rotation_increment_rad,max_translation_increment_m,"
           "solver_usable,solver_successful_steps,"
           "solver_unsuccessful_steps,primary_solver_usable,"
           "primary_solver_successful_steps,"
           "primary_solver_unsuccessful_steps,fallback_attempted,"
           "fallback_solver_usable,curvature_matching_enabled,"
           "curvature_valid,curvature_tangent_dimension,"
           "reference_curvature_max,target_curvature,"
           "recovery_curvature_min,recovery_curvature_median,"
           "recovery_curvature_max,added_information_min,"
           "added_information_median,added_information_max,"
           "continuity_operator_valid,continuity_operator_rank,"
           "continuity_symmetry_error,continuity_idempotence_error,"
           "pre_environment_residual_norm,pre_support_residual_norm,"
           "post_environment_residual_norm,post_support_residual_norm,"
           "counterfactual_enabled,counterfactual_solver_usable,"
           "counterfactual_solver_successful_steps,"
           "counterfactual_solver_unsuccessful_steps,"
           "counterfactual_total_increment_norm,"
           "counterfactual_projected_increment_norm,"
           "counterfactual_orthogonal_increment_norm,"
           "counterfactual_factor_residual_norm,"
           "counterfactual_environment_residual_norm,"
           "counterfactual_support_residual_norm,"
           "source_consensus_evaluated,source_consensus_sufficient,"
           "source_consensus_consistent,source_consensus_cosine,"
           "projected_casr_over_counterfactual,"
           "orthogonal_casr_over_counterfactual,"
           "support_curvature_audit_valid,support_curvature_rank,"
           "support_curvature_reference_min,"
           "support_curvature_reference_median,"
           "support_curvature_reference_mean,"
           "support_curvature_reference_max,"
           "support_curvature_projected_min,"
           "support_curvature_projected_median,"
           "support_curvature_projected_mean,"
           "support_curvature_projected_max,"
           "support_curvature_min_over_reference_max,"
           "support_curvature_mean_over_reference_mean,"
           "support_curvature_below_target_fraction\n";
    intervention_csv_stream_.flush();
  }

  void ObservabilityAnalyzer::LogCasrIntervention(
      const CasrInterventionReport &report)
  {
    if (!intervention_csv_stream_.is_open())
    {
      return;
    }
    intervention_csv_stream_
        << std::setprecision(12)
        << report.scan_timestamp_ns * Trajectory::NS_TO_S << ','
        << kCasrInterventionMethodVersion << ','
        << static_cast<int>(report.state) << ','
        << CasrInterventionStateName(report.state) << ','
        << static_cast<int>(report.enabled) << ','
        << static_cast<int>(report.apply_to_estimator) << ','
        << static_cast<int>(report.eligible) << ','
        << static_cast<int>(report.factor_added) << ','
        << static_cast<int>(report.applied) << ','
        << static_cast<int>(report.data_source) << ','
        << static_cast<int>(report.route) << ','
        << CasrRouteName(report.route) << ','
        << static_cast<int>(report.recovery_mechanism) << ','
        << CasrRecoveryMechanismName(report.recovery_mechanism) << ','
        << static_cast<int>(report.recovery_reference) << ','
        << CasrRecoveryReferenceName(report.recovery_reference) << ','
        << report.control_point_start_index << ','
        << report.control_point_num << ','
        << report.recovery_rank << ','
        << report.requested_activation_strength << ','
        << report.used_activation_strength << ','
        << report.characteristic_range << ','
        << report.base_information_weight << ','
        << report.max_effective_information_weight << ','
        << report.effective_information_weight << ','
        << report.sqrt_information_weight << ','
        << report.pre_total_increment_norm << ','
        << report.pre_projected_increment_norm << ','
        << report.pre_orthogonal_increment_norm << ','
        << report.pre_factor_residual_norm << ','
        << report.post_total_increment_norm << ','
        << report.post_projected_increment_norm << ','
        << report.post_orthogonal_increment_norm << ','
        << report.post_factor_residual_norm << ','
        << report.max_rotation_increment_rad << ','
        << report.max_translation_increment_m << ','
        << static_cast<int>(report.solver_usable) << ','
        << report.solver_successful_steps << ','
        << report.solver_unsuccessful_steps << ','
        << static_cast<int>(report.primary_solver_usable) << ','
        << report.primary_solver_successful_steps << ','
        << report.primary_solver_unsuccessful_steps << ','
        << static_cast<int>(report.fallback_attempted) << ','
        << static_cast<int>(report.fallback_solver_usable) << ','
        << static_cast<int>(report.curvature_matching_enabled) << ','
        << static_cast<int>(report.curvature_valid) << ','
        << report.curvature_tangent_dimension << ','
        << report.reference_curvature_max << ','
        << report.target_curvature << ','
        << report.recovery_curvature_min << ','
        << report.recovery_curvature_median << ','
        << report.recovery_curvature_max << ','
        << report.added_information_min << ','
        << report.added_information_median << ','
        << report.added_information_max << ','
        << static_cast<int>(report.continuity_operator_valid) << ','
        << report.continuity_operator_rank << ','
        << report.continuity_symmetry_error << ','
        << report.continuity_idempotence_error << ','
        << report.pre_environment_residual_norm << ','
        << report.pre_support_residual_norm << ','
        << report.post_environment_residual_norm << ','
        << report.post_support_residual_norm << ','
        << static_cast<int>(report.counterfactual_enabled) << ','
        << static_cast<int>(report.counterfactual_solver_usable) << ','
        << report.counterfactual_solver_successful_steps << ','
        << report.counterfactual_solver_unsuccessful_steps << ','
        << report.counterfactual_total_increment_norm << ','
        << report.counterfactual_projected_increment_norm << ','
        << report.counterfactual_orthogonal_increment_norm << ','
        << report.counterfactual_factor_residual_norm << ','
        << report.counterfactual_environment_residual_norm << ','
        << report.counterfactual_support_residual_norm << ','
        << static_cast<int>(report.source_consensus_evaluated) << ','
        << static_cast<int>(report.source_consensus_sufficient) << ','
        << static_cast<int>(report.source_consensus_consistent) << ','
        << report.source_consensus_cosine << ','
        << report.projected_casr_over_counterfactual << ','
        << report.orthogonal_casr_over_counterfactual << ','
        << static_cast<int>(report.support_curvature_audit_valid) << ','
        << report.support_curvature_rank << ','
        << report.support_curvature_reference_min << ','
        << report.support_curvature_reference_median << ','
        << report.support_curvature_reference_mean << ','
        << report.support_curvature_reference_max << ','
        << report.support_curvature_projected_min << ','
        << report.support_curvature_projected_median << ','
        << report.support_curvature_projected_mean << ','
        << report.support_curvature_projected_max << ','
        << report.support_curvature_min_over_reference_max << ','
        << report.support_curvature_mean_over_reference_mean << ','
        << report.support_curvature_below_target_fraction << '\n';
    intervention_csv_stream_.flush();
  }

  void ObservabilityAnalyzer::WriteCasrCsvRow(
      int64_t scan_timestamp_ns,
      const ObservabilityResult &original_result,
      const CasrShadowResult &real_result,
      const CasrShadowResult *injected_result)
  {
    if (!casr_csv_stream_.is_open())
    {
      return;
    }

    casr_csv_stream_
        << std::setprecision(12)
        << scan_timestamp_ns * Trajectory::NS_TO_S << ','
        << static_cast<int>(original_result.valid) << ','
        << static_cast<int>(original_result.degenerate_state) << ','
        << original_result.relative_eigenvalues[0];

    const auto write_block = [&](bool available,
                                 const ObservabilityResult &support_result,
                                 const CasrShadowResult &casr_result)
    {
      casr_csv_stream_
          << ',' << static_cast<int>(available)
          << ',' << static_cast<int>(casr_result.valid)
          << ',' << static_cast<int>(support_result.support_valid)
          << ',' << static_cast<int>(
                         support_result.support_degenerate_state)
          << ',' << support_result.support_quality_min
          << ',' << support_result.support_knot_mode_num
          << ',' << casr_result.cause
          << ',' << static_cast<int>(casr_result.route)
          << ',' << (available ? CasrRouteName(casr_result.route)
                              : "not_available")
          << ',' << casr_result.environment_rank
          << ',' << casr_result.support_rank
          << ',' << casr_result.common_rank
          << ',' << casr_result.recovery_rank
          << ',' << casr_result.overlap_score
          << ',' << casr_result.principal_cosine_min
          << ',' << casr_result.principal_cosine_max
          << ',' << casr_result.environment_exclusive_ratio
          << ',' << casr_result.support_exclusive_ratio;
      for (int i = 0; i < 6; ++i)
      {
        casr_csv_stream_ << ','
                         << casr_result.support_pose_eigenvalues[i];
      }
      for (int row = 0; row < 6; ++row)
      {
        for (int col = row; col < 6; ++col)
        {
          casr_csv_stream_ << ','
                           << casr_result.recovery_projector(row, col);
        }
      }
      casr_csv_stream_
          << ',' << kCasrShadowMethodVersion
          << ',' << casr_result.support_control_point_start_index
          << ',' << casr_result.support_knot_dimension
          << ',' << casr_result.representative_pose_rank
          << ',' << casr_result.environment_lift_residual
          << ',' << casr_result.recovery_basis_orthogonality_error
          << ',' << static_cast<int>(casr_result.stable_route)
          << ',' << (available
                         ? CasrRouteName(casr_result.stable_route)
                         : "not_available")
          << ',' << casr_result.route_candidate_count
          << ',' << casr_result.temporal_projector_similarity
          << ',' << casr_result.temporal_overlap_control_point_num
          << ',' << casr_result.temporal_previous_overlap_rank
          << ',' << casr_result.temporal_current_overlap_rank
          << ',' << casr_result.temporal_forced_overlap_rank
          << ',' << casr_result.temporal_projector_affinity
          << ',' << casr_result.projector_consistency_count
          << ',' << static_cast<int>(casr_result.recovery_ready)
          << ',' << static_cast<int>(casr_result.scheduler_state)
          << ',' << CasrSchedulerStateName(casr_result.scheduler_state)
          << ',' << static_cast<int>(casr_result.scheduler_eligible)
          << ',' << static_cast<int>(casr_result.scheduler_active)
          << ',' << casr_result.scheduler_environment_confidence
          << ',' << casr_result.scheduler_support_confidence
          << ',' << casr_result.scheduler_cause_confidence
          << ',' << casr_result.scheduler_temporal_confidence
          << ',' << casr_result.scheduler_persistence_confidence
          << ',' << casr_result.scheduler_principal_confidence
          << ',' << casr_result.scheduler_raw_confidence
          << ',' << casr_result.scheduler_target_strength
          << ',' << casr_result.scheduler_activation_strength
          << ',' << casr_result.scheduler_dt_s;
    };

    write_block(true, original_result, real_result);
    const ObservabilityResult empty_support;
    const CasrShadowResult empty_casr;
    if (injected_result)
    {
      write_block(true, last_injection_result_.injected_support,
                  *injected_result);
    }
    else
    {
      write_block(false, empty_support, empty_casr);
    }
    casr_csv_stream_ << '\n';
    casr_csv_stream_.flush();
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
                << " | anchors="
                << last_injection_result_.metadata.boundary_anchor_sample_num
                << " | injected_q=" << injected.support_quality_min
                << " | injected_support_state="
                << static_cast<int>(injected.support_degenerate_state)
                << " | injected_cause="
                << DegeneracyCauseName(
                       last_injection_result_.degeneracy_cause);
    }
    if (casr_shadow_enabled_)
    {
      std::cout << " | casr_route="
                << CasrRouteName(last_casr_result_.route)
                << " | casr_rank(e/s/c/r)="
                << last_casr_result_.environment_rank << "/"
                << last_casr_result_.support_rank << "/"
                << last_casr_result_.common_rank << "/"
                << last_casr_result_.recovery_rank
                << " | casr_overlap="
                << last_casr_result_.overlap_score
                << " | casr_stable="
                << CasrRouteName(last_casr_result_.stable_route)
                << " | casr_ready="
                << static_cast<int>(last_casr_result_.recovery_ready)
                << " | casr_temporal_similarity="
                << last_casr_result_.temporal_projector_similarity
                << " | casr_scheduler="
                << CasrSchedulerStateName(
                       last_casr_result_.scheduler_state)
                << " | casr_activation="
                << last_casr_result_.scheduler_activation_strength;
      if (support_injection_enabled_)
      {
        std::cout << " | injected_casr_route="
                  << CasrRouteName(last_injected_casr_result_.route)
                  << " | injected_casr_rank(e/s/c/r)="
                  << last_injected_casr_result_.environment_rank << "/"
                  << last_injected_casr_result_.support_rank << "/"
                  << last_injected_casr_result_.common_rank << "/"
                  << last_injected_casr_result_.recovery_rank
                  << " | injected_casr_overlap="
                  << last_injected_casr_result_.overlap_score
                  << " | injected_casr_stable="
                  << CasrRouteName(
                         last_injected_casr_result_.stable_route)
                  << " | injected_casr_ready="
                  << static_cast<int>(
                         last_injected_casr_result_.recovery_ready)
                  << " | injected_casr_temporal_similarity="
                  << last_injected_casr_result_
                         .temporal_projector_similarity
                  << " | injected_casr_scheduler="
                  << CasrSchedulerStateName(
                         last_injected_casr_result_.scheduler_state)
                  << " | injected_casr_activation="
                  << last_injected_casr_result_
                         .scheduler_activation_strength;
      }
    }
    std::cout << std::endl;
  }

} // namespace cocolic
