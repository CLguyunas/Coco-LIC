/*
 * Cause-adaptive subspace recovery (CASR) estimator intervention.
 */

#include <degeneracy/casr_intervention.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace cocolic
{
  namespace
  {
    template <typename T>
    T ReadValue(const YAML::Node &node, const char *key,
                const T &default_value)
    {
      return node && node[key] ? node[key].as<T>() : default_value;
    }

    double ReadFiniteDouble(const YAML::Node &node, const char *key,
                            double default_value)
    {
      const double value = ReadValue<double>(node, key, default_value);
      return std::isfinite(value) ? value : default_value;
    }

    bool IsSchedulableRoute(CasrRoute route)
    {
      return route == CasrRoute::EnvironmentCandidate ||
             route == CasrRoute::SupportCandidate ||
             route == CasrRoute::CoupledCommonCandidate;
    }

    double Median(const Eigen::VectorXd &values)
    {
      if (values.size() <= 0)
      {
        return 0.0;
      }
      std::vector<double> ordered(values.data(),
                                  values.data() + values.size());
      std::sort(ordered.begin(), ordered.end());
      const size_t middle = ordered.size() / 2;
      if (ordered.size() % 2 == 1)
      {
        return ordered[middle];
      }
      return 0.5 * (ordered[middle - 1] + ordered[middle]);
    }
  } // namespace

  const char *CasrInterventionStateName(CasrInterventionState state)
  {
    switch (state)
    {
    case CasrInterventionState::Disabled:
      return "disabled";
    case CasrInterventionState::InvalidInput:
      return "invalid_input";
    case CasrInterventionState::UnsafeRoute:
      return "unsafe_route";
    case CasrInterventionState::RouteMismatch:
      return "route_mismatch";
    case CasrInterventionState::NotReady:
      return "not_ready";
    case CasrInterventionState::SchedulerInactive:
      return "scheduler_inactive";
    case CasrInterventionState::ActivationTooLow:
      return "activation_too_low";
    case CasrInterventionState::InvalidBasis:
      return "invalid_basis";
    case CasrInterventionState::InvalidControlRange:
      return "invalid_control_range";
    case CasrInterventionState::MissingReference:
      return "missing_reference";
    case CasrInterventionState::DryRun:
      return "dry_run";
    case CasrInterventionState::Applied:
      return "applied";
    case CasrInterventionState::SolverFailure:
      return "solver_failure";
    case CasrInterventionState::DiagnosticsCopyBlocked:
      return "diagnostics_copy_blocked";
    case CasrInterventionState::SolverFailureRecovered:
      return "solver_failure_recovered";
    case CasrInterventionState::CurvatureInvalid:
      return "curvature_invalid";
    case CasrInterventionState::CurvatureSufficient:
      return "curvature_sufficient";
    default:
      return "unknown";
    }
  }

  CasrInterventionConfig ReadCasrInterventionConfig(
      const YAML::Node &node)
  {
    CasrInterventionConfig config;
    // Only experiment mode and the single recovery-strength hyperparameter
    // are public configuration. All numerical safeguards and the curvature
    // target live in dso_fixed_config.h and cannot be overridden by YAML.
    config.enabled = ReadValue<bool>(node, "enabled", false);
    config.apply_to_estimator =
        ReadValue<bool>(node, "apply_to_estimator", false);
    config.curvature_gain = std::max(
        0.0, ReadFiniteDouble(node, "curvature_gain",
                              dso_fixed::kDefaultCurvatureGain));
    return config;
  }

  CasrInterventionPlan BuildCasrInterventionPlan(
      const CasrInterventionConfig &config,
      const CasrShadowResult &casr_result,
      double characteristic_range,
      int trajectory_control_point_num)
  {
    CasrInterventionPlan plan;
    if (!config.enabled)
    {
      return plan;
    }

    plan.requested_activation_strength =
        casr_result.scheduler_activation_strength;
    plan.characteristic_range = characteristic_range;
    plan.control_point_start_index =
        casr_result.support_control_point_start_index;
    plan.control_point_num = casr_result.support_knot_dimension / 6;
    plan.recovery_rank = casr_result.recovery_rank;

    if (!casr_result.valid || !std::isfinite(characteristic_range) ||
        characteristic_range <= 0.0)
    {
      plan.state = CasrInterventionState::InvalidInput;
      return plan;
    }
    if (casr_result.data_source != CasrDataSource::RealMeasurements)
    {
      plan.state = CasrInterventionState::DiagnosticsCopyBlocked;
      return plan;
    }
    if (!IsSchedulableRoute(casr_result.route))
    {
      plan.state = CasrInterventionState::UnsafeRoute;
      return plan;
    }
    if (casr_result.route != casr_result.stable_route)
    {
      plan.state = CasrInterventionState::RouteMismatch;
      return plan;
    }
    if (!casr_result.recovery_ready || !casr_result.scheduler_eligible)
    {
      plan.state = CasrInterventionState::NotReady;
      return plan;
    }
    if (!casr_result.scheduler_active ||
        !std::isfinite(casr_result.scheduler_activation_strength) ||
        casr_result.scheduler_activation_strength <= 0.0)
    {
      plan.state = CasrInterventionState::SchedulerInactive;
      return plan;
    }
    if (casr_result.scheduler_activation_strength <
        config.min_activation_strength)
    {
      plan.state = CasrInterventionState::ActivationTooLow;
      return plan;
    }

    const Eigen::MatrixXd &basis = casr_result.recovery_knot_basis;
    if (casr_result.support_knot_dimension <= 0 ||
        casr_result.support_knot_dimension % 6 != 0 ||
        plan.control_point_num <= 0 ||
        plan.control_point_num > config.max_control_points ||
        plan.recovery_rank <= 0 ||
        plan.recovery_rank > config.max_recovery_rank ||
        basis.rows() != casr_result.support_knot_dimension ||
        basis.cols() != plan.recovery_rank || !basis.allFinite())
    {
      plan.state = CasrInterventionState::InvalidBasis;
      return plan;
    }
    const Eigen::MatrixXd basis_orthogonality_error =
        basis.transpose() * basis -
        Eigen::MatrixXd::Identity(plan.recovery_rank,
                                  plan.recovery_rank);
    if (!basis_orthogonality_error.allFinite() ||
        basis_orthogonality_error.cwiseAbs().maxCoeff() >
            config.max_basis_orthogonality_error)
    {
      plan.state = CasrInterventionState::InvalidBasis;
      return plan;
    }
    if (plan.control_point_start_index < 0 ||
        trajectory_control_point_num <= 0 ||
        plan.control_point_start_index + plan.control_point_num >
            trajectory_control_point_num)
    {
      plan.state = CasrInterventionState::InvalidControlRange;
      return plan;
    }

    plan.used_activation_strength = std::min(
        config.max_activation_strength,
        casr_result.scheduler_activation_strength);
    plan.curvature_matching_enabled = config.curvature_matching_enabled;
    plan.recovery_basis_rotation = Eigen::MatrixXd::Identity(
        plan.recovery_rank, plan.recovery_rank);
    if (!config.curvature_matching_enabled)
    {
      plan.effective_information_weight = std::min(
          config.max_effective_information_weight,
          config.base_information_weight * plan.used_activation_strength);
      plan.sqrt_information_weight =
          std::sqrt(std::max(0.0, plan.effective_information_weight));
      if (!std::isfinite(plan.sqrt_information_weight) ||
          plan.sqrt_information_weight <= 0.0)
      {
        plan.state = CasrInterventionState::InvalidInput;
        return plan;
      }
      plan.effective_information_weights = Eigen::VectorXd::Constant(
          plan.recovery_rank, plan.effective_information_weight);
      plan.sqrt_information_weights = Eigen::VectorXd::Constant(
          plan.recovery_rank, plan.sqrt_information_weight);
    }

    plan.eligible = true;
    plan.state = config.apply_to_estimator
                     ? CasrInterventionState::Applied
                     : CasrInterventionState::DryRun;
    return plan;
  }

  bool FinalizeCasrInterventionPlanWithCurvature(
      const CasrInterventionConfig &config,
      const CasrCurvatureEstimate &estimate,
      CasrInterventionPlan &plan)
  {
    if (!plan.eligible || !config.curvature_matching_enabled)
    {
      return plan.eligible;
    }

    plan.curvature_valid = estimate.valid;
    plan.curvature_tangent_dimension = estimate.tangent_dimension;
    plan.reference_curvature_max = estimate.reference_curvature_max;
    if (!estimate.valid || plan.recovery_rank <= 0 ||
        estimate.recovery_curvatures.size() != plan.recovery_rank ||
        estimate.recovery_eigenvectors.rows() != plan.recovery_rank ||
        estimate.recovery_eigenvectors.cols() != plan.recovery_rank ||
        !estimate.recovery_curvatures.allFinite() ||
        !estimate.recovery_eigenvectors.allFinite() ||
        !std::isfinite(estimate.reference_curvature_max) ||
        estimate.reference_curvature_max < config.curvature_min_reference)
    {
      plan.eligible = false;
      plan.state = CasrInterventionState::CurvatureInvalid;
      return false;
    }

    plan.recovery_basis_rotation = estimate.recovery_eigenvectors;
    plan.recovery_curvature_min = estimate.recovery_curvatures.minCoeff();
    plan.recovery_curvature_median = Median(estimate.recovery_curvatures);
    plan.recovery_curvature_max = estimate.recovery_curvatures.maxCoeff();
    plan.target_curvature =
        config.curvature_target_relative_to_max *
        estimate.reference_curvature_max;
    const double maximum_added_information =
        config.curvature_max_added_relative_to_max *
        estimate.reference_curvature_max;

    plan.effective_information_weights = Eigen::VectorXd::Zero(
        plan.recovery_rank);
    for (int i = 0; i < plan.recovery_rank; ++i)
    {
      const double missing_curvature = std::max(
          0.0, plan.target_curvature - estimate.recovery_curvatures[i]);
      plan.effective_information_weights[i] =
          std::min(maximum_added_information,
                   plan.used_activation_strength * config.curvature_gain *
                       missing_curvature);
    }
    if (!plan.effective_information_weights.allFinite())
    {
      plan.eligible = false;
      plan.state = CasrInterventionState::CurvatureInvalid;
      return false;
    }

    plan.added_information_min =
        plan.effective_information_weights.minCoeff();
    plan.added_information_median =
        Median(plan.effective_information_weights);
    plan.added_information_max =
        plan.effective_information_weights.maxCoeff();
    plan.effective_information_weight = plan.added_information_max;
    plan.sqrt_information_weights =
        plan.effective_information_weights.array().max(0.0).sqrt().matrix();
    plan.sqrt_information_weight =
        plan.sqrt_information_weights.maxCoeff();
    if (!std::isfinite(plan.sqrt_information_weight) ||
        plan.sqrt_information_weight <= 0.0)
    {
      plan.eligible = false;
      plan.state = CasrInterventionState::CurvatureSufficient;
      return false;
    }

    plan.state = config.apply_to_estimator
                     ? CasrInterventionState::Applied
                     : CasrInterventionState::DryRun;
    return true;
  }

} // namespace cocolic
