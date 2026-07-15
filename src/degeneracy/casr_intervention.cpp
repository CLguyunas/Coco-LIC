/*
 * Cause-adaptive subspace recovery (CASR) estimator intervention.
 */

#include <degeneracy/casr_intervention.h>

#include <algorithm>
#include <cmath>

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
    default:
      return "unknown";
    }
  }

  CasrInterventionConfig ReadCasrInterventionConfig(
      const YAML::Node &node)
  {
    CasrInterventionConfig config;
    config.enabled = ReadValue<bool>(node, "enabled", false);
    config.apply_to_estimator =
        ReadValue<bool>(node, "apply_to_estimator", false);
    config.output_csv = ReadValue<bool>(node, "output_csv", true);
    config.base_information_weight = std::max(
        0.0, ReadFiniteDouble(node, "base_information_weight", 1.0));
    config.max_effective_information_weight = std::max(
        0.0, ReadFiniteDouble(
                 node, "max_effective_information_weight", 100.0));
    config.min_activation_strength = std::clamp(
        ReadFiniteDouble(node, "min_activation_strength", 5e-2),
        0.0, 1.0);
    config.max_activation_strength = std::clamp(
        ReadFiniteDouble(node, "max_activation_strength", 1.0),
        config.min_activation_strength, 1.0);
    config.max_control_points = std::max(
        4, ReadValue<int>(node, "max_control_points", 32));
    config.max_recovery_rank = std::max(
        1, ReadValue<int>(node, "max_recovery_rank", 32));
    config.max_basis_orthogonality_error = std::max(
        0.0, ReadFiniteDouble(
                 node, "max_basis_orthogonality_error", 1e-6));
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

    plan.eligible = true;
    plan.state = config.apply_to_estimator
                     ? CasrInterventionState::Applied
                     : CasrInterventionState::DryRun;
    return plan;
  }

} // namespace cocolic
