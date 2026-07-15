/*
 * Cause-adaptive subspace recovery (CASR) estimator intervention.
 *
 * This layer consumes the independently audited CASR shadow result.  It does
 * not decide the recovery direction itself; it only validates whether that
 * result is safe to use and converts the scheduler strength into a bounded
 * information weight.
 */

#pragma once

#include <degeneracy/casr_shadow.h>

#include <yaml-cpp/yaml.h>

#include <cstdint>

namespace cocolic
{

  inline constexpr char kCasrInterventionMethodVersion[] =
      "knot_space_increment_anchor_v1";

  enum class CasrInterventionState : int
  {
    Disabled = 0,
    InvalidInput = 1,
    UnsafeRoute = 2,
    RouteMismatch = 3,
    NotReady = 4,
    SchedulerInactive = 5,
    ActivationTooLow = 6,
    InvalidBasis = 7,
    InvalidControlRange = 8,
    MissingReference = 9,
    DryRun = 10,
    Applied = 11,
    SolverFailure = 12,
    DiagnosticsCopyBlocked = 13,
    SolverFailureRecovered = 14
  };

  const char *CasrInterventionStateName(CasrInterventionState state);

  struct CasrInterventionConfig
  {
    // enabled creates the intervention audit stream.  The estimator is
    // changed only when apply_to_estimator is also true.
    bool enabled = false;
    bool apply_to_estimator = false;
    bool output_csv = true;

    // The factor cost is
    //   0.5 * base_information_weight * activation * ||B^T D delta||^2,
    // where D delta stacks [r*dtheta, dp] for every active control point.
    double base_information_weight = 1.0;
    double max_effective_information_weight = 100.0;
    double min_activation_strength = 5e-2;
    double max_activation_strength = 1.0;
    int max_control_points = 32;
    int max_recovery_rank = 32;
    double max_basis_orthogonality_error = 1e-6;
  };

  CasrInterventionConfig ReadCasrInterventionConfig(
      const YAML::Node &node);

  struct CasrInterventionPlan
  {
    CasrInterventionState state = CasrInterventionState::Disabled;
    bool eligible = false;
    int control_point_start_index = -1;
    int control_point_num = 0;
    int recovery_rank = 0;
    double requested_activation_strength = 0.0;
    double used_activation_strength = 0.0;
    double characteristic_range = 1.0;
    double effective_information_weight = 0.0;
    double sqrt_information_weight = 0.0;
  };

  CasrInterventionPlan BuildCasrInterventionPlan(
      const CasrInterventionConfig &config,
      const CasrShadowResult &casr_result,
      double characteristic_range,
      int trajectory_control_point_num);

  struct CasrInterventionReport
  {
    CasrInterventionState state = CasrInterventionState::Disabled;
    bool enabled = false;
    bool apply_to_estimator = false;
    bool eligible = false;
    bool factor_added = false;
    bool applied = false;
    int64_t scan_timestamp_ns = 0;
    CasrRoute route = CasrRoute::Invalid;
    CasrDataSource data_source = CasrDataSource::RealMeasurements;
    int control_point_start_index = -1;
    int control_point_num = 0;
    int recovery_rank = 0;
    double requested_activation_strength = 0.0;
    double used_activation_strength = 0.0;
    double characteristic_range = 1.0;
    double base_information_weight = 0.0;
    double max_effective_information_weight = 0.0;
    double effective_information_weight = 0.0;
    double sqrt_information_weight = 0.0;

    double pre_total_increment_norm = 0.0;
    double pre_projected_increment_norm = 0.0;
    double pre_orthogonal_increment_norm = 0.0;
    double pre_factor_residual_norm = 0.0;
    double post_total_increment_norm = 0.0;
    double post_projected_increment_norm = 0.0;
    double post_orthogonal_increment_norm = 0.0;
    double post_factor_residual_norm = 0.0;
    double max_rotation_increment_rad = 0.0;
    double max_translation_increment_m = 0.0;

    bool solver_usable = false;
    int solver_successful_steps = 0;
    int solver_unsuccessful_steps = 0;
    bool primary_solver_usable = false;
    int primary_solver_successful_steps = 0;
    int primary_solver_unsuccessful_steps = 0;
    bool fallback_attempted = false;
    bool fallback_solver_usable = false;
  };

} // namespace cocolic
