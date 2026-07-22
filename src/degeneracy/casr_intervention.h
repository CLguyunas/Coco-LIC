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
#include <degeneracy/dso_fixed_config.h>

#include <yaml-cpp/yaml.h>

#include <cstdint>

namespace cocolic
{

  inline constexpr char kCasrInterventionMethodVersion[] =
      "cause_differential_recovery_v3";

  enum class CasrRecoveryMechanism : int
  {
    None = 0,
    PropagationReference = 1,
    SplineIncrementContinuity = 2,
    CoupledSourceConsensus = 3
  };

  const char *CasrRecoveryMechanismName(CasrRecoveryMechanism mechanism);

  enum class CasrRecoveryReference : int
  {
    None = 0,
    ImuPriorPropagation = 1,
    NonuniformSplineContinuity = 2,
    PropagationAndSplineConsensus = 3
  };

  const char *CasrRecoveryReferenceName(CasrRecoveryReference reference);

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
    SolverFailureRecovered = 14,
    CurvatureInvalid = 15,
    CurvatureSufficient = 16,
    InvalidTemporalSupport = 17,
    SourceConsensusInsufficient = 18,
    SourceConsensusConflict = 19
  };

  const char *CasrInterventionStateName(CasrInterventionState state);

  struct CasrInterventionConfig
  {
    // enabled creates the intervention audit stream.  The estimator is
    // changed only when apply_to_estimator is also true.
    bool enabled = false;
    bool apply_to_estimator = false;
    bool output_csv = dso_fixed::kInterventionOutputCsv;

    // Internal legacy fallback retained for unit-level compatibility. The
    // production YAML path always enables curvature matching.
    double base_information_weight =
        dso_fixed::kLegacyBaseInformationWeight;
    double max_effective_information_weight =
        dso_fixed::kLegacyMaxEffectiveInformationWeight;

    // Match every recovery direction to the curvature of the actual final
    // LIC problem before the intervention factor is added.  Curvatures are
    // expressed in the same metric [r*dtheta, dp] used by the CASR basis.
    bool curvature_matching_enabled = dso_fixed::kCurvatureMatchingEnabled;
    double curvature_target_relative_to_max =
        dso_fixed::kCurvatureTargetRelativeToMax;
    double curvature_max_added_relative_to_max =
        dso_fixed::kCurvatureMaxAddedRelativeToMax;
    // The only estimator-strength hyperparameter exposed in YAML.
    double curvature_gain = dso_fixed::kDefaultCurvatureGain;
    double curvature_min_reference = dso_fixed::kCurvatureMinReference;

    // When armed, solve the unmodified and CASR problems from the exact same
    // parameter snapshot. The baseline solution is retained as a safe
    // fallback if the CASR solve is unusable.
    bool counterfactual_validation = dso_fixed::kCounterfactualValidation;
    double counterfactual_ratio_denominator_floor =
        dso_fixed::kCounterfactualRatioDenominatorFloor;
    double min_activation_strength = dso_fixed::kMinActivationStrength;
    double max_activation_strength = dso_fixed::kMaxActivationStrength;
    int max_control_points = dso_fixed::kInterventionMaxControlPoints;
    int max_recovery_rank = dso_fixed::kMaxRecoveryRank;
    double max_basis_orthogonality_error =
        dso_fixed::kMaxBasisOrthogonalityError;
  };

  struct CasrCurvatureEstimate
  {
    bool valid = false;
    int tangent_dimension = 0;
    double reference_curvature_max = 0.0;
    Eigen::VectorXd recovery_curvatures;
    // Columns rotate the original orthonormal recovery basis into the
    // eigen-directions of its projected LIC curvature.
    Eigen::MatrixXd recovery_eigenvectors;
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
    Eigen::VectorXd effective_information_weights;
    Eigen::VectorXd sqrt_information_weights;
    Eigen::MatrixXd recovery_basis_rotation;

    CasrRecoveryMechanism recovery_mechanism =
        CasrRecoveryMechanism::None;
    CasrRecoveryReference recovery_reference =
        CasrRecoveryReference::None;

    bool curvature_matching_enabled = false;
    bool curvature_valid = false;
    int curvature_tangent_dimension = 0;
    double reference_curvature_max = 0.0;
    double target_curvature = 0.0;
    double recovery_curvature_min = 0.0;
    double recovery_curvature_median = 0.0;
    double recovery_curvature_max = 0.0;
    double added_information_min = 0.0;
    double added_information_median = 0.0;
    double added_information_max = 0.0;
  };

  CasrInterventionPlan BuildCasrInterventionPlan(
      const CasrInterventionConfig &config,
      const CasrShadowResult &casr_result,
      double characteristic_range,
      int trajectory_control_point_num);

  // Converts an already safety-gated plan into independently weighted
  // recovery eigen-directions. Returns false and fails closed when the
  // estimator curvature cannot be trusted.
  bool FinalizeCasrInterventionPlanWithCurvature(
      const CasrInterventionConfig &config,
      const CasrCurvatureEstimate &estimate,
      CasrInterventionPlan &plan);

  struct CasrSourceConsensus
  {
    bool evaluated = false;
    bool sufficient = false;
    bool consistent = false;
    double environment_norm = 0.0;
    double support_norm = 0.0;
    double cosine = 0.0;
  };

  CasrSourceConsensus EvaluateCasrSourceConsensus(
      const Eigen::VectorXd &environment_coordinates,
      const Eigen::VectorXd &support_coordinates);

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
    CasrRecoveryMechanism recovery_mechanism =
        CasrRecoveryMechanism::None;
    CasrRecoveryReference recovery_reference =
        CasrRecoveryReference::None;
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

    bool curvature_matching_enabled = false;
    bool curvature_valid = false;
    int curvature_tangent_dimension = 0;
    double reference_curvature_max = 0.0;
    double target_curvature = 0.0;
    double recovery_curvature_min = 0.0;
    double recovery_curvature_median = 0.0;
    double recovery_curvature_max = 0.0;
    double added_information_min = 0.0;
    double added_information_median = 0.0;
    double added_information_max = 0.0;

    bool continuity_operator_valid = false;
    int continuity_operator_rank = 0;
    double continuity_symmetry_error = 0.0;
    double continuity_idempotence_error = 0.0;
    double pre_environment_residual_norm = 0.0;
    double pre_support_residual_norm = 0.0;
    double post_environment_residual_norm = 0.0;
    double post_support_residual_norm = 0.0;
    double counterfactual_environment_residual_norm = 0.0;
    double counterfactual_support_residual_norm = 0.0;
    bool source_consensus_evaluated = false;
    bool source_consensus_sufficient = false;
    bool source_consensus_consistent = false;
    double source_consensus_cosine = 0.0;

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

    bool counterfactual_enabled = false;
    bool counterfactual_solver_usable = false;
    int counterfactual_solver_successful_steps = 0;
    int counterfactual_solver_unsuccessful_steps = 0;
    double counterfactual_total_increment_norm = 0.0;
    double counterfactual_projected_increment_norm = 0.0;
    double counterfactual_orthogonal_increment_norm = 0.0;
    double counterfactual_factor_residual_norm = 0.0;
    double projected_casr_over_counterfactual = 0.0;
    double orthogonal_casr_over_counterfactual = 0.0;

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
