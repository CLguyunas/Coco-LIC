#include <degeneracy/casr_shadow.h>

#include <gtest/gtest.h>

#include <cmath>

namespace cocolic
{
  namespace
  {
    CasrShadowResult EvaluateSupportCandidate(
        CasrShadowEvaluator &evaluator, CasrTemporalState &state,
        int control_point_start_index,
        const Eigen::MatrixXd &support_basis,
        double scan_timestamp_s = -1.0,
        double support_quality_min = 0.0)
    {
      const int dimension = static_cast<int>(support_basis.rows());
      Eigen::MatrixXd information =
          Eigen::MatrixXd::Identity(dimension, dimension);
      Eigen::MatrixXd cross = Eigen::MatrixXd::Zero(dimension, 6);
      Eigen::MatrixXd representative =
          Eigen::MatrixXd::Zero(6, dimension);
      representative.leftCols(6).setIdentity();

      CasrShadowInput input;
      input.environment_valid = true;
      input.support_valid = true;
      input.environment_state = false;
      input.support_state = true;
      input.scan_timestamp_s = scan_timestamp_s >= 0.0
                                   ? scan_timestamp_s
                                   : 0.1 * control_point_start_index;
      input.support_quality_min = support_quality_min;
      input.support_control_point_start_index =
          control_point_start_index;
      input.support_knot_weak_basis = &support_basis;
      input.reference_pose_information = &information;
      input.reference_pose_cross = &cross;
      input.representative_pose_mapping = &representative;
      return evaluator.Evaluate(input, &state);
    }

    Eigen::MatrixXd ConstantKnotDirection(int control_point_num)
    {
      Eigen::MatrixXd basis =
          Eigen::MatrixXd::Zero(6 * control_point_num, 1);
      const double weight = 1.0 / std::sqrt(control_point_num);
      for (int i = 0; i < control_point_num; ++i)
      {
        basis(6 * i, 0) = weight;
      }
      return basis;
    }

    Eigen::MatrixXd CoordinateBasis(int dimension, int first_axis, int rank)
    {
      Eigen::MatrixXd basis = Eigen::MatrixXd::Zero(dimension, rank);
      for (int i = 0; i < rank; ++i)
      {
        basis(first_axis + i, i) = 1.0;
      }
      return basis;
    }

    CasrShadowResult EvaluateCoupledCandidate(
        CasrShadowEvaluator &evaluator, CasrTemporalState &state,
        const Eigen::MatrixXd &support_basis, double scan_timestamp_s)
    {
      const int dimension = static_cast<int>(support_basis.rows());
      Eigen::MatrixXd information =
          Eigen::MatrixXd::Identity(dimension, dimension);
      Eigen::MatrixXd cross = Eigen::MatrixXd::Zero(dimension, 6);
      cross(0, 0) = 1.0;
      Eigen::MatrixXd representative =
          Eigen::MatrixXd::Zero(6, dimension);
      representative.leftCols(6).setIdentity();

      CasrShadowInput input;
      input.environment_valid = true;
      input.support_valid = true;
      input.environment_state = true;
      input.support_state = true;
      input.scan_timestamp_s = scan_timestamp_s;
      input.support_quality_min = 0.0;
      input.environment_relative_eigenvalues.setOnes();
      input.environment_relative_eigenvalues[0] = 1e-3;
      input.environment_eigenvectors.setIdentity();
      input.support_control_point_start_index = 0;
      input.support_knot_weak_basis = &support_basis;
      input.reference_pose_information = &information;
      input.reference_pose_cross = &cross;
      input.representative_pose_mapping = &representative;
      return evaluator.Evaluate(input, &state);
    }
  } // namespace

  TEST(CasrShadowAudit, PreservesUngatedSupportBasisWithoutRoutingIt)
  {
    CasrShadowConfig config;
    config.enabled = true;
    CasrShadowEvaluator evaluator;
    evaluator.Configure(config);

    const Eigen::MatrixXd support_basis = ConstantKnotDirection(4);
    const int dimension = static_cast<int>(support_basis.rows());
    const Eigen::MatrixXd information =
        Eigen::MatrixXd::Identity(dimension, dimension);
    const Eigen::MatrixXd cross = Eigen::MatrixXd::Zero(dimension, 6);
    Eigen::MatrixXd representative = Eigen::MatrixXd::Zero(6, dimension);
    representative.leftCols(6).setIdentity();

    CasrShadowInput input;
    input.environment_valid = true;
    input.support_valid = true;
    input.environment_state = false;
    input.support_state = false;
    input.support_quality_min = 0.8;
    input.environment_relative_eigenvalues.setOnes();
    input.environment_eigenvectors.setIdentity();
    input.support_control_point_start_index = 10;
    input.support_knot_weak_basis = &support_basis;
    input.reference_pose_information = &information;
    input.reference_pose_cross = &cross;
    input.representative_pose_mapping = &representative;

    const CasrShadowResult result = evaluator.Evaluate(input, nullptr);
    ASSERT_TRUE(result.valid);
    EXPECT_EQ(result.cause, 0);
    EXPECT_EQ(result.route, CasrRoute::Inactive);
    EXPECT_EQ(result.support_rank, 0);
    EXPECT_EQ(result.support_knot_basis.cols(), 0);
    ASSERT_EQ(result.support_audit_knot_basis.cols(), 1);
    const Eigen::MatrixXd audit_gram =
        result.support_audit_knot_basis.transpose() *
        result.support_audit_knot_basis;
    EXPECT_NEAR(audit_gram(0, 0), 1.0, 1e-12);
    EXPECT_EQ(result.recovery_rank, 0);
    EXPECT_EQ(result.recovery_knot_basis.cols(), 0);
  }

  TEST(CasrShadowTemporalGate, IsInvariantToRollingWindowBoundaryLoss)
  {
    CasrShadowConfig config;
    config.enabled = true;
    config.projector_consecutive_scans = 3;
    config.projector_similarity_threshold = 0.8;
    CasrShadowEvaluator evaluator;
    evaluator.Configure(config);
    CasrTemporalState state;
    const Eigen::MatrixXd basis = ConstantKnotDirection(4);

    const CasrShadowResult first =
        EvaluateSupportCandidate(evaluator, state, 10, basis);
    EXPECT_EQ(first.route, CasrRoute::SupportCandidate);
    EXPECT_DOUBLE_EQ(first.temporal_projector_similarity, 1.0);
    EXPECT_EQ(first.temporal_overlap_control_point_num, 0);
    EXPECT_EQ(first.projector_consistency_count, 1);
    EXPECT_FALSE(first.recovery_ready);

    const CasrShadowResult second =
        EvaluateSupportCandidate(evaluator, state, 11, basis);
    EXPECT_NEAR(second.temporal_projector_similarity, 1.0, 1e-12);
    EXPECT_EQ(second.temporal_overlap_control_point_num, 3);
    EXPECT_EQ(second.temporal_previous_overlap_rank, 1);
    EXPECT_EQ(second.temporal_current_overlap_rank, 1);
    EXPECT_EQ(second.temporal_forced_overlap_rank, 0);
    EXPECT_NEAR(second.temporal_projector_affinity, 1.0, 1e-12);
    EXPECT_EQ(second.projector_consistency_count, 2);
    EXPECT_FALSE(second.recovery_ready);

    const CasrShadowResult third =
        EvaluateSupportCandidate(evaluator, state, 12, basis);
    EXPECT_NEAR(third.temporal_projector_similarity, 1.0, 1e-12);
    EXPECT_EQ(third.temporal_overlap_control_point_num, 3);
    EXPECT_EQ(third.temporal_previous_overlap_rank, 1);
    EXPECT_EQ(third.temporal_current_overlap_rank, 1);
    EXPECT_EQ(third.temporal_forced_overlap_rank, 0);
    EXPECT_NEAR(third.temporal_projector_affinity, 1.0, 1e-12);
    EXPECT_EQ(third.projector_consistency_count, 3);
    EXPECT_TRUE(third.recovery_ready);
  }

  TEST(CasrShadowProvenance, PreservesDiagnosticsCopyOnInvalidInput)
  {
    CasrShadowConfig config;
    config.enabled = true;
    CasrShadowEvaluator evaluator;
    evaluator.Configure(config);
    CasrShadowInput input;
    input.data_source = CasrDataSource::DiagnosticsCopy;

    const CasrShadowResult result = evaluator.Evaluate(input, nullptr);
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.data_source, CasrDataSource::DiagnosticsCopy);
  }

  TEST(CasrShadowTemporalGate, RejectsDisjointGlobalKnotWindows)
  {
    CasrShadowConfig config;
    config.enabled = true;
    config.projector_consecutive_scans = 2;
    config.projector_similarity_threshold = 0.8;
    CasrShadowEvaluator evaluator;
    evaluator.Configure(config);
    CasrTemporalState state;
    const Eigen::MatrixXd basis = ConstantKnotDirection(4);

    EvaluateSupportCandidate(evaluator, state, 0, basis);
    const CasrShadowResult disjoint =
        EvaluateSupportCandidate(evaluator, state, 10, basis);
    EXPECT_DOUBLE_EQ(disjoint.temporal_projector_similarity, 0.0);
    EXPECT_EQ(disjoint.temporal_overlap_control_point_num, 0);
    EXPECT_EQ(disjoint.projector_consistency_count, 1);
    EXPECT_FALSE(disjoint.recovery_ready);
  }

  TEST(CasrShadowTemporalGate, RemovesForcedHighRankIntersection)
  {
    CasrShadowConfig config;
    config.enabled = true;
    config.projector_consecutive_scans = 2;
    config.projector_similarity_threshold = 0.8;
    CasrShadowEvaluator evaluator;
    evaluator.Configure(config);
    CasrTemporalState state;

    // In R^18, two rank-12 subspaces must share at least six dimensions.
    // That forced intersection is not evidence of temporal continuity.
    const Eigen::MatrixXd first_basis = CoordinateBasis(18, 0, 12);
    const Eigen::MatrixXd second_basis = CoordinateBasis(18, 6, 12);
    EvaluateSupportCandidate(evaluator, state, 0, first_basis);
    const CasrShadowResult forced_only =
        EvaluateSupportCandidate(evaluator, state, 0, second_basis);
    EXPECT_EQ(forced_only.temporal_overlap_control_point_num, 3);
    EXPECT_EQ(forced_only.temporal_previous_overlap_rank, 12);
    EXPECT_EQ(forced_only.temporal_current_overlap_rank, 12);
    EXPECT_EQ(forced_only.temporal_forced_overlap_rank, 6);
    EXPECT_NEAR(forced_only.temporal_projector_affinity, 6.0, 1e-12);
    EXPECT_NEAR(forced_only.temporal_projector_similarity, 0.0, 1e-12);
    EXPECT_EQ(forced_only.projector_consistency_count, 1);
    EXPECT_FALSE(forced_only.recovery_ready);

    const CasrShadowResult identical =
        EvaluateSupportCandidate(evaluator, state, 0, second_basis);
    EXPECT_EQ(identical.temporal_forced_overlap_rank, 6);
    EXPECT_NEAR(identical.temporal_projector_affinity, 12.0, 1e-12);
    EXPECT_NEAR(identical.temporal_projector_similarity, 1.0, 1e-12);
    EXPECT_EQ(identical.projector_consistency_count, 2);
    EXPECT_TRUE(identical.recovery_ready);
  }

  TEST(CasrShadowScheduler, SuppressesThresholdEdgeReadyCandidate)
  {
    CasrShadowConfig config;
    config.enabled = true;
    config.projector_consecutive_scans = 3;
    config.projector_similarity_threshold = 0.8;
    config.scheduler_enabled = true;
    config.scheduler_projector_full_confidence = 0.95;
    config.scheduler_persistence_full_scans = 5;
    config.scheduler_enter_confidence = 0.25;
    CasrShadowEvaluator evaluator;
    evaluator.Configure(config);
    CasrTemporalState state;

    Eigen::MatrixXd stable = Eigen::MatrixXd::Zero(24, 1);
    stable(0, 0) = 1.0;
    Eigen::MatrixXd marginal = Eigen::MatrixXd::Zero(24, 1);
    const double similarity = 0.8005;
    marginal(0, 0) = std::sqrt(similarity);
    marginal(1, 0) = std::sqrt(1.0 - similarity);

    EvaluateSupportCandidate(evaluator, state, 0, stable, 0.0);
    EvaluateSupportCandidate(evaluator, state, 0, stable, 0.1);
    const CasrShadowResult result =
        EvaluateSupportCandidate(evaluator, state, 0, marginal, 0.2);

    EXPECT_TRUE(result.recovery_ready);
    EXPECT_TRUE(result.scheduler_eligible);
    EXPECT_NEAR(result.temporal_projector_similarity,
                similarity, 1e-12);
    EXPECT_NEAR(result.scheduler_temporal_confidence,
                (similarity - 0.8) / 0.15, 1e-12);
    EXPECT_LT(result.scheduler_raw_confidence,
              config.scheduler_enter_confidence);
    EXPECT_EQ(result.scheduler_state,
              CasrSchedulerState::BelowEnterConfidence);
    EXPECT_FALSE(result.scheduler_active);
    EXPECT_DOUBLE_EQ(result.scheduler_activation_strength, 0.0);
  }

  TEST(CasrShadowScheduler, RampsCauseAwareStrengthUsingElapsedTime)
  {
    CasrShadowConfig config;
    config.enabled = true;
    config.projector_consecutive_scans = 3;
    config.projector_similarity_threshold = 0.8;
    config.scheduler_enabled = true;
    config.scheduler_persistence_full_scans = 5;
    config.scheduler_enter_confidence = 0.25;
    config.scheduler_exit_confidence = 0.1;
    config.scheduler_rise_time_s = 0.5;
    config.scheduler_fall_time_s = 0.2;
    CasrShadowEvaluator evaluator;
    evaluator.Configure(config);
    CasrTemporalState state;
    const Eigen::MatrixXd basis = ConstantKnotDirection(4);

    EvaluateSupportCandidate(evaluator, state, 0, basis, 0.0);
    EvaluateSupportCandidate(evaluator, state, 0, basis, 0.1);
    const CasrShadowResult third =
        EvaluateSupportCandidate(evaluator, state, 0, basis, 0.2);
    EXPECT_TRUE(third.scheduler_eligible);
    EXPECT_NEAR(third.scheduler_cause_confidence, 1.0, 1e-12);
    EXPECT_NEAR(third.scheduler_temporal_confidence, 1.0, 1e-12);
    EXPECT_NEAR(third.scheduler_persistence_confidence, 1.0 / 3.0,
                1e-12);
    EXPECT_NEAR(third.scheduler_target_strength, 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(third.scheduler_activation_strength, 0.2, 1e-12);
    EXPECT_EQ(third.scheduler_state, CasrSchedulerState::Ramping);

    const CasrShadowResult fourth =
        EvaluateSupportCandidate(evaluator, state, 0, basis, 0.3);
    EXPECT_NEAR(fourth.scheduler_target_strength, 2.0 / 3.0, 1e-12);
    EXPECT_NEAR(fourth.scheduler_activation_strength, 0.4, 1e-12);

    const CasrShadowResult fifth =
        EvaluateSupportCandidate(evaluator, state, 0, basis, 0.4);
    EXPECT_NEAR(fifth.scheduler_target_strength, 1.0, 1e-12);
    EXPECT_NEAR(fifth.scheduler_activation_strength, 0.6, 1e-12);

    EvaluateSupportCandidate(evaluator, state, 0, basis, 0.5);
    const CasrShadowResult fully_active =
        EvaluateSupportCandidate(evaluator, state, 0, basis, 0.6);
    EXPECT_EQ(fully_active.scheduler_state, CasrSchedulerState::Active);
    EXPECT_NEAR(fully_active.scheduler_activation_strength, 1.0, 1e-12);

    const CasrShadowResult falling = EvaluateSupportCandidate(
        evaluator, state, 0, basis, 0.7, 0.049);
    EXPECT_TRUE(falling.scheduler_eligible);
    EXPECT_LT(falling.scheduler_raw_confidence,
              config.scheduler_exit_confidence);
    EXPECT_EQ(falling.scheduler_state,
              CasrSchedulerState::BelowExitConfidence);
    EXPECT_NEAR(falling.scheduler_target_strength, 0.0, 1e-12);
    EXPECT_NEAR(falling.scheduler_activation_strength, 0.5, 1e-12);

    const CasrShadowResult faded = EvaluateSupportCandidate(
        evaluator, state, 0, basis, 0.8, 0.049);
    EXPECT_EQ(faded.scheduler_state,
              CasrSchedulerState::BelowEnterConfidence);
    EXPECT_FALSE(faded.scheduler_active);
    EXPECT_NEAR(faded.scheduler_activation_strength, 0.0, 1e-12);

    CasrShadowInput invalid;
    invalid.environment_valid = true;
    invalid.support_valid = false;
    invalid.scan_timestamp_s = 0.9;
    const CasrShadowResult reset = evaluator.Evaluate(invalid, &state);
    EXPECT_FALSE(reset.valid);
    EXPECT_EQ(reset.scheduler_state, CasrSchedulerState::InvalidInput);
    EXPECT_FALSE(reset.scheduler_active);
    EXPECT_DOUBLE_EQ(reset.scheduler_activation_strength, 0.0);
    EXPECT_FALSE(state.scheduler_active);
    EXPECT_DOUBLE_EQ(state.scheduler_strength, 0.0);
  }

  TEST(CasrShadowScheduler, BlocksConflictAndScoresCoupledCommonDirection)
  {
    CasrShadowConfig config;
    config.enabled = true;
    config.projector_consecutive_scans = 3;
    config.scheduler_enabled = true;
    config.scheduler_persistence_full_scans = 5;
    CasrShadowEvaluator evaluator;
    evaluator.Configure(config);

    CasrTemporalState conflict_state;
    Eigen::MatrixXd orthogonal = Eigen::MatrixXd::Zero(24, 1);
    orthogonal(1, 0) = 1.0;
    const CasrShadowResult conflict = EvaluateCoupledCandidate(
        evaluator, conflict_state, orthogonal, 0.0);
    EXPECT_EQ(conflict.route, CasrRoute::CoupledConflict);
    EXPECT_EQ(conflict.scheduler_state, CasrSchedulerState::UnsafeRoute);
    EXPECT_FALSE(conflict.scheduler_eligible);
    EXPECT_DOUBLE_EQ(conflict.scheduler_activation_strength, 0.0);

    CasrTemporalState common_state;
    Eigen::MatrixXd common = Eigen::MatrixXd::Zero(24, 1);
    common(0, 0) = 0.8;
    common(1, 0) = 0.6;
    EvaluateCoupledCandidate(evaluator, common_state, common, 0.0);
    EvaluateCoupledCandidate(evaluator, common_state, common, 0.1);
    const CasrShadowResult ready = EvaluateCoupledCandidate(
        evaluator, common_state, common, 0.2);
    EXPECT_EQ(ready.route, CasrRoute::CoupledCommonCandidate);
    EXPECT_TRUE(ready.recovery_ready);
    EXPECT_NEAR(ready.principal_cosine_max, 0.8, 1e-6);
    EXPECT_NEAR(ready.scheduler_principal_confidence, 0.5, 1e-6);
    EXPECT_NEAR(ready.scheduler_cause_confidence, 0.5, 1e-6);
    EXPECT_NEAR(ready.scheduler_raw_confidence, 1.0 / 3.0, 1e-6);
    EXPECT_NEAR(ready.scheduler_activation_strength, 0.2, 1e-6);
  }
} // namespace cocolic
