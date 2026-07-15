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
        const Eigen::MatrixXd &support_basis)
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
  } // namespace

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
} // namespace cocolic
