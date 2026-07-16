#include <degeneracy/casr_intervention.h>
#include <odom/factor/analytic_diff/casr_subspace_factor.h>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace cocolic
{
  namespace
  {
    CasrShadowResult ReadyEnvironmentResult()
    {
      CasrShadowResult result;
      result.valid = true;
      result.route = CasrRoute::EnvironmentCandidate;
      result.stable_route = result.route;
      result.recovery_ready = true;
      result.scheduler_eligible = true;
      result.scheduler_active = true;
      result.scheduler_activation_strength = 0.6;
      result.support_control_point_start_index = 4;
      result.support_knot_dimension = 12;
      result.recovery_rank = 1;
      result.recovery_knot_basis = Eigen::MatrixXd::Zero(12, 1);
      result.recovery_knot_basis(0, 0) = 1.0;
      result.recovery_basis_orthogonality_error = 0.0;
      return result;
    }
  } // namespace

  TEST(CasrInterventionConfig, ExposesOnlyModeSwitchesAndCurvatureGain)
  {
    const CasrInterventionConfig defaults =
        ReadCasrInterventionConfig(YAML::Node());
    EXPECT_DOUBLE_EQ(defaults.curvature_gain,
                     dso_fixed::kDefaultCurvatureGain);

    const YAML::Node node = YAML::Load(R"(
enabled: true
apply_to_estimator: true
curvature_gain: 0.02
output_csv: false
curvature_matching_enabled: false
curvature_target_relative_to_max: 0.9
curvature_max_added_relative_to_max: 0.9
curvature_min_reference: 1.0
counterfactual_validation: false
counterfactual_ratio_denominator_floor: 1.0
base_information_weight: 99.0
max_effective_information_weight: 99.0
min_activation_strength: 0.9
max_activation_strength: 0.9
max_control_points: 4
max_recovery_rank: 1
max_basis_orthogonality_error: 0.5
)");

    const CasrInterventionConfig config =
        ReadCasrInterventionConfig(node);
    EXPECT_TRUE(config.enabled);
    EXPECT_TRUE(config.apply_to_estimator);
    EXPECT_DOUBLE_EQ(config.curvature_gain, 0.02);

    // Former YAML keys are implementation constants and cannot override the
    // production path.
    EXPECT_EQ(config.output_csv, dso_fixed::kInterventionOutputCsv);
    EXPECT_EQ(config.curvature_matching_enabled,
              dso_fixed::kCurvatureMatchingEnabled);
    EXPECT_DOUBLE_EQ(config.curvature_target_relative_to_max,
                     dso_fixed::kCurvatureTargetRelativeToMax);
    EXPECT_DOUBLE_EQ(config.curvature_max_added_relative_to_max,
                     dso_fixed::kCurvatureMaxAddedRelativeToMax);
    EXPECT_DOUBLE_EQ(config.curvature_min_reference,
                     dso_fixed::kCurvatureMinReference);
    EXPECT_EQ(config.counterfactual_validation,
              dso_fixed::kCounterfactualValidation);
    EXPECT_DOUBLE_EQ(config.counterfactual_ratio_denominator_floor,
                     dso_fixed::kCounterfactualRatioDenominatorFloor);
    EXPECT_DOUBLE_EQ(config.base_information_weight,
                     dso_fixed::kLegacyBaseInformationWeight);
    EXPECT_DOUBLE_EQ(config.max_effective_information_weight,
                     dso_fixed::kLegacyMaxEffectiveInformationWeight);
    EXPECT_DOUBLE_EQ(config.min_activation_strength,
                     dso_fixed::kMinActivationStrength);
    EXPECT_DOUBLE_EQ(config.max_activation_strength,
                     dso_fixed::kMaxActivationStrength);
    EXPECT_EQ(config.max_control_points,
              dso_fixed::kInterventionMaxControlPoints);
    EXPECT_EQ(config.max_recovery_rank, dso_fixed::kMaxRecoveryRank);
    EXPECT_DOUBLE_EQ(config.max_basis_orthogonality_error,
                     dso_fixed::kMaxBasisOrthogonalityError);
  }

  TEST(CasrInterventionPlan, RequiresExplicitEstimatorArming)
  {
    CasrInterventionConfig config;
    config.enabled = true;
    config.apply_to_estimator = false;
    config.curvature_matching_enabled = false;
    config.base_information_weight = 4.0;
    const CasrShadowResult result = ReadyEnvironmentResult();

    const CasrInterventionPlan dry_run =
        BuildCasrInterventionPlan(config, result, 2.0, 10);
    EXPECT_TRUE(dry_run.eligible);
    EXPECT_EQ(dry_run.state, CasrInterventionState::DryRun);
    EXPECT_EQ(dry_run.control_point_num, 2);
    EXPECT_NEAR(dry_run.effective_information_weight, 2.4, 1e-12);
    EXPECT_NEAR(dry_run.sqrt_information_weight, std::sqrt(2.4), 1e-12);

    config.apply_to_estimator = true;
    const CasrInterventionPlan armed =
        BuildCasrInterventionPlan(config, result, 2.0, 10);
    EXPECT_TRUE(armed.eligible);
    EXPECT_EQ(armed.state, CasrInterventionState::Applied);
  }

  TEST(CasrInterventionPlan, RejectsUnsafeAndStaleRoutes)
  {
    CasrInterventionConfig config;
    config.enabled = true;
    config.apply_to_estimator = true;
    config.curvature_matching_enabled = false;

    CasrShadowResult conflict = ReadyEnvironmentResult();
    conflict.route = CasrRoute::CoupledConflict;
    conflict.stable_route = conflict.route;
    const CasrInterventionPlan unsafe =
        BuildCasrInterventionPlan(config, conflict, 2.0, 10);
    EXPECT_FALSE(unsafe.eligible);
    EXPECT_EQ(unsafe.state, CasrInterventionState::UnsafeRoute);

    CasrShadowResult mismatch = ReadyEnvironmentResult();
    mismatch.stable_route = CasrRoute::Inactive;
    const CasrInterventionPlan stale =
        BuildCasrInterventionPlan(config, mismatch, 2.0, 10);
    EXPECT_FALSE(stale.eligible);
    EXPECT_EQ(stale.state, CasrInterventionState::RouteMismatch);
  }

  TEST(CasrInterventionPlan, CurvatureMatchingWeightsEachDirection)
  {
    CasrInterventionConfig config;
    config.enabled = true;
    config.apply_to_estimator = true;
    config.curvature_matching_enabled = true;
    config.curvature_target_relative_to_max = 6e-3;
    config.curvature_max_added_relative_to_max = 2e-2;

    CasrShadowResult result = ReadyEnvironmentResult();
    result.recovery_rank = 2;
    result.recovery_knot_basis = Eigen::MatrixXd::Zero(12, 2);
    result.recovery_knot_basis(0, 0) = 1.0;
    result.recovery_knot_basis(1, 1) = 1.0;
    CasrInterventionPlan plan =
        BuildCasrInterventionPlan(config, result, 2.0, 10);
    ASSERT_TRUE(plan.eligible);

    CasrCurvatureEstimate estimate;
    estimate.valid = true;
    estimate.tangent_dimension = 12;
    estimate.reference_curvature_max = 1000.0;
    estimate.recovery_curvatures = Eigen::Vector2d(1.0, 8.0);
    estimate.recovery_eigenvectors = Eigen::Matrix2d::Identity();
    ASSERT_TRUE(FinalizeCasrInterventionPlanWithCurvature(
        config, estimate, plan));
    ASSERT_EQ(plan.effective_information_weights.size(), 2);
    EXPECT_NEAR(plan.effective_information_weights[0], 3.0, 1e-12);
    EXPECT_DOUBLE_EQ(plan.effective_information_weights[1], 0.0);
    EXPECT_NEAR(plan.sqrt_information_weights[0], std::sqrt(3.0), 1e-12);
    EXPECT_DOUBLE_EQ(plan.sqrt_information_weights[1], 0.0);

    plan = BuildCasrInterventionPlan(config, result, 2.0, 10);
    estimate.recovery_curvatures = Eigen::Vector2d(7.0, 8.0);
    EXPECT_FALSE(FinalizeCasrInterventionPlanWithCurvature(
        config, estimate, plan));
    EXPECT_FALSE(plan.eligible);
    EXPECT_EQ(plan.state, CasrInterventionState::CurvatureSufficient);
  }

  TEST(CasrInterventionPlan, RejectsDiagnosticsCopyEvenWhenOtherwiseReady)
  {
    CasrInterventionConfig config;
    config.enabled = true;
    config.apply_to_estimator = true;
    CasrShadowResult result = ReadyEnvironmentResult();
    result.data_source = CasrDataSource::DiagnosticsCopy;

    const CasrInterventionPlan plan =
        BuildCasrInterventionPlan(config, result, 2.0, 10);
    EXPECT_FALSE(plan.eligible);
    EXPECT_EQ(plan.state,
              CasrInterventionState::DiagnosticsCopyBlocked);
  }

  TEST(CasrInterventionPlan, CapsInformationAndRechecksOrthonormality)
  {
    CasrInterventionConfig config;
    config.enabled = true;
    config.apply_to_estimator = true;
    config.curvature_matching_enabled = false;
    config.base_information_weight = 1000.0;
    config.max_effective_information_weight = 7.0;
    CasrShadowResult result = ReadyEnvironmentResult();

    const CasrInterventionPlan capped =
        BuildCasrInterventionPlan(config, result, 2.0, 10);
    ASSERT_TRUE(capped.eligible);
    EXPECT_DOUBLE_EQ(capped.effective_information_weight, 7.0);

    result.recovery_knot_basis(0, 0) = 2.0;
    // Do not trust the cached diagnostic scalar: the intervention must
    // recompute B^T B directly at the estimator boundary.
    result.recovery_basis_orthogonality_error = 0.0;
    const CasrInterventionPlan invalid =
        BuildCasrInterventionPlan(config, result, 2.0, 10);
    EXPECT_FALSE(invalid.eligible);
    EXPECT_EQ(invalid.state, CasrInterventionState::InvalidBasis);
  }

  TEST(CasrSubspaceFactor, PenalizesOnlyRecoveryCoordinates)
  {
    using analytic_derivative::CasrSubspaceFactor;
    using SO3d = Sophus::SO3<double>;

    Eigen::MatrixXd basis = Eigen::MatrixXd::Zero(6, 2);
    basis(0, 0) = 1.0;
    basis(3, 1) = 1.0;
    Eigen::aligned_vector<SO3d> reference_rotations(1, SO3d());
    Eigen::aligned_vector<Eigen::Vector3d> reference_positions(
        1, Eigen::Vector3d::Zero());
    CasrSubspaceFactor factor(basis, reference_rotations,
                              reference_positions, 3.0, 2.0);
    ASSERT_TRUE(factor.IsValid());

    Eigen::aligned_vector<SO3d> rotations(
        1, SO3d::exp(Eigen::Vector3d(0.1, 0.2, 0.0)));
    Eigen::aligned_vector<Eigen::Vector3d> positions(
        1, Eigen::Vector3d(0.4, 0.7, 0.0));
    std::vector<double const *> parameter_blocks{
        rotations[0].data(), positions[0].data()};
    Eigen::Vector2d residual;
    ASSERT_TRUE(factor.Evaluate(parameter_blocks.data(), residual.data(),
                                nullptr));
    EXPECT_NEAR(residual[0], 0.6, 1e-10);
    EXPECT_NEAR(residual[1], 0.8, 1e-12);
    // Rotation-y and translation-y are orthogonal to the selected subspace.
  }

  TEST(CasrSubspaceFactor, AnalyticRotationJacobianMatchesRightIncrement)
  {
    using analytic_derivative::CasrSubspaceFactor;
    using SO3d = Sophus::SO3<double>;

    Eigen::MatrixXd basis = Eigen::MatrixXd::Zero(6, 1);
    basis.block<3, 1>(0, 0) =
        Eigen::Vector3d(1.0, 2.0, -1.0).normalized();
    Eigen::aligned_vector<SO3d> reference_rotations(1, SO3d());
    Eigen::aligned_vector<Eigen::Vector3d> reference_positions(
        1, Eigen::Vector3d::Zero());
    CasrSubspaceFactor factor(basis, reference_rotations,
                              reference_positions, 2.5, 1.7);

    Eigen::aligned_vector<SO3d> rotations(
        1, SO3d::exp(Eigen::Vector3d(0.12, -0.08, 0.05)));
    Eigen::aligned_vector<Eigen::Vector3d> positions(
        1, Eigen::Vector3d::Zero());
    std::vector<double const *> parameter_blocks{
        rotations[0].data(), positions[0].data()};
    double residual = 0.0;
    Eigen::Matrix<double, 1, 4, Eigen::RowMajor> rotation_jacobian;
    Eigen::Matrix<double, 1, 3, Eigen::RowMajor> position_jacobian;
    double *jacobians[] = {rotation_jacobian.data(),
                           position_jacobian.data()};
    ASSERT_TRUE(factor.Evaluate(parameter_blocks.data(), &residual,
                                jacobians));

    const double epsilon = 1e-7;
    for (int axis = 0; axis < 3; ++axis)
    {
      Eigen::Vector3d increment = Eigen::Vector3d::Zero();
      increment[axis] = epsilon;
      SO3d perturbed = rotations[0] * SO3d::exp(increment);
      std::vector<double const *> perturbed_blocks{
          perturbed.data(), positions[0].data()};
      double perturbed_residual = 0.0;
      ASSERT_TRUE(factor.Evaluate(perturbed_blocks.data(),
                                  &perturbed_residual, nullptr));
      const double numerical =
          (perturbed_residual - residual) / epsilon;
      EXPECT_NEAR(rotation_jacobian(0, axis), numerical, 2e-6);
    }
    EXPECT_DOUBLE_EQ(rotation_jacobian(0, 3), 0.0);
  }

  TEST(CasrSubspaceFactor, AppliesIndependentDirectionWeights)
  {
    using analytic_derivative::CasrSubspaceFactor;
    using SO3d = Sophus::SO3<double>;

    Eigen::MatrixXd basis = Eigen::MatrixXd::Zero(6, 2);
    basis(0, 0) = 1.0;
    basis(3, 1) = 1.0;
    Eigen::aligned_vector<SO3d> reference_rotations(1, SO3d());
    Eigen::aligned_vector<Eigen::Vector3d> reference_positions(
        1, Eigen::Vector3d::Zero());
    CasrSubspaceFactor factor(
        basis, reference_rotations, reference_positions, 3.0,
        Eigen::Vector2d(2.0, 4.0));

    Eigen::aligned_vector<SO3d> rotations(
        1, SO3d::exp(Eigen::Vector3d(0.1, 0.0, 0.0)));
    Eigen::aligned_vector<Eigen::Vector3d> positions(
        1, Eigen::Vector3d(0.4, 0.0, 0.0));
    std::vector<double const *> parameter_blocks{
        rotations[0].data(), positions[0].data()};
    Eigen::Vector2d residual;
    ASSERT_TRUE(factor.Evaluate(parameter_blocks.data(), residual.data(),
                                nullptr));
    EXPECT_NEAR(residual[0], 0.6, 1e-10);
    EXPECT_NEAR(residual[1], 1.6, 1e-12);
  }

  TEST(CasrSubspaceFactor, UsesInterleavedKnotCoordinatesWithGroupedBlocks)
  {
    using analytic_derivative::CasrSubspaceFactor;
    using SO3d = Sophus::SO3<double>;

    // The Ceres blocks are [R0, R1, p0, p1], while the CASR basis rows are
    // [r*dtheta0, dp0, r*dtheta1, dp1]. Select p1.z explicitly.
    Eigen::MatrixXd basis = Eigen::MatrixXd::Zero(12, 1);
    basis(11, 0) = 1.0;
    Eigen::aligned_vector<SO3d> reference_rotations(2, SO3d());
    Eigen::aligned_vector<Eigen::Vector3d> reference_positions(
        2, Eigen::Vector3d::Zero());
    CasrSubspaceFactor factor(basis, reference_rotations,
                              reference_positions, 4.0, 3.0);

    Eigen::aligned_vector<SO3d> rotations(2, SO3d());
    Eigen::aligned_vector<Eigen::Vector3d> positions(
        2, Eigen::Vector3d::Zero());
    positions[0].z() = 9.0;
    positions[1].z() = 0.25;
    std::vector<double const *> parameter_blocks{
        rotations[0].data(), rotations[1].data(),
        positions[0].data(), positions[1].data()};
    double residual = 0.0;
    ASSERT_TRUE(factor.Evaluate(parameter_blocks.data(), &residual,
                                nullptr));
    EXPECT_NEAR(residual, 0.75, 1e-12);
  }

} // namespace cocolic
