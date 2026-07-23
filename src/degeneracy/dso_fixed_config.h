/*
 * Fixed implementation constants for the DSO/CASR pipeline.
 *
 * These values are deliberately not YAML parameters. They describe logging,
 * numerical safeguards, capacity limits, temporal smoothing, or values that
 * are derived from the small set of published method hyperparameters.
 */

#pragma once

#include <cstdint>

namespace cocolic
{
  namespace dso_fixed
  {
    // Detector execution and numerical safeguards.
    inline constexpr bool kOutputCsv = true;
    inline constexpr bool kUseCorrespondenceScale = true;
    inline constexpr int kMinCorrespondences = 30;
    inline constexpr int kAnalyzeEveryNScans = 1;
    inline constexpr int kPrintEveryNScans = 20;
    inline constexpr double kLegacyRelativeEigenvalueThreshold = 1e-3;
    inline constexpr int kDetectorEnterConsecutiveScans = 10;
    inline constexpr int kDetectorExitConsecutiveScans = 10;
    inline constexpr double kMinCharacteristicRange = 1.0;
    inline constexpr double kMaxCharacteristicRange = 100.0;

    // Exact spline-support diagnostic implementation.
    inline constexpr bool kSupportEnabled = true;
    inline constexpr int kSupportReferenceSamplesPerInterval = 32;
    inline constexpr int kSupportMaxControlPoints = 32;
    inline constexpr int kSupportEnterConsecutiveScans = 10;
    inline constexpr int kSupportExitConsecutiveScans = 10;

    // Validation-only support injection defaults. Injection remains opt-in.
    inline constexpr bool kInjectionOutputCsv = true;
    inline constexpr double kInjectionDefaultSeverity = 0.5;
    inline constexpr double kInjectionDefaultPhaseStart = 0.0;
    inline constexpr double kInjectionDefaultPhaseEnd = 1.0;
    inline constexpr std::uint64_t kInjectionDefaultRandomSeed = 42;

    // CASR-v2 knot lift, temporal gate, and scheduler implementation.
    inline constexpr bool kCasrShadowOutputCsv = true;
    inline constexpr double kSupportBasisRelativeSingularThreshold = 1e-6;
    inline constexpr double kLiftRegularization = 1e-6;
    inline constexpr int kRouteConsecutiveScans = 3;
    inline constexpr int kProjectorConsecutiveScans = 3;
    inline constexpr double kSchedulerProjectorFullConfidence = 0.95;
    inline constexpr double kSchedulerPrincipalFullConfidence = 0.90;
    inline constexpr int kSchedulerPersistenceFullScans = 5;
    inline constexpr double kSchedulerEnterConfidence = 0.25;
    inline constexpr double kSchedulerExitConfidence = 0.10;
    inline constexpr double kSchedulerRiseTimeSeconds = 0.5;
    inline constexpr double kSchedulerFallTimeSeconds = 0.2;
    inline constexpr double kSchedulerMaxDtSeconds = 0.5;

    // Estimator intervention safeguards and the fixed curvature objective.
    inline constexpr bool kInterventionOutputCsv = true;
    inline constexpr double kLegacyBaseInformationWeight = 1.0;
    inline constexpr double kLegacyMaxEffectiveInformationWeight = 100.0;
    inline constexpr bool kCurvatureMatchingEnabled = true;
    inline constexpr double kCurvatureTargetRelativeToMax = 6e-3;
    inline constexpr double kCurvatureMaxAddedRelativeToMax = 2e-2;
    inline constexpr double kDefaultCurvatureGain = 1e-2;
    inline constexpr double kCurvatureMinReference = 1e-9;
    inline constexpr bool kCounterfactualValidation = true;
    inline constexpr double kCounterfactualRatioDenominatorFloor = 1e-6;
    inline constexpr double kMinActivationStrength = 5e-2;
    inline constexpr double kMaxActivationStrength = 1.0;
    inline constexpr int kInterventionMaxControlPoints = 32;
    inline constexpr int kMaxRecoveryRank = 32;
    inline constexpr double kMaxBasisOrthogonalityError = 1e-6;

    // Cause-specific recovery safeguards.  The coupled branch is admitted
    // only when the propagation-reference and spline-continuity corrections
    // have non-negligible magnitude and a non-negative cosine.  This is a
    // sign-consistency safety rule, not a dataset-tuned method parameter.
    inline constexpr double kSourceConsensusMinNorm = 1e-8;
    inline constexpr double kSourceConsensusMinCosine = 0.0;
    inline constexpr double kContinuityProjectorTolerance = 1e-8;
  } // namespace dso_fixed
} // namespace cocolic
