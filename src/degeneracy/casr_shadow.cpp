/*
 * Cause-adaptive subspace recovery (CASR-v2), shadow-only implementation.
 */

#include <degeneracy/casr_shadow.h>

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>

namespace cocolic
{
  const char *CasrSchedulerStateName(CasrSchedulerState state)
  {
    switch (state)
    {
    case CasrSchedulerState::Disabled:
      return "disabled";
    case CasrSchedulerState::InvalidInput:
      return "invalid_input";
    case CasrSchedulerState::UnsafeRoute:
      return "unsafe_route";
    case CasrSchedulerState::RouteMismatch:
      return "route_mismatch";
    case CasrSchedulerState::NoRecoveryBasis:
      return "no_recovery_basis";
    case CasrSchedulerState::NotReady:
      return "not_ready";
    case CasrSchedulerState::BelowEnterConfidence:
      return "below_enter_confidence";
    case CasrSchedulerState::BelowExitConfidence:
      return "below_exit_confidence";
    case CasrSchedulerState::Ramping:
      return "ramping";
    case CasrSchedulerState::Active:
      return "active";
    default:
      return "unknown";
    }
  }

  namespace
  {
    using DynamicBasis = Eigen::MatrixXd;
    using PoseBasis = Eigen::Matrix<double, 6, Eigen::Dynamic>;

    CasrMatrix6 Symmetric6(const CasrMatrix6 &matrix)
    {
      return 0.5 * (matrix + matrix.transpose());
    }

    DynamicBasis OrthonormalBasis(const Eigen::MatrixXd &matrix,
                                  double relative_threshold)
    {
      DynamicBasis empty(matrix.rows(), 0);
      if (matrix.rows() == 0 || matrix.cols() == 0 || !matrix.allFinite())
      {
        return empty;
      }

      Eigen::JacobiSVD<Eigen::MatrixXd> svd(
          matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
      const Eigen::VectorXd singular_values = svd.singularValues();
      if (singular_values.size() == 0 || !singular_values.allFinite())
      {
        return empty;
      }

      const double maximum = singular_values[0];
      if (!std::isfinite(maximum) || maximum <= 1e-12)
      {
        return empty;
      }

      const double threshold = std::max(1e-12, relative_threshold * maximum);
      int rank = 0;
      for (int i = 0; i < singular_values.size(); ++i)
      {
        if (singular_values[i] >= threshold)
        {
          ++rank;
        }
      }
      return svd.matrixU().leftCols(rank);
    }

    PoseBasis BuildEnvironmentPoseBasis(
        const CasrShadowInput &input,
        const CasrShadowConfig &config)
    {
      int rank = 0;
      for (int i = 0; i < 6; ++i)
      {
        if (input.environment_relative_eigenvalues[i] <
            config.environment_relative_threshold)
        {
          ++rank;
        }
      }
      if (input.environment_state && rank == 0)
      {
        rank = 1;
      }
      if (!input.environment_state)
      {
        rank = 0;
      }

      PoseBasis basis(6, rank);
      for (int i = 0; i < rank; ++i)
      {
        basis.col(i) = input.environment_eigenvectors.col(i);
      }
      return basis;
    }

    bool LiftEnvironmentBasis(const CasrShadowInput &input,
                              const CasrShadowConfig &config,
                              const PoseBasis &pose_basis,
                              DynamicBasis &knot_basis,
                              double &lift_residual)
    {
      if (!input.reference_pose_information ||
          !input.reference_pose_cross)
      {
        return false;
      }
      const Eigen::MatrixXd &reference_pose_information =
          *input.reference_pose_information;
      const Eigen::MatrixXd &reference_pose_cross =
          *input.reference_pose_cross;
      const int dimension =
          static_cast<int>(reference_pose_information.rows());
      knot_basis.resize(dimension, 0);
      lift_residual = 0.0;
      if (pose_basis.cols() == 0)
      {
        return true;
      }

      const Eigen::MatrixXd information =
          0.5 * (reference_pose_information +
                 reference_pose_information.transpose());
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(information);
      if (solver.info() != Eigen::Success ||
          !solver.eigenvalues().allFinite())
      {
        return false;
      }

      const double maximum = solver.eigenvalues().maxCoeff();
      if (!std::isfinite(maximum) || maximum <= 1e-12)
      {
        return false;
      }
      const double regularization =
          std::max(1e-12, config.lift_regularization * maximum);
      const Eigen::VectorXd inverse_regularized =
          (solver.eigenvalues().cwiseMax(0.0).array() + regularization)
              .inverse();
      const Eigen::MatrixXd rhs =
          reference_pose_cross * pose_basis;
      const Eigen::MatrixXd raw_lift =
          solver.eigenvectors() * inverse_regularized.asDiagonal() *
          solver.eigenvectors().transpose() * rhs;
      if (!raw_lift.allFinite())
      {
        return false;
      }

      double residual_energy =
          (raw_lift.transpose() * information * raw_lift).trace() -
          2.0 * (raw_lift.transpose() * rhs).trace() +
          static_cast<double>(pose_basis.cols());
      residual_energy = std::max(0.0, residual_energy);
      lift_residual = std::sqrt(
          residual_energy / static_cast<double>(pose_basis.cols()));

      knot_basis = OrthonormalBasis(
          raw_lift, config.support_basis_relative_singular_threshold);
      return knot_basis.cols() == pose_basis.cols();
    }

    bool BuildSupportBasis(const CasrShadowInput &input,
                           const CasrShadowConfig &config,
                           DynamicBasis &basis)
    {
      if (!input.reference_pose_information)
      {
        return false;
      }
      const int dimension =
          static_cast<int>(input.reference_pose_information->rows());
      basis.resize(dimension, 0);
      if (!input.support_state)
      {
        return true;
      }
      if (!input.support_knot_weak_basis ||
          input.support_knot_weak_basis->rows() != dimension ||
          input.support_knot_weak_basis->cols() == 0 ||
          !input.support_knot_weak_basis->allFinite())
      {
        return false;
      }

      basis = OrthonormalBasis(
          *input.support_knot_weak_basis,
          config.support_basis_relative_singular_threshold);
      return basis.cols() > 0;
    }

    CasrMatrix6 Projector6(const Eigen::MatrixXd &basis)
    {
      if (basis.rows() != 6 || basis.cols() == 0)
      {
        return CasrMatrix6::Zero();
      }
      return Symmetric6(basis * basis.transpose());
    }

    CasrMatrix6 RepresentativePoseProjector(
        const Eigen::MatrixXd &mapping, const DynamicBasis &knot_basis,
        int *rank)
    {
      if (rank)
      {
        *rank = 0;
      }
      if (mapping.rows() != 6 || mapping.cols() != knot_basis.rows() ||
          knot_basis.cols() == 0 || !mapping.allFinite())
      {
        return CasrMatrix6::Zero();
      }

      const DynamicBasis pose_basis =
          OrthonormalBasis(mapping * knot_basis, 1e-8);
      if (rank)
      {
        *rank = static_cast<int>(pose_basis.cols());
      }
      return Projector6(pose_basis);
    }

    CasrVector6 DescendingPoseEnergies(
        const Eigen::MatrixXd &mapping, const DynamicBasis &knot_basis)
    {
      CasrVector6 result = CasrVector6::Zero();
      if (mapping.rows() != 6 || mapping.cols() != knot_basis.rows() ||
          knot_basis.cols() == 0 || !mapping.allFinite())
      {
        return result;
      }

      const Eigen::MatrixXd mapped = mapping * knot_basis;
      CasrMatrix6 energy = Symmetric6(mapped * mapped.transpose());
      const double trace = energy.trace();
      if (!energy.allFinite() || !std::isfinite(trace) || trace <= 1e-12)
      {
        return result;
      }
      energy /= trace;
      Eigen::SelfAdjointEigenSolver<CasrMatrix6> solver(energy);
      if (solver.info() != Eigen::Success)
      {
        return result;
      }
      for (int i = 0; i < 6; ++i)
      {
        result[i] = std::max(0.0, solver.eigenvalues()[5 - i]);
      }
      return result;
    }

    double BasisOrthogonalityError(const DynamicBasis &basis)
    {
      if (basis.cols() == 0)
      {
        return 0.0;
      }
      const Eigen::MatrixXd error =
          basis.transpose() * basis -
          Eigen::MatrixXd::Identity(basis.cols(), basis.cols());
      return error.cwiseAbs().maxCoeff();
    }

    struct KnotSubspaceComparison
    {
      double similarity = 0.0;
      double projector_affinity = 0.0;
      int overlap_control_point_num = 0;
      int previous_overlap_rank = 0;
      int current_overlap_rank = 0;
      int forced_overlap_rank = 0;
    };

    KnotSubspaceComparison CompareKnotSubspaces(
        const DynamicBasis &first, int first_start_index,
        const DynamicBasis &second, int second_start_index)
    {
      KnotSubspaceComparison comparison;
      if (first.rows() % 6 != 0 || second.rows() % 6 != 0 ||
          first.cols() == 0 || second.cols() == 0 ||
          first_start_index < 0 || second_start_index < 0)
      {
        return comparison;
      }

      const int first_control_points = static_cast<int>(first.rows()) / 6;
      const int second_control_points = static_cast<int>(second.rows()) / 6;
      const int overlap_start = std::max(first_start_index,
                                         second_start_index);
      const int overlap_end = std::min(
          first_start_index + first_control_points,
          second_start_index + second_control_points);
      if (overlap_start >= overlap_end)
      {
        return comparison;
      }

      const int overlap_control_points = overlap_end - overlap_start;
      comparison.overlap_control_point_num = overlap_control_points;
      DynamicBasis first_overlap(6 * overlap_control_points, first.cols());
      DynamicBasis second_overlap(6 * overlap_control_points,
                                  second.cols());
      for (int global_index = overlap_start;
           global_index < overlap_end; ++global_index)
      {
        const int overlap_local = global_index - overlap_start;
        const int first_local = global_index - first_start_index;
        const int second_local = global_index - second_start_index;
        first_overlap.block(6 * overlap_local, 0, 6, first.cols()) =
            first.block(6 * first_local, 0, 6, first.cols());
        second_overlap.block(6 * overlap_local, 0, 6, second.cols()) =
            second.block(6 * second_local, 0, 6, second.cols());
      }

      // The active spline window normally advances by one control point per
      // scan. Re-orthonormalize both restrictions before comparing them so
      // that energy carried by non-overlapping boundary knots does not create
      // an artificial similarity loss. The comparison below also removes the
      // intersection forced solely by both subspaces having high rank.
      const DynamicBasis first_local_basis =
          OrthonormalBasis(first_overlap, 1e-8);
      const DynamicBasis second_local_basis =
          OrthonormalBasis(second_overlap, 1e-8);
      if (first_local_basis.cols() == 0 ||
          second_local_basis.cols() == 0)
      {
        return comparison;
      }

      comparison.previous_overlap_rank = first_local_basis.cols();
      comparison.current_overlap_rank = second_local_basis.cols();
      const int ambient_dimension = 6 * overlap_control_points;
      comparison.forced_overlap_rank = std::max(
          0, comparison.previous_overlap_rank +
                 comparison.current_overlap_rank - ambient_dimension);
      const Eigen::MatrixXd cross =
          first_local_basis.transpose() * second_local_basis;
      comparison.projector_affinity = cross.squaredNorm();
      const int maximum_rank = std::max(comparison.previous_overlap_rank,
                                        comparison.current_overlap_rank);
      const double normalizer = static_cast<double>(
          maximum_rank - comparison.forced_overlap_rank);

      // Two subspaces of ranks r1 and r2 in an n-dimensional ambient space
      // must intersect in at least max(0, r1 + r2 - n) dimensions. Remove
      // that unavoidable Grassmann intersection before normalizing so that a
      // high rank alone cannot satisfy the temporal consistency gate.
      if (normalizer <= 0.0)
      {
        comparison.similarity = 1.0;
        return comparison;
      }
      const double debiased_affinity =
          comparison.projector_affinity -
          static_cast<double>(comparison.forced_overlap_rank);
      comparison.similarity = std::max(
          0.0, std::min(1.0, debiased_affinity / normalizer));
      return comparison;
    }

    double ClampUnit(double value)
    {
      return std::max(0.0, std::min(1.0, value));
    }

    double DescendingConfidence(double value, double full_threshold,
                                double zero_threshold)
    {
      if (!std::isfinite(value))
      {
        return 0.0;
      }
      if (zero_threshold <= full_threshold)
      {
        return value <= full_threshold ? 1.0 : 0.0;
      }
      if (value <= full_threshold)
      {
        return 1.0;
      }
      if (value >= zero_threshold)
      {
        return 0.0;
      }
      return ClampUnit((zero_threshold - value) /
                       (zero_threshold - full_threshold));
    }

    double AscendingConfidence(double value, double zero_threshold,
                               double full_threshold)
    {
      if (!std::isfinite(value))
      {
        return 0.0;
      }
      if (full_threshold <= zero_threshold)
      {
        return value >= full_threshold ? 1.0 : 0.0;
      }
      if (value <= zero_threshold)
      {
        return 0.0;
      }
      if (value >= full_threshold)
      {
        return 1.0;
      }
      return ClampUnit((value - zero_threshold) /
                       (full_threshold - zero_threshold));
    }

    double PersistenceConfidence(const CasrShadowConfig &config,
                                 int consistency_count)
    {
      if (consistency_count < config.projector_consecutive_scans)
      {
        return 0.0;
      }
      const int span = config.scheduler_persistence_full_scans -
                       config.projector_consecutive_scans + 1;
      return ClampUnit(static_cast<double>(
                           consistency_count -
                           config.projector_consecutive_scans + 1) /
                       static_cast<double>(span));
    }

    bool IsSchedulableRoute(CasrRoute route)
    {
      return route == CasrRoute::EnvironmentCandidate ||
             route == CasrRoute::SupportCandidate ||
             route == CasrRoute::CoupledCommonCandidate;
    }

    void ResetSchedulerState(CasrTemporalState *state,
                             double scan_timestamp_s)
    {
      if (!state)
      {
        return;
      }
      state->scheduler_initialized = std::isfinite(scan_timestamp_s);
      state->scheduler_active = false;
      state->scheduler_strength = 0.0;
      state->scheduler_last_timestamp_s =
          std::isfinite(scan_timestamp_s) ? scan_timestamp_s : 0.0;
    }

    void BlockScheduler(CasrSchedulerState scheduler_state,
                        const CasrShadowInput &input,
                        CasrTemporalState *state,
                        CasrShadowResult &result)
    {
      ResetSchedulerState(state, input.scan_timestamp_s);
      result.scheduler_state = scheduler_state;
      result.scheduler_eligible = false;
      result.scheduler_active = false;
      result.scheduler_target_strength = 0.0;
      result.scheduler_activation_strength = 0.0;
      result.scheduler_dt_s = 0.0;
    }

    void UpdateShadowScheduler(const CasrShadowConfig &config,
                               const CasrShadowInput &input,
                               CasrTemporalState *state,
                               CasrShadowResult &result)
    {
      if (!config.scheduler_enabled)
      {
        BlockScheduler(CasrSchedulerState::Disabled, input, state, result);
        return;
      }

      const double environment_lambda =
          input.environment_relative_eigenvalues[0];
      result.scheduler_environment_confidence =
          input.environment_state
              ? DescendingConfidence(
                    environment_lambda,
                    config.scheduler_environment_full_confidence_threshold,
                    config.scheduler_environment_zero_confidence_threshold)
              : 0.0;
      result.scheduler_support_confidence =
          input.support_state
              ? DescendingConfidence(
                    input.support_quality_min,
                    config.scheduler_support_full_confidence_threshold,
                    config.scheduler_support_zero_confidence_threshold)
              : 0.0;
      result.scheduler_temporal_confidence = AscendingConfidence(
          result.temporal_projector_similarity,
          config.projector_similarity_threshold,
          config.scheduler_projector_full_confidence);
      result.scheduler_persistence_confidence = PersistenceConfidence(
          config, result.projector_consistency_count);
      result.scheduler_principal_confidence =
          result.route == CasrRoute::CoupledCommonCandidate
              ? AscendingConfidence(
                    result.principal_cosine_max,
                    config.principal_cosine_threshold,
                    config.scheduler_principal_full_confidence)
              : (IsSchedulableRoute(result.route) ? 1.0 : 0.0);

      switch (result.route)
      {
      case CasrRoute::EnvironmentCandidate:
        result.scheduler_cause_confidence =
            result.scheduler_environment_confidence;
        break;
      case CasrRoute::SupportCandidate:
        result.scheduler_cause_confidence =
            result.scheduler_support_confidence;
        break;
      case CasrRoute::CoupledCommonCandidate:
        result.scheduler_cause_confidence = std::min(
            result.scheduler_principal_confidence,
            std::min(result.scheduler_environment_confidence,
                     result.scheduler_support_confidence));
        break;
      default:
        result.scheduler_cause_confidence = 0.0;
        break;
      }

      result.scheduler_raw_confidence = std::min(
          result.scheduler_cause_confidence,
          std::min(result.scheduler_temporal_confidence,
                   result.scheduler_persistence_confidence));

      if (!result.valid)
      {
        BlockScheduler(CasrSchedulerState::InvalidInput, input, state,
                       result);
        return;
      }
      if (!IsSchedulableRoute(result.route))
      {
        BlockScheduler(CasrSchedulerState::UnsafeRoute, input, state,
                       result);
        return;
      }
      if (result.route != result.stable_route)
      {
        BlockScheduler(CasrSchedulerState::RouteMismatch, input, state,
                       result);
        return;
      }
      if (result.recovery_knot_basis.cols() == 0)
      {
        BlockScheduler(CasrSchedulerState::NoRecoveryBasis, input, state,
                       result);
        return;
      }
      if (!result.recovery_ready)
      {
        BlockScheduler(CasrSchedulerState::NotReady, input, state, result);
        return;
      }

      result.scheduler_eligible = true;
      if (!state)
      {
        result.scheduler_active =
            result.scheduler_raw_confidence >=
            config.scheduler_enter_confidence;
        result.scheduler_state = result.scheduler_active
                                     ? CasrSchedulerState::Active
                                     : CasrSchedulerState::BelowEnterConfidence;
        result.scheduler_target_strength =
            result.scheduler_active ? result.scheduler_raw_confidence : 0.0;
        result.scheduler_activation_strength =
            result.scheduler_target_strength;
        return;
      }

      double dt_s = 0.0;
      if (state->scheduler_initialized &&
          std::isfinite(input.scan_timestamp_s) &&
          input.scan_timestamp_s > state->scheduler_last_timestamp_s)
      {
        dt_s = std::min(config.scheduler_max_dt_s,
                        input.scan_timestamp_s -
                            state->scheduler_last_timestamp_s);
      }
      state->scheduler_initialized = std::isfinite(input.scan_timestamp_s);
      if (state->scheduler_initialized)
      {
        state->scheduler_last_timestamp_s = input.scan_timestamp_s;
      }
      result.scheduler_dt_s = dt_s;

      bool exited = false;
      if (!state->scheduler_active)
      {
        if (result.scheduler_raw_confidence >=
            config.scheduler_enter_confidence)
        {
          state->scheduler_active = true;
        }
      }
      else if (result.scheduler_raw_confidence <=
               config.scheduler_exit_confidence)
      {
        state->scheduler_active = false;
        exited = true;
      }

      result.scheduler_target_strength =
          state->scheduler_active ? result.scheduler_raw_confidence : 0.0;
      if (result.scheduler_target_strength > state->scheduler_strength)
      {
        const double step = dt_s / config.scheduler_rise_time_s;
        state->scheduler_strength = std::min(
            result.scheduler_target_strength,
            state->scheduler_strength + step);
      }
      else if (result.scheduler_target_strength < state->scheduler_strength)
      {
        const double step = dt_s / config.scheduler_fall_time_s;
        state->scheduler_strength = std::max(
            result.scheduler_target_strength,
            state->scheduler_strength - step);
      }
      state->scheduler_strength = ClampUnit(state->scheduler_strength);
      result.scheduler_activation_strength = state->scheduler_strength;
      result.scheduler_active = state->scheduler_strength > 1e-12;

      if (exited)
      {
        result.scheduler_state = CasrSchedulerState::BelowExitConfidence;
      }
      else if (!state->scheduler_active)
      {
        result.scheduler_state =
            CasrSchedulerState::BelowEnterConfidence;
      }
      else if (std::abs(result.scheduler_activation_strength -
                        result.scheduler_target_strength) > 1e-12)
      {
        result.scheduler_state = CasrSchedulerState::Ramping;
      }
      else
      {
        result.scheduler_state = CasrSchedulerState::Active;
      }
    }

    void UpdateTemporalGate(const CasrShadowConfig &config,
                            CasrTemporalState *state,
                            CasrShadowResult &result)
    {
      if (!state)
      {
        result.stable_route = result.route;
        result.temporal_projector_similarity =
            result.recovery_rank > 0 ? 1.0 : 0.0;
        result.projector_consistency_count =
            result.recovery_rank > 0 ? 1 : 0;
        result.recovery_ready = result.recovery_rank > 0;
        return;
      }

      bool route_switched = false;
      if (!state->initialized)
      {
        state->initialized = true;
        state->stable_route = result.route;
        state->pending_route = CasrRoute::Invalid;
        state->pending_route_count = 0;
        route_switched = true;
      }
      else if (result.route == state->stable_route)
      {
        state->pending_route = CasrRoute::Invalid;
        state->pending_route_count = 0;
      }
      else
      {
        if (result.route == state->pending_route)
        {
          ++state->pending_route_count;
        }
        else
        {
          state->pending_route = result.route;
          state->pending_route_count = 1;
        }
        result.route_candidate_count = state->pending_route_count;
        if (state->pending_route_count >= config.route_consecutive_scans)
        {
          state->stable_route = result.route;
          state->pending_route = CasrRoute::Invalid;
          state->pending_route_count = 0;
          route_switched = true;
        }
      }

      result.stable_route = state->stable_route;
      const bool route_matches = result.route == state->stable_route;
      const bool has_recovery_basis = result.recovery_knot_basis.cols() > 0;
      if (route_switched)
      {
        state->projector_consistency_count = 0;
        state->previous_recovery_basis.resize(0, 0);
        state->previous_control_point_start_index = -1;
      }

      if (route_matches && has_recovery_basis)
      {
        if (state->previous_recovery_basis.cols() == 0)
        {
          result.temporal_projector_similarity = 1.0;
          state->projector_consistency_count = 1;
        }
        else
        {
          const KnotSubspaceComparison comparison = CompareKnotSubspaces(
              state->previous_recovery_basis,
              state->previous_control_point_start_index,
              result.recovery_knot_basis,
              result.support_control_point_start_index);
          result.temporal_projector_similarity = comparison.similarity;
          result.temporal_overlap_control_point_num =
              comparison.overlap_control_point_num;
          result.temporal_previous_overlap_rank =
              comparison.previous_overlap_rank;
          result.temporal_current_overlap_rank =
              comparison.current_overlap_rank;
          result.temporal_forced_overlap_rank =
              comparison.forced_overlap_rank;
          result.temporal_projector_affinity =
              comparison.projector_affinity;
          if (result.temporal_projector_similarity >=
              config.projector_similarity_threshold)
          {
            ++state->projector_consistency_count;
          }
          else
          {
            state->projector_consistency_count = 1;
          }
        }
        state->previous_recovery_basis = result.recovery_knot_basis;
        state->previous_control_point_start_index =
            result.support_control_point_start_index;
      }
      else
      {
        result.temporal_projector_similarity = 0.0;
        state->projector_consistency_count = 0;
        if (route_matches)
        {
          state->previous_recovery_basis.resize(0, 0);
          state->previous_control_point_start_index = -1;
        }
      }

      result.projector_consistency_count =
          state->projector_consistency_count;
      result.recovery_ready =
          route_matches && has_recovery_basis &&
          state->projector_consistency_count >=
              config.projector_consecutive_scans;
    }
  } // namespace

  void CasrShadowEvaluator::Configure(const CasrShadowConfig &config)
  {
    config_ = config;
    if (!std::isfinite(config_.environment_relative_threshold))
    {
      config_.environment_relative_threshold = 6e-3;
    }
    if (!std::isfinite(
            config_.support_basis_relative_singular_threshold))
    {
      config_.support_basis_relative_singular_threshold = 1e-6;
    }
    if (!std::isfinite(config_.lift_regularization))
    {
      config_.lift_regularization = 1e-6;
    }
    if (!std::isfinite(config_.principal_cosine_threshold))
    {
      config_.principal_cosine_threshold = 7e-1;
    }
    if (!std::isfinite(config_.projector_similarity_threshold))
    {
      config_.projector_similarity_threshold = 8e-1;
    }
    if (!std::isfinite(
            config_.scheduler_environment_full_confidence_threshold))
    {
      config_.scheduler_environment_full_confidence_threshold = 3e-3;
    }
    if (!std::isfinite(
            config_.scheduler_environment_zero_confidence_threshold))
    {
      config_.scheduler_environment_zero_confidence_threshold = 6e-3;
    }
    if (!std::isfinite(
            config_.scheduler_support_full_confidence_threshold))
    {
      config_.scheduler_support_full_confidence_threshold = 2e-2;
    }
    if (!std::isfinite(
            config_.scheduler_support_zero_confidence_threshold))
    {
      config_.scheduler_support_zero_confidence_threshold = 5e-2;
    }
    if (!std::isfinite(config_.scheduler_projector_full_confidence))
    {
      config_.scheduler_projector_full_confidence = 9.5e-1;
    }
    if (!std::isfinite(config_.scheduler_principal_full_confidence))
    {
      config_.scheduler_principal_full_confidence = 9e-1;
    }
    if (!std::isfinite(config_.scheduler_enter_confidence))
    {
      config_.scheduler_enter_confidence = 2.5e-1;
    }
    if (!std::isfinite(config_.scheduler_exit_confidence))
    {
      config_.scheduler_exit_confidence = 1e-1;
    }
    if (!std::isfinite(config_.scheduler_rise_time_s))
    {
      config_.scheduler_rise_time_s = 5e-1;
    }
    if (!std::isfinite(config_.scheduler_fall_time_s))
    {
      config_.scheduler_fall_time_s = 2e-1;
    }
    if (!std::isfinite(config_.scheduler_max_dt_s))
    {
      config_.scheduler_max_dt_s = 5e-1;
    }

    config_.environment_relative_threshold =
        std::max(1e-12, config_.environment_relative_threshold);
    config_.support_basis_relative_singular_threshold =
        std::max(1e-12,
                 std::min(1.0,
                          config_.support_basis_relative_singular_threshold));
    config_.lift_regularization =
        std::max(1e-12, std::min(1.0, config_.lift_regularization));
    config_.principal_cosine_threshold =
        std::max(1e-12,
                 std::min(1.0, config_.principal_cosine_threshold));
    config_.route_consecutive_scans =
        std::max(1, config_.route_consecutive_scans);
    config_.projector_consecutive_scans =
        std::max(1, config_.projector_consecutive_scans);
    config_.projector_similarity_threshold =
        std::max(0.0,
                 std::min(1.0, config_.projector_similarity_threshold));
    config_.scheduler_environment_full_confidence_threshold =
        std::max(0.0,
                 config_.scheduler_environment_full_confidence_threshold);
    config_.scheduler_environment_zero_confidence_threshold = std::max(
        config_.scheduler_environment_full_confidence_threshold + 1e-12,
        config_.scheduler_environment_zero_confidence_threshold);
    config_.scheduler_support_full_confidence_threshold =
        std::max(0.0,
                 config_.scheduler_support_full_confidence_threshold);
    config_.scheduler_support_zero_confidence_threshold = std::max(
        config_.scheduler_support_full_confidence_threshold + 1e-12,
        config_.scheduler_support_zero_confidence_threshold);
    config_.scheduler_projector_full_confidence = std::max(
        config_.projector_similarity_threshold + 1e-12,
        std::min(1.0, config_.scheduler_projector_full_confidence));
    config_.scheduler_principal_full_confidence = std::max(
        config_.principal_cosine_threshold + 1e-12,
        std::min(1.0, config_.scheduler_principal_full_confidence));
    config_.scheduler_persistence_full_scans = std::max(
        config_.projector_consecutive_scans,
        config_.scheduler_persistence_full_scans);
    config_.scheduler_enter_confidence =
        ClampUnit(config_.scheduler_enter_confidence);
    config_.scheduler_exit_confidence = std::min(
        config_.scheduler_enter_confidence,
        ClampUnit(config_.scheduler_exit_confidence));
    config_.scheduler_rise_time_s =
        std::max(1e-6, config_.scheduler_rise_time_s);
    config_.scheduler_fall_time_s =
        std::max(1e-6, config_.scheduler_fall_time_s);
    config_.scheduler_max_dt_s =
        std::max(1e-6, config_.scheduler_max_dt_s);
  }

  CasrShadowResult CasrShadowEvaluator::Evaluate(
      const CasrShadowInput &input,
      CasrTemporalState *temporal_state) const
  {
    CasrShadowResult result;
    result.data_source = input.data_source;
    const auto invalid_result = [&]()
    {
      if (temporal_state)
      {
        temporal_state->projector_consistency_count = 0;
        temporal_state->previous_control_point_start_index = -1;
        temporal_state->previous_recovery_basis.resize(0, 0);
      }
      UpdateShadowScheduler(config_, input, temporal_state, result);
      return result;
    };
    if (!config_.enabled || !input.environment_valid ||
        !input.support_valid ||
        !input.environment_relative_eigenvalues.allFinite() ||
        !input.environment_eigenvectors.allFinite())
    {
      return invalid_result();
    }

    if (!input.support_knot_weak_basis ||
        !input.reference_pose_information ||
        !input.reference_pose_cross ||
        !input.representative_pose_mapping)
    {
      return invalid_result();
    }
    const Eigen::MatrixXd &support_knot_weak_basis =
        *input.support_knot_weak_basis;
    const Eigen::MatrixXd &reference_pose_information =
        *input.reference_pose_information;
    const Eigen::MatrixXd &reference_pose_cross =
        *input.reference_pose_cross;
    const Eigen::MatrixXd &representative_pose_mapping =
        *input.representative_pose_mapping;
    const int dimension =
        static_cast<int>(reference_pose_information.rows());
    if (dimension <= 0 || dimension % 6 != 0 ||
        input.support_control_point_start_index < 0 ||
        support_knot_weak_basis.rows() != dimension ||
        reference_pose_information.cols() != dimension ||
        reference_pose_cross.rows() != dimension ||
        reference_pose_cross.cols() != 6 ||
        representative_pose_mapping.rows() != 6 ||
        representative_pose_mapping.cols() != dimension ||
        !reference_pose_information.allFinite() ||
        !reference_pose_cross.allFinite() ||
        !representative_pose_mapping.allFinite())
    {
      return invalid_result();
    }

    result.cause = (input.environment_state ? 1 : 0) +
                   (input.support_state ? 2 : 0);
    result.support_control_point_start_index =
        input.support_control_point_start_index;
    result.support_knot_dimension = dimension;

    const PoseBasis environment_pose_basis =
        BuildEnvironmentPoseBasis(input, config_);
    if (!LiftEnvironmentBasis(input, config_, environment_pose_basis,
                              result.environment_knot_basis,
                              result.environment_lift_residual) ||
        !BuildSupportBasis(input, config_, result.support_knot_basis))
    {
      return invalid_result();
    }

    result.environment_rank =
        static_cast<int>(result.environment_knot_basis.cols());
    result.support_rank =
        static_cast<int>(result.support_knot_basis.cols());
    result.environment_projector = Projector6(environment_pose_basis);
    result.support_projector = RepresentativePoseProjector(
        representative_pose_mapping, result.support_knot_basis,
        nullptr);
    result.support_pose_eigenvalues = DescendingPoseEnergies(
        representative_pose_mapping, result.support_knot_basis);

    if (result.environment_rank > 0 && result.support_rank > 0)
    {
      const Eigen::MatrixXd cross =
          result.environment_knot_basis.transpose() *
          result.support_knot_basis;
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(
          cross, Eigen::ComputeThinU | Eigen::ComputeThinV);
      const Eigen::VectorXd singular_values = svd.singularValues();
      if (!singular_values.allFinite() || singular_values.size() == 0)
      {
        return invalid_result();
      }

      const double overlap_energy = singular_values.squaredNorm();
      const int normalizing_rank =
          std::min(result.environment_rank, result.support_rank);
      result.overlap_score = overlap_energy /
                             static_cast<double>(normalizing_rank);
      result.principal_cosine_max = singular_values.maxCoeff();
      result.principal_cosine_min = singular_values.minCoeff();
      result.environment_exclusive_ratio = std::max(
          0.0, 1.0 - overlap_energy /
                         static_cast<double>(result.environment_rank));
      result.support_exclusive_ratio = std::max(
          0.0, 1.0 - overlap_energy /
                         static_cast<double>(result.support_rank));

      for (int i = 0; i < singular_values.size(); ++i)
      {
        if (singular_values[i] >= config_.principal_cosine_threshold)
        {
          ++result.common_rank;
        }
      }
      if (result.common_rank > 0)
      {
        result.common_knot_basis =
            result.environment_knot_basis *
            svd.matrixU().leftCols(result.common_rank);
      }
      else
      {
        result.common_knot_basis.resize(dimension, 0);
      }
      result.common_projector = RepresentativePoseProjector(
          representative_pose_mapping, result.common_knot_basis,
          nullptr);
    }
    else
    {
      result.common_knot_basis.resize(dimension, 0);
    }

    result.route = SelectCasrRoute(
        true, result.cause, result.environment_rank, result.support_rank,
        result.common_rank);
    switch (result.route)
    {
    case CasrRoute::Inactive:
      result.recovery_knot_basis.resize(dimension, 0);
      break;
    case CasrRoute::EnvironmentCandidate:
      result.recovery_knot_basis = result.environment_knot_basis;
      break;
    case CasrRoute::SupportCandidate:
      result.recovery_knot_basis = result.support_knot_basis;
      break;
    case CasrRoute::CoupledCommonCandidate:
      result.recovery_knot_basis = result.common_knot_basis;
      break;
    case CasrRoute::CoupledConflict:
      // A coupled case without a reliable common direction is deliberately
      // deferred. Producing no knot basis is safer than inventing a direction.
      result.recovery_knot_basis.resize(dimension, 0);
      break;
    case CasrRoute::Invalid:
    default:
      return invalid_result();
    }

    result.recovery_rank =
        static_cast<int>(result.recovery_knot_basis.cols());
    result.recovery_basis_orthogonality_error =
        BasisOrthogonalityError(result.recovery_knot_basis);
    result.recovery_projector = RepresentativePoseProjector(
        representative_pose_mapping, result.recovery_knot_basis,
        &result.representative_pose_rank);
    result.valid = true;
    UpdateTemporalGate(config_, temporal_state, result);
    UpdateShadowScheduler(config_, input, temporal_state, result);
    return result;
  }

} // namespace cocolic
