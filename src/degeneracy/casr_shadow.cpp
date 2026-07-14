/*
 * Cause-adaptive subspace recovery (CASR), shadow-only implementation.
 */

#include <degeneracy/casr_shadow.h>

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>

namespace cocolic
{
  namespace
  {
    using DynamicBasis = Eigen::Matrix<double, 6, Eigen::Dynamic>;

    CasrMatrix6 Symmetric(const CasrMatrix6 &matrix)
    {
      return 0.5 * (matrix + matrix.transpose());
    }

    DynamicBasis BuildEnvironmentBasis(
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

      DynamicBasis basis(6, rank);
      for (int i = 0; i < rank; ++i)
      {
        basis.col(i) = input.environment_eigenvectors.col(i);
      }
      return basis;
    }

    bool BuildSupportBasis(const CasrShadowInput &input,
                           const CasrShadowConfig &config,
                           DynamicBasis &basis,
                           CasrVector6 &descending_eigenvalues)
    {
      basis.resize(6, 0);
      descending_eigenvalues.setZero();
      if (!input.support_state)
      {
        return true;
      }

      const CasrMatrix6 weakness =
          Symmetric(input.support_pose_weakness);
      if (!weakness.allFinite())
      {
        return false;
      }

      Eigen::SelfAdjointEigenSolver<CasrMatrix6> solver(weakness);
      if (solver.info() != Eigen::Success)
      {
        return false;
      }

      const CasrVector6 eigenvalues =
          solver.eigenvalues().cwiseMax(0.0);
      const double maximum = eigenvalues.maxCoeff();
      if (!std::isfinite(maximum) || maximum <= 1e-12)
      {
        return false;
      }

      int rank = 0;
      for (int i = 0; i < 6; ++i)
      {
        const double value = eigenvalues[5 - i];
        descending_eigenvalues[i] = value;
        if (value / maximum >= config.support_pose_relative_threshold)
        {
          ++rank;
        }
      }
      rank = std::max(1, rank);

      basis.resize(6, rank);
      for (int i = 0; i < rank; ++i)
      {
        basis.col(i) = solver.eigenvectors().col(5 - i);
      }
      return true;
    }

    CasrMatrix6 Projector(const DynamicBasis &basis)
    {
      if (basis.cols() == 0)
      {
        return CasrMatrix6::Zero();
      }
      return Symmetric(basis * basis.transpose());
    }
  } // namespace

  void CasrShadowEvaluator::Configure(const CasrShadowConfig &config)
  {
    config_ = config;
    if (!std::isfinite(config_.environment_relative_threshold))
    {
      config_.environment_relative_threshold = 6e-3;
    }
    if (!std::isfinite(config_.support_pose_relative_threshold))
    {
      config_.support_pose_relative_threshold = 1e-1;
    }
    if (!std::isfinite(config_.principal_cosine_threshold))
    {
      config_.principal_cosine_threshold = 7e-1;
    }
    config_.environment_relative_threshold =
        std::max(1e-12, config_.environment_relative_threshold);
    config_.support_pose_relative_threshold =
        std::max(1e-12,
                 std::min(1.0, config_.support_pose_relative_threshold));
    config_.principal_cosine_threshold =
        std::max(1e-12,
                 std::min(1.0, config_.principal_cosine_threshold));
  }

  CasrShadowResult CasrShadowEvaluator::Evaluate(
      const CasrShadowInput &input) const
  {
    CasrShadowResult result;
    if (!config_.enabled || !input.environment_valid ||
        !input.support_valid ||
        !input.environment_relative_eigenvalues.allFinite() ||
        !input.environment_eigenvectors.allFinite())
    {
      return result;
    }

    result.cause = (input.environment_state ? 1 : 0) +
                   (input.support_state ? 2 : 0);

    const DynamicBasis environment_basis =
        BuildEnvironmentBasis(input, config_);
    DynamicBasis support_basis;
    if (!BuildSupportBasis(input, config_, support_basis,
                           result.support_pose_eigenvalues))
    {
      return result;
    }

    result.environment_rank =
        static_cast<int>(environment_basis.cols());
    result.support_rank = static_cast<int>(support_basis.cols());
    result.environment_projector = Projector(environment_basis);
    result.support_projector = Projector(support_basis);

    if (result.environment_rank > 0 && result.support_rank > 0)
    {
      const Eigen::MatrixXd cross =
          environment_basis.transpose() * support_basis;
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(cross);
      const Eigen::VectorXd singular_values = svd.singularValues();
      if (!singular_values.allFinite() || singular_values.size() == 0)
      {
        return CasrShadowResult();
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

      CasrMatrix6 common_kernel =
          result.environment_projector * result.support_projector *
          result.environment_projector;
      common_kernel = Symmetric(common_kernel);
      Eigen::SelfAdjointEigenSolver<CasrMatrix6> common_solver(
          common_kernel);
      if (common_solver.info() != Eigen::Success)
      {
        return CasrShadowResult();
      }

      const double common_threshold =
          config_.principal_cosine_threshold *
          config_.principal_cosine_threshold;
      for (int i = 5; i >= 0; --i)
      {
        if (common_solver.eigenvalues()[i] >= common_threshold)
        {
          const CasrVector6 direction =
              common_solver.eigenvectors().col(i);
          result.common_projector.noalias() +=
              direction * direction.transpose();
          ++result.common_rank;
        }
      }
      result.common_projector = Symmetric(result.common_projector);
    }

    result.route = SelectCasrRoute(
        true, result.cause, result.environment_rank, result.support_rank,
        result.common_rank);
    switch (result.route)
    {
    case CasrRoute::Inactive:
      result.recovery_rank = 0;
      break;
    case CasrRoute::EnvironmentCandidate:
      result.recovery_projector = result.environment_projector;
      result.recovery_rank = result.environment_rank;
      break;
    case CasrRoute::SupportCandidate:
      result.recovery_projector = result.support_projector;
      result.recovery_rank = result.support_rank;
      break;
    case CasrRoute::CoupledCommonCandidate:
      result.recovery_projector = result.common_projector;
      result.recovery_rank = result.common_rank;
      break;
    case CasrRoute::CoupledConflict:
      // A coupled case without a reliable common direction is deliberately
      // deferred. Producing no projector is safer than inventing a direction.
      result.recovery_rank = 0;
      break;
    case CasrRoute::Invalid:
    default:
      return result;
    }

    result.valid = true;
    return result;
  }

} // namespace cocolic
