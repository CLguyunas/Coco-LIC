/*
 * Cause-specific CASR knot-space factors.
 *
 * Environment degradation uses a propagation-reference anchor. Spline
 * support degradation uses a time-aware affine-nullspace projector, so only
 * non-affine changes of the active control-point increments are penalized.
 * Coupled degradation stacks both residuals after an independent source
 * consistency gate in TrajectoryManager.
 */

#pragma once

#include <ceres/ceres.h>
#include <sophus_lib/so3.hpp>
#include <utils/sophus_utils.hpp>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/StdVector>

#include <algorithm>
#include <cmath>
#include <vector>

namespace cocolic
{
  namespace analytic_derivative
  {

    inline Eigen::MatrixXd BuildCasrAffineNullspaceProjector(
        const std::vector<double> &knot_times_seconds,
        const Eigen::aligned_vector<Sophus::SO3<double>>
            &reference_rotations)
    {
      const int knot_num = static_cast<int>(knot_times_seconds.size());
      if (knot_num < 3)
      {
        return Eigen::MatrixXd();
      }
      for (int i = 0; i < knot_num; ++i)
      {
        if (!std::isfinite(knot_times_seconds[static_cast<size_t>(i)]) ||
            (i > 0 &&
             knot_times_seconds[static_cast<size_t>(i)] <=
                 knot_times_seconds[static_cast<size_t>(i - 1)]))
        {
          return Eigen::MatrixXd();
        }
      }

      Eigen::VectorXd normalized_time(knot_num);
      double mean_time = 0.0;
      for (double time : knot_times_seconds)
      {
        mean_time += time;
      }
      mean_time /= static_cast<double>(knot_num);
      double time_scale = 0.0;
      for (int i = 0; i < knot_num; ++i)
      {
        normalized_time[i] =
            knot_times_seconds[static_cast<size_t>(i)] - mean_time;
        time_scale = std::max(time_scale, std::abs(normalized_time[i]));
      }
      if (!std::isfinite(time_scale) || time_scale <= 0.0)
      {
        return Eigen::MatrixXd();
      }
      normalized_time /= time_scale;

      Eigen::MatrixXd affine_basis(knot_num, 2);
      affine_basis.col(0).setOnes();
      affine_basis.col(1) = normalized_time;
      const Eigen::Matrix2d normal =
          affine_basis.transpose() * affine_basis;
      const Eigen::LDLT<Eigen::Matrix2d> ldlt(normal);
      if (ldlt.info() != Eigen::Success)
      {
        return Eigen::MatrixXd();
      }
      Eigen::MatrixXd knot_projector =
          Eigen::MatrixXd::Identity(knot_num, knot_num) -
          affine_basis * ldlt.solve(affine_basis.transpose());
      knot_projector =
          0.5 * (knot_projector + knot_projector.transpose()).eval();
      if (!knot_projector.allFinite())
      {
        return Eigen::MatrixXd();
      }

      if (!reference_rotations.empty() &&
          static_cast<int>(reference_rotations.size()) != knot_num)
      {
        return Eigen::MatrixXd();
      }

      // CASR uses interleaved coordinates
      // [r*dtheta_0, dp_0, ..., r*dtheta_(K-1), dp_(K-1)]. Rotation
      // increments live in different right-tangent frames.  The block
      // R_i^T R_j transports the j-th increment through the common world
      // frame before the affine temporal projection and transports it back
      // to the i-th tangent frame. Translation increments are already in the
      // world frame.
      Eigen::MatrixXd projector =
          Eigen::MatrixXd::Zero(6 * knot_num, 6 * knot_num);
      for (int row = 0; row < knot_num; ++row)
      {
        for (int col = 0; col < knot_num; ++col)
        {
          const double coefficient = knot_projector(row, col);
          Eigen::Matrix<double, 6, 6> block =
              Eigen::Matrix<double, 6, 6>::Zero();
          if (reference_rotations.empty())
          {
            block.block<3, 3>(0, 0) =
                coefficient * Eigen::Matrix3d::Identity();
          }
          else
          {
            block.block<3, 3>(0, 0) =
                coefficient *
                (reference_rotations[static_cast<size_t>(row)].inverse() *
                 reference_rotations[static_cast<size_t>(col)])
                    .matrix();
          }
          block.block<3, 3>(3, 3) =
              coefficient * Eigen::Matrix3d::Identity();
          projector.block<6, 6>(6 * row, 6 * col) = block;
        }
      }
      return projector;
    }

    inline Eigen::MatrixXd BuildCasrAffineNullspaceProjector(
        const std::vector<double> &knot_times_seconds)
    {
      return BuildCasrAffineNullspaceProjector(
          knot_times_seconds,
          Eigen::aligned_vector<Sophus::SO3<double>>());
    }

    inline Eigen::MatrixXd BuildCasrContinuityConstraintBasis(
        const Eigen::MatrixXd &recovery_basis,
        const Eigen::MatrixXd &affine_nullspace_projector)
    {
      if (recovery_basis.rows() <= 0 || recovery_basis.cols() <= 0 ||
          affine_nullspace_projector.rows() != recovery_basis.rows() ||
          affine_nullspace_projector.cols() != recovery_basis.rows() ||
          !recovery_basis.allFinite() ||
          !affine_nullspace_projector.allFinite())
      {
        return Eigen::MatrixXd();
      }
      return affine_nullspace_projector.transpose() * recovery_basis;
    }

    inline Eigen::MatrixXd BuildCasrCoupledConstraintBasis(
        const Eigen::MatrixXd &recovery_basis,
        const Eigen::MatrixXd &affine_nullspace_projector)
    {
      const Eigen::MatrixXd continuity_basis =
          BuildCasrContinuityConstraintBasis(
              recovery_basis, affine_nullspace_projector);
      if (continuity_basis.rows() != recovery_basis.rows() ||
          continuity_basis.cols() != recovery_basis.cols())
      {
        return Eigen::MatrixXd();
      }
      Eigen::MatrixXd stacked(recovery_basis.rows(),
                              2 * recovery_basis.cols());
      stacked.leftCols(recovery_basis.cols()) = recovery_basis;
      stacked.rightCols(recovery_basis.cols()) = continuity_basis;
      return stacked;
    }

    inline Eigen::VectorXd BuildCasrCoupledWeights(
        const Eigen::VectorXd &sqrt_information_weights)
    {
      if (sqrt_information_weights.size() <= 0)
      {
        return Eigen::VectorXd();
      }
      Eigen::VectorXd stacked(2 * sqrt_information_weights.size());
      const Eigen::VectorXd conservative =
          sqrt_information_weights / std::sqrt(2.0);
      stacked.head(sqrt_information_weights.size()) = conservative;
      stacked.tail(sqrt_information_weights.size()) = conservative;
      return stacked;
    }

    class CasrSubspaceFactor : public ceres::CostFunction
    {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      using SO3d = Sophus::SO3<double>;

      CasrSubspaceFactor(
          const Eigen::MatrixXd &recovery_basis,
          const Eigen::aligned_vector<SO3d> &reference_rotations,
          const Eigen::aligned_vector<Eigen::Vector3d> &reference_positions,
          double characteristic_range,
          const Eigen::VectorXd &sqrt_information_weights)
          : recovery_basis_(recovery_basis),
            reference_rotations_(reference_rotations),
            reference_positions_(reference_positions),
            characteristic_range_(characteristic_range),
            sqrt_information_weights_(sqrt_information_weights),
            control_point_num_(
                static_cast<int>(reference_rotations.size()))
      {
        set_num_residuals(static_cast<int>(recovery_basis_.cols()));
        for (int i = 0; i < control_point_num_; ++i)
        {
          mutable_parameter_block_sizes()->push_back(4);
        }
        for (int i = 0; i < control_point_num_; ++i)
        {
          mutable_parameter_block_sizes()->push_back(3);
        }
      }

      CasrSubspaceFactor(
          const Eigen::MatrixXd &recovery_basis,
          const Eigen::aligned_vector<SO3d> &reference_rotations,
          const Eigen::aligned_vector<Eigen::Vector3d> &reference_positions,
          double characteristic_range,
          double sqrt_information_weight)
          : CasrSubspaceFactor(
                recovery_basis, reference_rotations, reference_positions,
                characteristic_range,
                Eigen::VectorXd::Constant(
                    recovery_basis.cols(), sqrt_information_weight))
      {
      }

      bool IsValid() const
      {
        return control_point_num_ > 0 && num_residuals() > 0 &&
               static_cast<int>(reference_positions_.size()) ==
                   control_point_num_ &&
               recovery_basis_.rows() == 6 * control_point_num_ &&
               recovery_basis_.cols() == num_residuals() &&
               recovery_basis_.allFinite() &&
               std::isfinite(characteristic_range_) &&
               characteristic_range_ > 0.0 &&
               sqrt_information_weights_.size() == num_residuals() &&
               sqrt_information_weights_.allFinite() &&
               (sqrt_information_weights_.array() >= 0.0).all() &&
               sqrt_information_weights_.maxCoeff() > 0.0;
      }

      bool Evaluate(double const *const *parameters, double *residuals,
                    double **jacobians) const override
      {
        if (!IsValid())
        {
          return false;
        }

        Eigen::VectorXd scaled_increment =
            Eigen::VectorXd::Zero(6 * control_point_num_);
        std::vector<Eigen::Matrix3d,
                    Eigen::aligned_allocator<Eigen::Matrix3d>>
            rotation_log_jacobians(
                static_cast<size_t>(control_point_num_),
                Eigen::Matrix3d::Identity());
        for (int i = 0; i < control_point_num_; ++i)
        {
          const Eigen::Map<SO3d const> rotation(parameters[i]);
          const Eigen::Map<Eigen::Vector3d const> position(
              parameters[control_point_num_ + i]);
          const Eigen::Vector3d rotation_increment =
              (reference_rotations_[static_cast<size_t>(i)].inverse() *
               rotation)
                  .log();
          if (!rotation_increment.allFinite() || !position.allFinite())
          {
            return false;
          }
          scaled_increment.segment<3>(6 * i) =
              characteristic_range_ * rotation_increment;
          scaled_increment.segment<3>(6 * i + 3) =
              position - reference_positions_[static_cast<size_t>(i)];
          if (jacobians)
          {
            Sophus::rightJacobianInvSO3(
                rotation_increment,
                rotation_log_jacobians[static_cast<size_t>(i)]);
          }
        }

        Eigen::Map<Eigen::VectorXd> residual(residuals, num_residuals());
        residual =
            (sqrt_information_weights_.array() *
             (recovery_basis_.transpose() * scaled_increment).array())
                .matrix();

        if (!jacobians)
        {
          return residual.allFinite();
        }
        for (int i = 0; i < control_point_num_; ++i)
        {
          if (jacobians[i])
          {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>
                jacobian_rotation(jacobians[i], num_residuals(), 4);
            jacobian_rotation.setZero();
            jacobian_rotation.leftCols<3>() =
                characteristic_range_ *
                recovery_basis_.block(6 * i, 0, 3, num_residuals())
                    .transpose() *
                rotation_log_jacobians[static_cast<size_t>(i)];
            jacobian_rotation.leftCols<3>() =
                sqrt_information_weights_.asDiagonal() *
                jacobian_rotation.leftCols<3>().eval();
          }
          if (jacobians[control_point_num_ + i])
          {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>
                jacobian_position(
                    jacobians[control_point_num_ + i],
                    num_residuals(), 3);
            jacobian_position =
                sqrt_information_weights_.asDiagonal() *
                recovery_basis_.block(6 * i + 3, 0, 3,
                                      num_residuals())
                    .transpose();
          }
        }
        return residual.allFinite();
      }

    private:
      Eigen::MatrixXd recovery_basis_;
      Eigen::aligned_vector<SO3d> reference_rotations_;
      Eigen::aligned_vector<Eigen::Vector3d> reference_positions_;
      double characteristic_range_ = 1.0;
      Eigen::VectorXd sqrt_information_weights_;
      int control_point_num_ = 0;
    };

    class CasrPropagationReferenceFactor final
        : public CasrSubspaceFactor
    {
    public:
      CasrPropagationReferenceFactor(
          const Eigen::MatrixXd &recovery_basis,
          const Eigen::aligned_vector<SO3d> &reference_rotations,
          const Eigen::aligned_vector<Eigen::Vector3d> &reference_positions,
          double characteristic_range,
          const Eigen::VectorXd &sqrt_information_weights)
          : CasrSubspaceFactor(
                recovery_basis, reference_rotations, reference_positions,
                characteristic_range, sqrt_information_weights)
      {
      }
    };

    class CasrSplineContinuityFactor final : public CasrSubspaceFactor
    {
    public:
      CasrSplineContinuityFactor(
          const Eigen::MatrixXd &recovery_basis,
          const Eigen::MatrixXd &affine_nullspace_projector,
          const Eigen::aligned_vector<SO3d> &reference_rotations,
          const Eigen::aligned_vector<Eigen::Vector3d> &reference_positions,
          double characteristic_range,
          const Eigen::VectorXd &sqrt_information_weights)
          : CasrSubspaceFactor(
                BuildCasrContinuityConstraintBasis(
                    recovery_basis, affine_nullspace_projector),
                reference_rotations, reference_positions,
                characteristic_range, sqrt_information_weights)
      {
      }
    };

    class CasrCoupledConsensusFactor final : public CasrSubspaceFactor
    {
    public:
      CasrCoupledConsensusFactor(
          const Eigen::MatrixXd &recovery_basis,
          const Eigen::MatrixXd &affine_nullspace_projector,
          const Eigen::aligned_vector<SO3d> &reference_rotations,
          const Eigen::aligned_vector<Eigen::Vector3d> &reference_positions,
          double characteristic_range,
          const Eigen::VectorXd &sqrt_information_weights)
          : CasrSubspaceFactor(
                BuildCasrCoupledConstraintBasis(
                    recovery_basis, affine_nullspace_projector),
                reference_rotations, reference_positions,
                characteristic_range,
                BuildCasrCoupledWeights(sqrt_information_weights))
      {
      }
    };

  } // namespace analytic_derivative
} // namespace cocolic
