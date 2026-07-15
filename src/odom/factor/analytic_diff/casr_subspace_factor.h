/*
 * CASR knot-space increment anchoring factor.
 */

#pragma once

#include <ceres/ceres.h>
#include <sophus_lib/so3.hpp>
#include <utils/sophus_utils.hpp>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <algorithm>
#include <cmath>
#include <vector>

namespace cocolic
{
  namespace analytic_derivative
  {

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
          double sqrt_information_weight)
          : recovery_basis_(recovery_basis),
            reference_rotations_(reference_rotations),
            reference_positions_(reference_positions),
            characteristic_range_(characteristic_range),
            sqrt_information_weight_(sqrt_information_weight),
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
               std::isfinite(sqrt_information_weight_) &&
               sqrt_information_weight_ > 0.0;
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
        residual = sqrt_information_weight_ *
                   recovery_basis_.transpose() * scaled_increment;

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
                sqrt_information_weight_ * characteristic_range_ *
                recovery_basis_.block(6 * i, 0, 3, num_residuals())
                    .transpose() *
                rotation_log_jacobians[static_cast<size_t>(i)];
          }
          if (jacobians[control_point_num_ + i])
          {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>
                jacobian_position(
                    jacobians[control_point_num_ + i],
                    num_residuals(), 3);
            jacobian_position =
                sqrt_information_weight_ *
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
      double sqrt_information_weight_ = 0.0;
      int control_point_num_ = 0;
    };

  } // namespace analytic_derivative
} // namespace cocolic
