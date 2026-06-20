#pragma once

#include <array>

#include <ceres/ceres.h>
#include <Eigen/Eigen>

#include <lidar/lidar_feature.h>
#include <odom/factor/analytic_diff/lidar_feature_factor.h>

namespace cocolic
{
namespace analytic_derivative
{

struct LinearizedLoamFeatureNURBSData
{
  using SO3d = Sophus::SO3<double>;
  using Vec3d = Eigen::Vector3d;
  using Row3d = Eigen::Matrix<double, 1, 3>;

  double residual0 = 0.0;

  std::array<SO3d, 4> R_ref;
  std::array<Vec3d, 4> p_ref;

  std::array<Row3d, 4> J_rot;
  std::array<Row3d, 4> J_pos;
};

class LinearizedLoamFeatureFactorNURBS : public ceres::CostFunction
{
public:
  using SO3d = Sophus::SO3<double>;
  using Vec3d = Eigen::Vector3d;
  using Row3d = Eigen::Matrix<double, 1, 3>;

  explicit LinearizedLoamFeatureFactorNURBS(
      const LinearizedLoamFeatureNURBSData &data)
      : data_(data)
  {
    set_num_residuals(1);

    for (size_t i = 0; i < 4; ++i)
      mutable_parameter_block_sizes()->push_back(4);

    for (size_t i = 0; i < 4; ++i)
      mutable_parameter_block_sizes()->push_back(3);
  }

  virtual bool Evaluate(double const *const *parameters,
                        double *residuals,
                        double **jacobians) const
  {
    double r = data_.residual0;

    for (int i = 0; i < 4; ++i)
    {
      Eigen::Map<SO3d const> R_cur(parameters[i]);
      const Vec3d delta_theta =
          (data_.R_ref[i].inverse() * R_cur).log();

      r += data_.J_rot[i] * delta_theta;
    }

    for (int i = 0; i < 4; ++i)
    {
      Eigen::Map<Vec3d const> p_cur(parameters[4 + i]);
      const Vec3d delta_p = p_cur - data_.p_ref[i];

      r += data_.J_pos[i] * delta_p;
    }

    residuals[0] = r;

    if (!jacobians)
      return true;

    for (int i = 0; i < 4; ++i)
    {
      if (jacobians[i])
      {
        Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>> J(jacobians[i]);
        J.setZero();
        J.template block<1, 3>(0, 0) = data_.J_rot[i];
      }
    }

    for (int i = 0; i < 4; ++i)
    {
      if (jacobians[4 + i])
      {
        Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> J(jacobians[4 + i]);
        J = data_.J_pos[i];
      }
    }

    return true;
  }

private:
  LinearizedLoamFeatureNURBSData data_;
};

class LoamNURBSPoseRemapLinearizationHelper
{
public:
  using Functor = LoamFeatureFactorNURBS;
  using SO3View = Functor::SO3View;
  using R3View = Functor::R3View;

  using SO3d = Sophus::SO3<double>;
  using Vec3d = Eigen::Vector3d;
  using Mat3d = Eigen::Matrix3d;
  using Row3d = Eigen::Matrix<double, 1, 3>;

  static bool Build(const PointCorrespondence &pc,
                    const std::pair<int, double> &su,
                    const Eigen::Matrix4d &blending_matrix,
                    const Eigen::Matrix4d &cumulative_blending_matrix,
                    const SO3d &S_GtoM,
                    const Vec3d &p_GinM,
                    const SO3d &S_LtoI,
                    const Vec3d &p_LinI,
                    double weight,
                    double const *const *parameters,
                    const Eigen::Matrix<double, 6, 6> &pose_remap_matrix,
                    LinearizedLoamFeatureNURBSData *data)
  {
    if (!data)
      return false;

    typename SO3View::JacobianStruct J_R;
    typename R3View::JacobianStruct J_p;

    for (int i = 0; i < 4; ++i)
    {
      Eigen::Map<SO3d const> R_ref(parameters[i]);
      data->R_ref[i] = R_ref;
    }

    for (int i = 0; i < 4; ++i)
    {
      Eigen::Map<Vec3d const> p_ref(parameters[4 + i]);
      data->p_ref[i] = p_ref;
    }

    Vec3d p_Lk = pc.point;
    Vec3d p_IK = S_LtoI * p_Lk + p_LinI;

    SO3d S_ItoG = SO3View::EvaluateRpNURBS(su,
                                            cumulative_blending_matrix,
                                            parameters,
                                            &J_R);

    Vec3d p_IinG = R3View::evaluateNURBS(su,
                                         blending_matrix,
                                         parameters + 4,
                                         &J_p);

    Vec3d p_M = S_GtoM * (S_ItoG * p_IK + p_IinG) + p_GinM;

    Vec3d J_pi = Vec3d::Zero();
    double residual = 0.0;

    if (GeometryType::Plane == pc.geo_type)
    {
      residual = p_M.transpose() * pc.geo_plane.head(3) + pc.geo_plane[3];
      J_pi = pc.geo_plane.head(3);
    }
    else
    {
      Vec3d dist_vec = (p_M - pc.geo_point).cross(pc.geo_normal);
      residual = dist_vec.norm();

      if (residual < 1e-12)
        return false;

      J_pi = -dist_vec.transpose() / residual * SO3d::hat(pc.geo_normal);
    }

    data->residual0 = residual * weight;

    Mat3d J_Xm_R = -S_GtoM.matrix() * S_ItoG.matrix() * SO3d::hat(p_IK);

    Vec3d jac_lhs_R = J_pi.transpose() * J_Xm_R;
    Vec3d jac_lhs_P = J_pi.transpose() * S_GtoM.matrix();

    Eigen::Matrix<double, 1, 6> h_pose;
    h_pose.template block<1, 3>(0, 0) = weight * jac_lhs_R.transpose();
    h_pose.template block<1, 3>(0, 3) = weight * jac_lhs_P.transpose();

    Eigen::Matrix<double, 1, 6> h_pose_remap = h_pose * pose_remap_matrix;

    Row3d h_R = h_pose_remap.template block<1, 3>(0, 0);
    Row3d h_P = h_pose_remap.template block<1, 3>(0, 3);

    for (int i = 0; i < 4; ++i)
    {
      data->J_rot[i] = h_R * J_R.d_val_d_knot[i];
      data->J_pos[i] = J_p.d_val_d_knot[i] * h_P;
    }

    if (!std::isfinite(data->residual0))
      return false;

    for (int i = 0; i < 4; ++i)
    {
      if (!data->J_rot[i].allFinite() || !data->J_pos[i].allFinite())
        return false;
    }

    return true;
  }
};

} // namespace analytic_derivative
} // namespace cocolic
