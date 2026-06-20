#include <odom/lidar_degeneracy_analyzer.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sys/stat.h>
#include <sys/types.h>

namespace cocolic
{

LidarDegeneracyParam::LidarDegeneracyParam(const YAML::Node &node)
{
  if (!node)
    return;

  if (node["enable"])
    enable = node["enable"].as<bool>();

  if (node["stage"])
    stage = node["stage"].as<int>();

  if (node["verify_enable"])
    verify_enable = node["verify_enable"].as<bool>();

  if (node["low_dim_mode"])
    low_dim_mode = node["low_dim_mode"].as<std::string>();

  if (node["log_enable"])
    log_enable = node["log_enable"].as<bool>();

  if (node["log_dir"])
    log_dir = node["log_dir"].as<std::string>();

  if (node["replace_lidar_when_degenerate"])
    replace_lidar_when_degenerate =
        node["replace_lidar_when_degenerate"].as<bool>();

  if (node["min_corr_num"])
    min_corr_num = node["min_corr_num"].as<int>();

  if (node["min_time_span_ratio"])
    min_time_span_ratio = node["min_time_span_ratio"].as<double>();

  if (node["eigen_ratio_threshold"])
    eigen_ratio_threshold = node["eigen_ratio_threshold"].as<double>();

  if (node["condition_number_threshold"])
    condition_number_threshold =
        node["condition_number_threshold"].as<double>();

  if (node["trans_rot_scale"])
    trans_rot_scale = node["trans_rot_scale"].as<double>();

  if (node["remap_gamma"])
    remap_gamma = node["remap_gamma"].as<double>();

  if (node["time_block_num"])
    time_block_num = node["time_block_num"].as<int>();

  if (node["min_block_corr_num"])
    min_block_corr_num = node["min_block_corr_num"].as<int>();

  time_block_num = std::max(1, time_block_num);
  min_block_corr_num = std::max(1, min_block_corr_num);
}

LidarDegeneracyAnalyzer::LidarDegeneracyAnalyzer(
    const LidarDegeneracyParam &param)
    : param_(param)
{
}

int LidarDegeneracyAnalyzer::QDim() const
{
  // Route-B-local uses the pose-level 6-DoF Jacobian H_pose.
  // low_dim_mode is kept only for yaml compatibility.
  return 6;
}

int LidarDegeneracyAnalyzer::GetTimeBlockIndex(
    int64_t t_point,
    int64_t opt_min_t_ns,
    int64_t opt_max_t_ns) const
{
  const int block_num = std::max(1, param_.time_block_num);
  const int64_t duration = std::max<int64_t>(1, opt_max_t_ns - opt_min_t_ns);

  double ratio = static_cast<double>(t_point - opt_min_t_ns) /
                 static_cast<double>(duration);

  int block_idx = static_cast<int>(std::floor(ratio * block_num));

  if (block_idx < 0)
    block_idx = 0;
  if (block_idx >= block_num)
    block_idx = block_num - 1;

  return block_idx;
}

LidarDegeneracyReport LidarDegeneracyAnalyzer::Analyze(
    const Eigen::aligned_vector<PointCorrespondence> &point_corrs,
    const Trajectory::Ptr &trajectory,
    int64_t opt_min_t_ns,
    int64_t opt_max_t_ns,
    int64_t last_scan_end_ns,
    const SO3d &S_GtoM,
    const Eigen::Vector3d &p_GinM,
    const SO3d &S_LtoI,
    const Eigen::Vector3d &p_LinI,
    double lidar_weight,
    bool use_lidar_scale)
{
  LidarDegeneracyReport report;
  report.enabled = param_.enable;
  report.stage = param_.stage;
  report.q_dim = QDim();

  report.frame_idx = frame_idx_++;
  report.opt_min_t_ns = opt_min_t_ns;
  report.opt_max_t_ns = opt_max_t_ns;
  report.remap_gamma = param_.remap_gamma;
  report.pose_remap_matrix.setIdentity();
  report.block_num = std::max(1, param_.time_block_num);

  last_use_remap_flags_.assign(point_corrs.size(), false);
  last_pose_remap_matrices_.assign(
      point_corrs.size(), PoseRemapMatrix::Identity());
  last_block_reports_.clear();

  if (!param_.enable || param_.stage <= 0 || !trajectory)
    return report;

  const int q = report.q_dim;
  const int block_num = std::max(1, param_.time_block_num);
  const int64_t window_duration =
      std::max<int64_t>(1, opt_max_t_ns - opt_min_t_ns);
  const int64_t block_duration =
      std::max<int64_t>(1, window_duration / block_num);

  std::vector<PoseBlockWork, Eigen::aligned_allocator<PoseBlockWork>> blocks(
      block_num);

  for (int b = 0; b < block_num; ++b)
  {
    blocks[b].block_idx = b;
    blocks[b].block_start_t_ns = opt_min_t_ns + b * block_duration;

    if (b == block_num - 1)
      blocks[b].block_end_t_ns = opt_max_t_ns;
    else
      blocks[b].block_end_t_ns =
          std::min<int64_t>(opt_max_t_ns,
                            opt_min_t_ns + (b + 1) * block_duration);

    blocks[b].min_t_ns = std::numeric_limits<int64_t>::max();
    blocks[b].max_t_ns = std::numeric_limits<int64_t>::min();
    blocks[b].A_pose.setZero();
  }

  Eigen::Matrix<double, 6, 6> A_pose_window =
      Eigen::Matrix<double, 6, 6>::Zero();

  std::vector<Eigen::Matrix<double, 1, 6>,
              Eigen::aligned_allocator<Eigen::Matrix<double, 1, 6>>>
      h_pose_rows_window;

  double residual_norm2_window = 0.0;
  bool has_time_window = false;
  int64_t min_t_window = std::numeric_limits<int64_t>::max();
  int64_t max_t_window = std::numeric_limits<int64_t>::min();

  for (size_t i = 0; i < point_corrs.size(); ++i)
  {
    const auto &pc = point_corrs[i];

    // Keep exactly the same temporal filtering logic as the original LiDAR factor.
    if (pc.t_point < opt_min_t_ns)
      continue;
    if (pc.t_point >= opt_max_t_ns)
      continue;
    if (pc.t_point < last_scan_end_ns)
      continue;

    const double weight =
        use_lidar_scale ? (lidar_weight * pc.scale) : lidar_weight;

    Eigen::Matrix<double, 1, 6> h_pose;
    double residual = 0.0;

    if (!BuildPoseJacobian(pc,
                           trajectory,
                           S_GtoM,
                           p_GinM,
                           S_LtoI,
                           p_LinI,
                           weight,
                           &h_pose,
                           &residual))
    {
      continue;
    }

    if (!h_pose.allFinite() || !std::isfinite(residual))
      continue;

    const int block_idx = GetTimeBlockIndex(pc.t_point,
                                            opt_min_t_ns,
                                            opt_max_t_ns);

    PoseRowData row;
    row.point_index = static_cast<int>(i);
    row.h_pose = h_pose;
    row.residual = residual;
    row.geo_type = pc.geo_type;
    row.t_point = pc.t_point;

    PoseBlockWork &block = blocks[block_idx];
    block.rows.push_back(row);
    block.A_pose.noalias() += h_pose.transpose() * h_pose;
    block.residual_norm2 += residual * residual;
    block.used_corr_num++;

    if (pc.geo_type == GeometryType::Plane)
      block.plane_num++;
    else
      block.line_num++;

    block.min_t_ns = std::min(block.min_t_ns, pc.t_point);
    block.max_t_ns = std::max(block.max_t_ns, pc.t_point);
    block.has_time = true;

    A_pose_window.noalias() += h_pose.transpose() * h_pose;
    h_pose_rows_window.push_back(h_pose);
    residual_norm2_window += residual * residual;

    report.used_corr_num++;
    if (pc.geo_type == GeometryType::Plane)
      report.plane_num++;
    else
      report.line_num++;

    min_t_window = std::min(min_t_window, pc.t_point);
    max_t_window = std::max(max_t_window, pc.t_point);
    has_time_window = true;
  }

  // A_red field is kept for backward compatibility. In Route-B-local it stores
  // the window-level A_pose, while final replacement decisions are block-level.
  report.A_red = A_pose_window;

  if (report.used_corr_num <= 0 || !has_time_window)
  {
    WriteWindowLog(report);
    return report;
  }

  report.min_t_ns = min_t_window;
  report.max_t_ns = max_t_window;
  report.mid_t_ns = min_t_window + (max_t_window - min_t_window) / 2;

  const double window_denom =
      static_cast<double>(std::max<int64_t>(1, opt_max_t_ns - opt_min_t_ns));

  report.time_span_ratio =
      static_cast<double>(max_t_window - min_t_window) / window_denom;

  report.residual_norm = std::sqrt(residual_norm2_window);
  report.residual_rms =
      std::sqrt(residual_norm2_window /
                static_cast<double>(std::max(1, report.used_corr_num)));

  // Window-level diagnostic is still logged for reference only.
  Eigen::VectorXd inv_sqrt_diag_window;
  Eigen::VectorXd raw_eigenvalues_window;
  Eigen::VectorXd eigenvalues_window;
  Eigen::MatrixXd eigenvectors_window;

  const bool window_valid = AnalyzePoseMatrix(A_pose_window,
                                              h_pose_rows_window,
                                              &inv_sqrt_diag_window,
                                              &raw_eigenvalues_window,
                                              &eigenvalues_window,
                                              &eigenvectors_window,
                                              &report.raw_min_eigen,
                                              &report.raw_max_eigen,
                                              &report.raw_eigen_ratio,
                                              &report.raw_condition_number,
                                              &report.min_eigen,
                                              &report.max_eigen,
                                              &report.eigen_ratio,
                                              &report.condition_number,
                                              &report.pose_remap_matrix,
                                              &report.weak_dim,
                                              &report.weak_eigenvalues,
                                              &report.weak_energy_before,
                                              &report.weak_energy_after,
                                              &report.weak_attenuation_ratio,
                                              &report.strong_energy_before,
                                              &report.strong_energy_after,
                                              &report.strong_preserve_ratio);

  if (window_valid)
  {
    report.valid = true;
    report.raw_eigenvalues = raw_eigenvalues_window;
    report.eigenvalues = eigenvalues_window;
    report.eigenvectors = eigenvectors_window;
    report.A_norm = inv_sqrt_diag_window.asDiagonal() *
                    A_pose_window *
                    inv_sqrt_diag_window.asDiagonal();
    report.spectral_degenerate =
        (report.eigen_ratio < param_.eigen_ratio_threshold) ||
        (report.condition_number > param_.condition_number_threshold);
    report.raw_degenerate = report.spectral_degenerate;
  }

  ApplyWindowVerification(&report);

  // Block-level diagnostic and replacement decisions.
  for (int b = 0; b < block_num; ++b)
  {
    PoseBlockWork &block = blocks[b];

    LidarDegeneracyBlockReport block_report;
    block_report.frame_idx = report.frame_idx;
    block_report.block_idx = b;
    block_report.stage = param_.stage;
    block_report.q_dim = q;
    block_report.block_start_t_ns = block.block_start_t_ns;
    block_report.block_end_t_ns = block.block_end_t_ns;
    block_report.used_corr_num = block.used_corr_num;
    block_report.plane_num = block.plane_num;
    block_report.line_num = block.line_num;
    block_report.remap_gamma = param_.remap_gamma;

    if (block.used_corr_num > 0 && block.has_time)
    {
      block_report.min_t_ns = block.min_t_ns;
      block_report.max_t_ns = block.max_t_ns;
      block_report.mid_t_ns =
          block.min_t_ns + (block.max_t_ns - block.min_t_ns) / 2;

      const double block_denom =
          static_cast<double>(std::max<int64_t>(1,
                                               block.block_end_t_ns -
                                                   block.block_start_t_ns));

      block_report.time_span_ratio =
          static_cast<double>(block.max_t_ns - block.min_t_ns) / block_denom;

      block_report.residual_norm = std::sqrt(block.residual_norm2);
      block_report.residual_rms =
          std::sqrt(block.residual_norm2 /
                    static_cast<double>(std::max(1, block.used_corr_num)));

      std::vector<Eigen::Matrix<double, 1, 6>,
                  Eigen::aligned_allocator<Eigen::Matrix<double, 1, 6>>>
          h_pose_rows_block;
      h_pose_rows_block.reserve(block.rows.size());

      for (const auto &row : block.rows)
        h_pose_rows_block.push_back(row.h_pose);

      Eigen::VectorXd inv_sqrt_diag_block;
      Eigen::VectorXd raw_eigenvalues_block;
      Eigen::VectorXd eigenvalues_block;
      Eigen::MatrixXd eigenvectors_block;

      const bool block_valid = AnalyzePoseMatrix(block.A_pose,
                                                 h_pose_rows_block,
                                                 &inv_sqrt_diag_block,
                                                 &raw_eigenvalues_block,
                                                 &eigenvalues_block,
                                                 &eigenvectors_block,
                                                 &block_report.raw_min_eigen,
                                                 &block_report.raw_max_eigen,
                                                 &block_report.raw_eigen_ratio,
                                                 &block_report.raw_condition_number,
                                                 &block_report.min_eigen,
                                                 &block_report.max_eigen,
                                                 &block_report.eigen_ratio,
                                                 &block_report.condition_number,
                                                 &block_report.pose_remap_matrix,
                                                 &block_report.weak_dim,
                                                 &block_report.weak_eigenvalues,
                                                 &block_report.weak_energy_before,
                                                 &block_report.weak_energy_after,
                                                 &block_report.weak_attenuation_ratio,
                                                 &block_report.strong_energy_before,
                                                 &block_report.strong_energy_after,
                                                 &block_report.strong_preserve_ratio);

      block_report.valid = block_valid;

      if (block_valid)
      {
        block_report.eigenvalues = eigenvalues_block;
        block_report.eigenvectors = eigenvectors_block;

        block_report.spectral_degenerate =
            (block_report.eigen_ratio < param_.eigen_ratio_threshold) ||
            (block_report.condition_number > param_.condition_number_threshold);

        const bool enough_corr =
            block_report.used_corr_num >= param_.min_block_corr_num;
        const bool finite_values =
            std::isfinite(block_report.eigen_ratio) &&
            std::isfinite(block_report.condition_number);

        // For local time blocks, reliability is based on correspondence count and
        // finite spectrum. Time-span ratio is logged but not used as a hard gate,
        // because a valid local LiDAR batch can be temporally compact.
        block_report.reliable = enough_corr && finite_values;
        block_report.final_degenerate =
            block_report.reliable && block_report.spectral_degenerate;
      }
    }

    if (block_report.valid)
      report.valid_block_num++;
    if (block_report.reliable)
      report.reliable_block_num++;
    if (block_report.spectral_degenerate)
      report.spectral_block_num++;
    if (block_report.final_degenerate)
      report.final_block_num++;

    if (block_report.final_degenerate)
    {
      block_report.remap_item_num = block.used_corr_num;
      for (const auto &row : block.rows)
      {
        if (row.point_index >= 0 &&
            static_cast<size_t>(row.point_index) < last_use_remap_flags_.size())
        {
          last_use_remap_flags_[row.point_index] = true;
          last_pose_remap_matrices_[row.point_index] =
              block_report.pose_remap_matrix;
        }
      }
    }

    report.remap_item_num += block_report.remap_item_num;
    last_block_reports_.push_back(block_report);
    WriteBlockLog(block_report);
  }

  report.final_degenerate = report.final_block_num > 0;
  report.is_degenerate = report.final_degenerate;
  report.raw_degenerate = report.spectral_degenerate;
  report.reliable = report.reliable_block_num > 0;

  WriteWindowLog(report);

  return report;
}

bool LidarDegeneracyAnalyzer::BuildPoseJacobian(
    const PointCorrespondence &pc,
    const Trajectory::Ptr &trajectory,
    const SO3d &S_GtoM,
    const Eigen::Vector3d &p_GinM,
    const SO3d &S_LtoI,
    const Eigen::Vector3d &p_LinI,
    double weight,
    Eigen::Matrix<double, 1, 6> *h_pose,
    double *residual_out) const
{
  if (!trajectory || !h_pose)
    return false;

  const int64_t time_ns = pc.t_point;

  std::pair<int, double> su;
  trajectory->GetIdxT(time_ns, su);

  const int start_idx = su.first - 3;

  if (start_idx < 0)
    return false;

  if (start_idx >= static_cast<int>(trajectory->blending_mats.size()))
    return false;

  if (start_idx >= static_cast<int>(trajectory->cumu_blending_mats.size()))
    return false;

  using Functor = analytic_derivative::LoamFeatureFactorNURBS;
  using SO3View = Functor::SO3View;
  using R3View = Functor::R3View;
  using Vec3d = Eigen::Vector3d;
  using Mat3d = Eigen::Matrix3d;

  typename SO3View::JacobianStruct J_R;
  typename R3View::JacobianStruct J_p;

  double const *parameters[8];

  for (int i = 0; i < 4; ++i)
    parameters[i] = trajectory->getKnotSO3(start_idx + i).data();

  for (int i = 0; i < 4; ++i)
    parameters[4 + i] = trajectory->getKnotPos(start_idx + i).data();

  const Eigen::Matrix4d &blending_matrix =
      trajectory->blending_mats[start_idx];

  const Eigen::Matrix4d &cumulative_blending_matrix =
      trajectory->cumu_blending_mats[start_idx];

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

  residual *= weight;

  Mat3d J_Xm_R = -S_GtoM.matrix() * S_ItoG.matrix() * SO3d::hat(p_IK);

  Vec3d jac_lhs_R = J_pi.transpose() * J_Xm_R;
  Vec3d jac_lhs_P = J_pi.transpose() * S_GtoM.matrix();

  h_pose->setZero();
  h_pose->template block<1, 3>(0, 0) = weight * jac_lhs_R.transpose();
  h_pose->template block<1, 3>(0, 3) = weight * jac_lhs_P.transpose();

  if (residual_out)
    *residual_out = residual;

  return h_pose->allFinite() && std::isfinite(residual);
}

bool LidarDegeneracyAnalyzer::AnalyzePoseMatrix(
    const Eigen::Matrix<double, 6, 6> &A_pose,
    const std::vector<Eigen::Matrix<double, 1, 6>,
                      Eigen::aligned_allocator<Eigen::Matrix<double, 1, 6>>> &h_pose_rows,
    Eigen::VectorXd *inv_sqrt_diag,
    Eigen::VectorXd *raw_eigenvalues,
    Eigen::VectorXd *eigenvalues,
    Eigen::MatrixXd *eigenvectors,
    double *raw_min_eigen,
    double *raw_max_eigen,
    double *raw_eigen_ratio,
    double *raw_condition_number,
    double *min_eigen,
    double *max_eigen,
    double *eigen_ratio,
    double *condition_number,
    Eigen::Matrix<double, 6, 6> *pose_remap_matrix,
    int *weak_dim,
    Eigen::VectorXd *weak_eigenvalues,
    double *weak_energy_before,
    double *weak_energy_after,
    double *weak_attenuation_ratio,
    double *strong_energy_before,
    double *strong_energy_after,
    double *strong_preserve_ratio) const
{
  const int q = 6;
  const double eps = 1e-12;

  if (h_pose_rows.empty())
    return false;

  if (inv_sqrt_diag)
  {
    inv_sqrt_diag->resize(q);
    inv_sqrt_diag->setZero();
  }

  if (pose_remap_matrix)
    pose_remap_matrix->setIdentity();

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> raw_es(A_pose);

  if (raw_es.info() != Eigen::Success)
    return false;

  Eigen::VectorXd raw_eval = raw_es.eigenvalues();

  if (raw_eigenvalues)
    *raw_eigenvalues = raw_eval;

  const double raw_min = std::max(0.0, raw_eval[0]);
  const double raw_max = std::max(0.0, raw_eval[raw_eval.size() - 1]);

  if (raw_min_eigen)
    *raw_min_eigen = raw_min;
  if (raw_max_eigen)
    *raw_max_eigen = raw_max;
  if (raw_eigen_ratio)
    *raw_eigen_ratio = raw_min / std::max(raw_max, eps);
  if (raw_condition_number)
    *raw_condition_number = raw_max / std::max(raw_min, eps);

  Eigen::VectorXd local_inv_sqrt_diag(q);
  local_inv_sqrt_diag.setZero();

  for (int i = 0; i < q; ++i)
  {
    const double di = A_pose(i, i);

    if (di > eps && std::isfinite(di))
      local_inv_sqrt_diag[i] = 1.0 / std::sqrt(di + eps);
    else
      local_inv_sqrt_diag[i] = 0.0;
  }

  if (inv_sqrt_diag)
    *inv_sqrt_diag = local_inv_sqrt_diag;

  Eigen::Matrix<double, 6, 6> D = local_inv_sqrt_diag.asDiagonal();
  Eigen::Matrix<double, 6, 6> A_norm = D * A_pose * D;
  A_norm = 0.5 * (A_norm + A_norm.transpose());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(A_norm);

  if (es.info() != Eigen::Success)
    return false;

  Eigen::VectorXd eval = es.eigenvalues();
  Eigen::MatrixXd evec = es.eigenvectors();

  if (eigenvalues)
    *eigenvalues = eval;
  if (eigenvectors)
    *eigenvectors = evec;

  const double min_ev = std::max(0.0, eval[0]);
  const double max_ev = std::max(0.0, eval[eval.size() - 1]);

  if (min_eigen)
    *min_eigen = min_ev;
  if (max_eigen)
    *max_eigen = max_ev;
  if (eigen_ratio)
    *eigen_ratio = min_ev / std::max(max_ev, eps);
  if (condition_number)
    *condition_number = max_ev / std::max(min_ev, eps);

  Eigen::MatrixXd H_pose(h_pose_rows.size(), q);
  for (size_t i = 0; i < h_pose_rows.size(); ++i)
    H_pose.row(static_cast<int>(i)) = h_pose_rows[i];

  Eigen::MatrixXd H_norm = H_pose * D;

  Eigen::VectorXd sqrt_diag(q);
  sqrt_diag.setZero();
  for (int i = 0; i < q; ++i)
  {
    if (local_inv_sqrt_diag[i] > eps && std::isfinite(local_inv_sqrt_diag[i]))
      sqrt_diag[i] = 1.0 / local_inv_sqrt_diag[i];
    else
      sqrt_diag[i] = 0.0;
  }
  Eigen::Matrix<double, 6, 6> D_inv = sqrt_diag.asDiagonal();

  const double lambda_max = std::max(eps, eval[eval.size() - 1]);

  std::vector<int> weak_ids;
  std::vector<int> strong_ids;

  for (int i = 0; i < q; ++i)
  {
    const double ratio = eval[i] / lambda_max;

    if (ratio < param_.eigen_ratio_threshold)
      weak_ids.push_back(i);
    else
      strong_ids.push_back(i);
  }

  if (weak_dim)
    *weak_dim = static_cast<int>(weak_ids.size());

  if (weak_ids.empty())
    return true;

  Eigen::MatrixXd V_weak(q, static_cast<int>(weak_ids.size()));
  Eigen::VectorXd weak_eval(static_cast<int>(weak_ids.size()));

  for (int i = 0; i < static_cast<int>(weak_ids.size()); ++i)
  {
    const int id = weak_ids[i];
    V_weak.col(i) = evec.col(id);
    weak_eval[i] = eval[id];
  }

  if (weak_eigenvalues)
    *weak_eigenvalues = weak_eval;

  Eigen::Matrix<double, 6, 6> P_q =
      Eigen::Matrix<double, 6, 6>::Identity() -
      (1.0 - param_.remap_gamma) * V_weak * V_weak.transpose();

  Eigen::MatrixXd H_norm_remap = H_norm * P_q;

  const double weak_before = (H_norm * V_weak).squaredNorm();
  const double weak_after = (H_norm_remap * V_weak).squaredNorm();

  if (weak_energy_before)
    *weak_energy_before = weak_before;
  if (weak_energy_after)
    *weak_energy_after = weak_after;
  if (weak_attenuation_ratio)
    *weak_attenuation_ratio = weak_after / std::max(weak_before, eps);

  if (!strong_ids.empty())
  {
    Eigen::MatrixXd V_strong(q, static_cast<int>(strong_ids.size()));

    for (int i = 0; i < static_cast<int>(strong_ids.size()); ++i)
      V_strong.col(i) = evec.col(strong_ids[i]);

    const double strong_before = (H_norm * V_strong).squaredNorm();
    const double strong_after = (H_norm_remap * V_strong).squaredNorm();

    if (strong_energy_before)
      *strong_energy_before = strong_before;
    if (strong_energy_after)
      *strong_energy_after = strong_after;
    if (strong_preserve_ratio)
      *strong_preserve_ratio = strong_after / std::max(strong_before, eps);
  }

  if (pose_remap_matrix)
  {
    // Convert normalized-space remapping back to original pose-jacobian coordinates:
    // h_pose_remap = h_pose * D * P_q * D^{-1}.
    *pose_remap_matrix = D * P_q * D_inv;
  }

  return true;
}

void LidarDegeneracyAnalyzer::ApplyWindowVerification(
    LidarDegeneracyReport *report) const
{
  if (!report || !report->valid)
    return;

  if (!param_.verify_enable)
  {
    report->reliable = true;
    report->is_degenerate = report->raw_degenerate;
    return;
  }

  const bool enough_corr =
      report->used_corr_num >= param_.min_corr_num;

  const bool enough_time =
      report->time_span_ratio >= param_.min_time_span_ratio;

  const bool finite_values =
      std::isfinite(report->eigen_ratio) &&
      std::isfinite(report->condition_number);

  report->reliable =
      enough_corr && enough_time && finite_values;

  report->is_degenerate =
      report->reliable && report->raw_degenerate;
}

void LidarDegeneracyAnalyzer::EnsureLogDir() const
{
  if (param_.log_dir.empty())
    return;

  mkdir(param_.log_dir.c_str(), 0755);
}

void LidarDegeneracyAnalyzer::WriteWindowLog(
    const LidarDegeneracyReport &report) const
{
  if (!param_.log_enable)
    return;

  EnsureLogDir();

  const std::string path =
      param_.log_dir + "/lidar_degeneracy_pose_window.csv";

  const bool file_exists =
      static_cast<bool>(std::ifstream(path));

  std::ofstream ofs(path, std::ios::app);

  if (!ofs.is_open())
    return;

  if (!file_exists)
  {
    ofs << "stage,frame_idx,"
        << "valid,reliable,spectral_degenerate,final_degenerate,"
        << "q_dim,used_corr_num,plane_num,line_num,"
        << "block_num,valid_block_num,reliable_block_num,spectral_block_num,final_block_num,remap_item_num,"
        << "opt_min_t_ns,opt_max_t_ns,min_t_ns,max_t_ns,mid_t_ns,"
        << "time_span_ratio,"
        << "raw_min_eigen,raw_max_eigen,raw_eigen_ratio,raw_condition_number,"
        << "norm_min_eigen,norm_max_eigen,norm_eigen_ratio,norm_condition_number,"
        << "weak_dim,remap_gamma,"
        << "residual_norm,residual_rms,"
        << "weak_energy_before,weak_energy_after,weak_attenuation_ratio,"
        << "strong_energy_before,strong_energy_after,strong_preserve_ratio,"
        << "norm_eigenvalues,weak_eigenvalues\n";
  }

  ofs << report.stage << ","
      << report.frame_idx << ","
      << report.valid << ","
      << report.reliable << ","
      << report.spectral_degenerate << ","
      << report.final_degenerate << ","
      << report.q_dim << ","
      << report.used_corr_num << ","
      << report.plane_num << ","
      << report.line_num << ","
      << report.block_num << ","
      << report.valid_block_num << ","
      << report.reliable_block_num << ","
      << report.spectral_block_num << ","
      << report.final_block_num << ","
      << report.remap_item_num << ","
      << report.opt_min_t_ns << ","
      << report.opt_max_t_ns << ","
      << report.min_t_ns << ","
      << report.max_t_ns << ","
      << report.mid_t_ns << ","
      << std::setprecision(12)
      << report.time_span_ratio << ","
      << report.raw_min_eigen << ","
      << report.raw_max_eigen << ","
      << report.raw_eigen_ratio << ","
      << report.raw_condition_number << ","
      << report.min_eigen << ","
      << report.max_eigen << ","
      << report.eigen_ratio << ","
      << report.condition_number << ","
      << report.weak_dim << ","
      << report.remap_gamma << ","
      << report.residual_norm << ","
      << report.residual_rms << ","
      << report.weak_energy_before << ","
      << report.weak_energy_after << ","
      << report.weak_attenuation_ratio << ","
      << report.strong_energy_before << ","
      << report.strong_energy_after << ","
      << report.strong_preserve_ratio << ",";

  ofs << "\"";
  if (report.eigenvalues.size() > 0)
  {
    for (int i = 0; i < report.eigenvalues.size(); ++i)
    {
      if (i)
        ofs << " ";
      ofs << report.eigenvalues[i];
    }
  }
  ofs << "\",";

  ofs << "\"";
  if (report.weak_eigenvalues.size() > 0)
  {
    for (int i = 0; i < report.weak_eigenvalues.size(); ++i)
    {
      if (i)
        ofs << " ";
      ofs << report.weak_eigenvalues[i];
    }
  }
  ofs << "\"\n";
}

void LidarDegeneracyAnalyzer::WriteBlockLog(
    const LidarDegeneracyBlockReport &block) const
{
  if (!param_.log_enable)
    return;

  EnsureLogDir();

  const std::string path =
      param_.log_dir + "/lidar_degeneracy_pose_block.csv";

  const bool file_exists =
      static_cast<bool>(std::ifstream(path));

  std::ofstream ofs(path, std::ios::app);

  if (!ofs.is_open())
    return;

  if (!file_exists)
  {
    ofs << "stage,frame_idx,block_idx,"
        << "valid,reliable,spectral_degenerate,final_degenerate,"
        << "q_dim,used_corr_num,plane_num,line_num,remap_item_num,"
        << "block_start_t_ns,block_end_t_ns,min_t_ns,max_t_ns,mid_t_ns,"
        << "time_span_ratio,"
        << "raw_min_eigen,raw_max_eigen,raw_eigen_ratio,raw_condition_number,"
        << "norm_min_eigen,norm_max_eigen,norm_eigen_ratio,norm_condition_number,"
        << "weak_dim,remap_gamma,"
        << "residual_norm,residual_rms,"
        << "weak_energy_before,weak_energy_after,weak_attenuation_ratio,"
        << "strong_energy_before,strong_energy_after,strong_preserve_ratio,"
        << "norm_eigenvalues,weak_eigenvalues\n";
  }

  ofs << block.stage << ","
      << block.frame_idx << ","
      << block.block_idx << ","
      << block.valid << ","
      << block.reliable << ","
      << block.spectral_degenerate << ","
      << block.final_degenerate << ","
      << block.q_dim << ","
      << block.used_corr_num << ","
      << block.plane_num << ","
      << block.line_num << ","
      << block.remap_item_num << ","
      << block.block_start_t_ns << ","
      << block.block_end_t_ns << ","
      << block.min_t_ns << ","
      << block.max_t_ns << ","
      << block.mid_t_ns << ","
      << std::setprecision(12)
      << block.time_span_ratio << ","
      << block.raw_min_eigen << ","
      << block.raw_max_eigen << ","
      << block.raw_eigen_ratio << ","
      << block.raw_condition_number << ","
      << block.min_eigen << ","
      << block.max_eigen << ","
      << block.eigen_ratio << ","
      << block.condition_number << ","
      << block.weak_dim << ","
      << block.remap_gamma << ","
      << block.residual_norm << ","
      << block.residual_rms << ","
      << block.weak_energy_before << ","
      << block.weak_energy_after << ","
      << block.weak_attenuation_ratio << ","
      << block.strong_energy_before << ","
      << block.strong_energy_after << ","
      << block.strong_preserve_ratio << ",";

  ofs << "\"";
  if (block.eigenvalues.size() > 0)
  {
    for (int i = 0; i < block.eigenvalues.size(); ++i)
    {
      if (i)
        ofs << " ";
      ofs << block.eigenvalues[i];
    }
  }
  ofs << "\",";

  ofs << "\"";
  if (block.weak_eigenvalues.size() > 0)
  {
    for (int i = 0; i < block.weak_eigenvalues.size(); ++i)
    {
      if (i)
        ofs << " ";
      ofs << block.weak_eigenvalues[i];
    }
  }
  ofs << "\"\n";
}

} // namespace cocolic
