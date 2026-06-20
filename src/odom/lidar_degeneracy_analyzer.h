#pragma once

#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <spline/trajectory.h>
#include <utils/parameter_struct.h>
#include <lidar/lidar_feature.h>
#include <odom/factor/analytic_diff/lidar_feature_factor.h>

namespace cocolic
{

struct LidarDegeneracyParam
{
  bool enable = false;

  // 0: original
  // 1: diagnostic only
  // 2: diagnostic + pose-level remapping verification
  // 3: replace original LiDAR factor with pose-remapped linearized factor
  int stage = 0;

  bool verify_enable = true;

  // Route-B uses pose-level 6-DoF Jacobian H_pose only.
  // This option is kept for yaml compatibility; stage 1/2/3 always use q_dim = 6.
  std::string low_dim_mode = "pose6";

  bool log_enable = true;
  std::string log_dir = "./degeneracy_log";

  // Stage 3 switch. Keep false in stage 1/2.
  bool replace_lidar_when_degenerate = false;

  // Window-level reliability parameters.
  int min_corr_num = 50;
  double min_time_span_ratio = 0.5;
  double eigen_ratio_threshold = 0.01;
  double condition_number_threshold = 1000.0;

  // Kept for yaml compatibility. Route-B pose-level H_pose does not use this.
  double trans_rot_scale = 1.0;

  // Stage-2/3 remapping ratio for weak pose directions.
  // gamma = 0.1 means weak pose directions keep 10% Jacobian magnitude.
  double remap_gamma = 0.1;

  // Route-B-local: split one optimization window into local LiDAR time blocks.
  // Each block has its own H_pose_block / P_pose_block.
  int time_block_num = 4;
  int min_block_corr_num = 30;

  LidarDegeneracyParam() = default;
  explicit LidarDegeneracyParam(const YAML::Node &node);
};

struct LidarDegeneracyReport
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool enabled = false;
  int stage = 0;

  bool valid = false;
  bool reliable = false;

  // Backward-compatible field name.
  bool raw_degenerate = false;
  bool is_degenerate = false;

  // Route-B always uses pose-level 6-DoF space:
  // [right perturbation rotation(3), translation(3)].
  int q_dim = 6;
  int used_corr_num = 0;
  int plane_num = 0;
  int line_num = 0;

  int64_t min_t_ns = 0;
  int64_t max_t_ns = 0;
  double time_span_ratio = 0.0;

  int64_t frame_idx = 0;
  int64_t opt_min_t_ns = 0;
  int64_t opt_max_t_ns = 0;
  int64_t mid_t_ns = 0;

  bool spectral_degenerate = false;
  bool final_degenerate = false;

  // Window/block summary.
  int block_num = 0;
  int valid_block_num = 0;
  int reliable_block_num = 0;
  int spectral_block_num = 0;
  int final_block_num = 0;
  int remap_item_num = 0;

  // Raw A_pose eigen information, only for logging and comparison.
  Eigen::VectorXd raw_eigenvalues;
  double raw_min_eigen = 0.0;
  double raw_max_eigen = 0.0;
  double raw_eigen_ratio = 0.0;
  double raw_condition_number = 0.0;

  // Normalized A_norm eigen information. Decision uses these values.
  Eigen::VectorXd eigenvalues;
  Eigen::MatrixXd eigenvectors;

  double min_eigen = 0.0;
  double max_eigen = 0.0;
  double eigen_ratio = 0.0;
  double condition_number = 0.0;

  // A_red is now A_pose = H_pose^T H_pose.
  // Name kept for compatibility with earlier analysis scripts.
  Eigen::MatrixXd A_red;
  Eigen::MatrixXd A_norm;

  int weak_dim = 0;
  double remap_gamma = 0.1;

  double residual_norm = 0.0;
  double residual_rms = 0.0;

  double weak_energy_before = -1.0;
  double weak_energy_after = -1.0;
  double weak_attenuation_ratio = -1.0;

  double strong_energy_before = -1.0;
  double strong_energy_after = -1.0;
  double strong_preserve_ratio = -1.0;

  Eigen::VectorXd weak_eigenvalues;

  // For window-level compatibility only. Route-B-local stage 3 uses one matrix per item.
  Eigen::Matrix<double, 6, 6> pose_remap_matrix =
      Eigen::Matrix<double, 6, 6>::Identity();
};

struct LidarDegeneracyBlockReport
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int64_t frame_idx = 0;
  int block_idx = 0;
  int stage = 0;
  int q_dim = 6;

  int64_t block_start_t_ns = 0;
  int64_t block_end_t_ns = 0;
  int64_t min_t_ns = 0;
  int64_t max_t_ns = 0;
  int64_t mid_t_ns = 0;
  double time_span_ratio = 0.0;

  int used_corr_num = 0;
  int plane_num = 0;
  int line_num = 0;
  int remap_item_num = 0;

  bool valid = false;
  bool reliable = false;
  bool spectral_degenerate = false;
  bool final_degenerate = false;

  double raw_min_eigen = 0.0;
  double raw_max_eigen = 0.0;
  double raw_eigen_ratio = 0.0;
  double raw_condition_number = 0.0;

  Eigen::VectorXd eigenvalues;
  Eigen::MatrixXd eigenvectors;

  double min_eigen = 0.0;
  double max_eigen = 0.0;
  double eigen_ratio = 0.0;
  double condition_number = 0.0;

  int weak_dim = 0;
  double remap_gamma = 0.1;

  double residual_norm = 0.0;
  double residual_rms = 0.0;

  double weak_energy_before = -1.0;
  double weak_energy_after = -1.0;
  double weak_attenuation_ratio = -1.0;

  double strong_energy_before = -1.0;
  double strong_energy_after = -1.0;
  double strong_preserve_ratio = -1.0;

  Eigen::VectorXd weak_eigenvalues;

  Eigen::Matrix<double, 6, 6> pose_remap_matrix =
      Eigen::Matrix<double, 6, 6>::Identity();
};

class LidarDegeneracyAnalyzer
{
public:
  using SO3d = Sophus::SO3<double>;
  using Ptr = std::shared_ptr<LidarDegeneracyAnalyzer>;
  using PoseRemapMatrix = Eigen::Matrix<double, 6, 6>;
  using PoseRemapMatrixVector =
      std::vector<PoseRemapMatrix, Eigen::aligned_allocator<PoseRemapMatrix>>;
  using BlockReportVector =
      std::vector<LidarDegeneracyBlockReport,
                  Eigen::aligned_allocator<LidarDegeneracyBlockReport>>;

  explicit LidarDegeneracyAnalyzer(const LidarDegeneracyParam &param);

  const LidarDegeneracyParam &param() const { return param_; }

  LidarDegeneracyReport Analyze(
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
      bool use_lidar_scale);

  // These vectors are aligned with the original point_corrs index.
  // A true flag means that this residual belongs to a degenerate time block
  // and should use the corresponding pose_remap_matrix in stage 3.
  const std::vector<bool> &last_use_remap_flags() const
  {
    return last_use_remap_flags_;
  }

  const PoseRemapMatrixVector &last_pose_remap_matrices() const
  {
    return last_pose_remap_matrices_;
  }

  const BlockReportVector &last_block_reports() const
  {
    return last_block_reports_;
  }

private:
  struct PoseRowData
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int point_index = -1;
    Eigen::Matrix<double, 1, 6> h_pose = Eigen::Matrix<double, 1, 6>::Zero();
    double residual = 0.0;
    GeometryType geo_type = GeometryType::Plane;
    int64_t t_point = 0;
  };

  struct PoseBlockWork
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int block_idx = 0;
    int64_t block_start_t_ns = 0;
    int64_t block_end_t_ns = 0;
    int64_t min_t_ns = 0;
    int64_t max_t_ns = 0;
    bool has_time = false;

    Eigen::Matrix<double, 6, 6> A_pose = Eigen::Matrix<double, 6, 6>::Zero();
    std::vector<PoseRowData, Eigen::aligned_allocator<PoseRowData>> rows;

    int used_corr_num = 0;
    int plane_num = 0;
    int line_num = 0;
    double residual_norm2 = 0.0;
  };

private:
  int QDim() const;

  int GetTimeBlockIndex(int64_t t_point,
                        int64_t opt_min_t_ns,
                        int64_t opt_max_t_ns) const;

  bool BuildPoseJacobian(
      const PointCorrespondence &pc,
      const Trajectory::Ptr &trajectory,
      const SO3d &S_GtoM,
      const Eigen::Vector3d &p_GinM,
      const SO3d &S_LtoI,
      const Eigen::Vector3d &p_LinI,
      double weight,
      Eigen::Matrix<double, 1, 6> *h_pose,
      double *residual_out) const;

  bool AnalyzePoseMatrix(
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
      double *strong_preserve_ratio) const;

  void ApplyWindowVerification(LidarDegeneracyReport *report) const;
  void WriteWindowLog(const LidarDegeneracyReport &report) const;
  void WriteBlockLog(const LidarDegeneracyBlockReport &block) const;
  void EnsureLogDir() const;

private:
  LidarDegeneracyParam param_;
  int64_t frame_idx_ = 0;

  std::vector<bool> last_use_remap_flags_;
  PoseRemapMatrixVector last_pose_remap_matrices_;
  BlockReportVector last_block_reports_;
};

} // namespace cocolic
