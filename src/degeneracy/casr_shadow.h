/*
 * Cause-adaptive subspace recovery (CASR), shadow-only stage.
 *
 * CASR-v2 compares the environment and spline-support weak directions in the
 * active 6K control-point space. The evaluator constructs candidate knot-space
 * recovery bases, but never applies them to the estimator. Estimator
 * integration is deliberately kept outside this stage until the bases and the
 * temporal gate have been validated from logs.
 */

#pragma once

#include <degeneracy/casr_route.h>

#include <Eigen/Core>

namespace cocolic
{

  inline constexpr char kCasrShadowMethodVersion[] =
      "knot_space_v2_overlap";

  using CasrVector6 = Eigen::Matrix<double, 6, 1>;
  using CasrMatrix6 = Eigen::Matrix<double, 6, 6>;

  struct CasrShadowConfig
  {
    bool enabled = false;
    double environment_relative_threshold = 6e-3;
    double support_basis_relative_singular_threshold = 1e-6;
    double lift_regularization = 1e-6;
    double principal_cosine_threshold = 7e-1;
    int route_consecutive_scans = 3;
    int projector_consecutive_scans = 3;
    double projector_similarity_threshold = 8e-1;
  };

  struct CasrShadowInput
  {
    bool environment_valid = false;
    bool support_valid = false;
    bool environment_state = false;
    bool support_state = false;
    CasrVector6 environment_relative_eigenvalues = CasrVector6::Zero();
    CasrMatrix6 environment_eigenvectors = CasrMatrix6::Identity();

    // All matrices below use the same active knot ordering:
    // [r*dtheta_0, dp_0, ..., r*dtheta_(K-1), dp_(K-1)]. The support basis is
    // Euclidean-orthonormal. The scan-wide environment lift solves
    //   min_x mean_j ||B_j x - e||^2 + lambda ||x||^2,
    // where B_j maps scaled knot perturbations to the scaled LiDAR pose state
    // [r*dtheta_map, dp_map] at a reference timestamp.
    int support_control_point_start_index = -1;
    const Eigen::MatrixXd *support_knot_weak_basis = nullptr;
    const Eigen::MatrixXd *reference_pose_information = nullptr;
    const Eigen::MatrixXd *reference_pose_cross = nullptr;
    const Eigen::MatrixXd *representative_pose_mapping = nullptr;
  };

  struct CasrShadowResult
  {
    bool valid = false;
    int cause = -1;
    CasrRoute route = CasrRoute::Invalid;
    int environment_rank = 0;
    int support_rank = 0;
    int common_rank = 0;
    int recovery_rank = 0;
    int representative_pose_rank = 0;
    int support_control_point_start_index = -1;
    int support_knot_dimension = 0;
    double overlap_score = 0.0;
    double principal_cosine_min = 0.0;
    double principal_cosine_max = 0.0;
    double environment_exclusive_ratio = 0.0;
    double support_exclusive_ratio = 0.0;
    double environment_lift_residual = 0.0;
    double recovery_basis_orthogonality_error = 0.0;

    // Raw cause routing remains visible. stable_route and recovery_ready are
    // an independent shadow-only temporal gate; they are never consumed by
    // Ceres or any estimator state in this stage.
    CasrRoute stable_route = CasrRoute::Invalid;
    int route_candidate_count = 0;
    double temporal_projector_similarity = 0.0;
    int temporal_overlap_control_point_num = 0;
    int projector_consistency_count = 0;
    bool recovery_ready = false;

    // The six-dimensional matrices are representative-time summaries for CSV
    // inspection only. Routing and recovery_rank are computed exclusively in
    // the 6K knot space.
    CasrVector6 support_pose_eigenvalues = CasrVector6::Zero();
    CasrMatrix6 environment_projector = CasrMatrix6::Zero();
    CasrMatrix6 support_projector = CasrMatrix6::Zero();
    CasrMatrix6 common_projector = CasrMatrix6::Zero();
    CasrMatrix6 recovery_projector = CasrMatrix6::Zero();

    Eigen::MatrixXd environment_knot_basis;
    Eigen::MatrixXd support_knot_basis;
    Eigen::MatrixXd common_knot_basis;
    Eigen::MatrixXd recovery_knot_basis;
  };

  struct CasrTemporalState
  {
    bool initialized = false;
    CasrRoute stable_route = CasrRoute::Invalid;
    CasrRoute pending_route = CasrRoute::Invalid;
    int pending_route_count = 0;
    int projector_consistency_count = 0;
    int previous_control_point_start_index = -1;
    Eigen::MatrixXd previous_recovery_basis;
  };

  class CasrShadowEvaluator
  {
  public:
    void Configure(const CasrShadowConfig &config);

    bool Enabled() const { return config_.enabled; }

    const CasrShadowConfig &Config() const { return config_; }

    CasrShadowResult Evaluate(const CasrShadowInput &input,
                              CasrTemporalState *temporal_state) const;

  private:
    CasrShadowConfig config_;
  };

} // namespace cocolic
