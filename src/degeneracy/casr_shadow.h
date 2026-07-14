/*
 * Cause-adaptive subspace recovery (CASR), shadow-only stage.
 *
 * The evaluator constructs candidate 6DoF recovery projectors but never
 * applies them to the estimator. Estimator integration is deliberately kept
 * outside this stage until the projectors have been validated from logs.
 */

#pragma once

#include <degeneracy/casr_route.h>

#include <Eigen/Core>

namespace cocolic
{

  using CasrVector6 = Eigen::Matrix<double, 6, 1>;
  using CasrMatrix6 = Eigen::Matrix<double, 6, 6>;

  struct CasrShadowConfig
  {
    bool enabled = false;
    double environment_relative_threshold = 6e-3;
    double support_pose_relative_threshold = 1e-1;
    double principal_cosine_threshold = 7e-1;
  };

  struct CasrShadowInput
  {
    bool environment_valid = false;
    bool support_valid = false;
    bool environment_state = false;
    bool support_state = false;
    CasrVector6 environment_relative_eigenvalues = CasrVector6::Zero();
    CasrMatrix6 environment_eigenvectors = CasrMatrix6::Identity();
    CasrMatrix6 support_pose_weakness = CasrMatrix6::Zero();
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
    double overlap_score = 0.0;
    double principal_cosine_min = 0.0;
    double principal_cosine_max = 0.0;
    double environment_exclusive_ratio = 0.0;
    double support_exclusive_ratio = 0.0;
    CasrVector6 support_pose_eigenvalues = CasrVector6::Zero();
    CasrMatrix6 environment_projector = CasrMatrix6::Zero();
    CasrMatrix6 support_projector = CasrMatrix6::Zero();
    CasrMatrix6 common_projector = CasrMatrix6::Zero();
    CasrMatrix6 recovery_projector = CasrMatrix6::Zero();
  };

  class CasrShadowEvaluator
  {
  public:
    void Configure(const CasrShadowConfig &config);

    bool Enabled() const { return config_.enabled; }

    const CasrShadowConfig &Config() const { return config_; }

    CasrShadowResult Evaluate(const CasrShadowInput &input) const;

  private:
    CasrShadowConfig config_;
  };

} // namespace cocolic
