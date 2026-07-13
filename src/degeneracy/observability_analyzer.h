/*
 * Coco-LIC: Continuous-Time Tightly-Coupled LiDAR-Inertial-Camera Odometry using Non-Uniform B-spline
 *
 * Detector-only extension: this file observes LiDAR geometry but never changes
 * the estimator, map, marginalization prior, or trajectory control points.
 */

#pragma once

#include <degeneracy/degeneracy_hysteresis.h>

#include <Eigen/Core>
#include <lidar/lidar_feature.h>
#include <spline/trajectory.h>
#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cocolic
{

  struct ObservabilityResult
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool valid = false;
    int64_t scan_timestamp_ns = 0;
    size_t correspondence_num = 0;
    size_t plane_num = 0;
    size_t line_num = 0;
    double characteristic_range = 1.0;
    double condition_number = 0.0;
    // Legacy hard-threshold count retained as a raw diagnostic.
    int weak_direction_num = 6;
    // Candidate count and temporal state use the hysteresis enter threshold.
    int candidate_weak_direction_num = 0;
    double degeneracy_score = 0.0;
    bool degenerate_state = false;
    int enter_counter = 0;
    int exit_counter = 0;

    // Exact non-uniform B-spline support diagnostic. The support matrix is
    // built from the same four-knot SO(3) analytic Jacobians and R3 basis
    // coefficients used by LoamFeatureFactorNURBS, but without any geometry
    // residual. Its generalized eigenvalues compare the timestamp support in
    // the current scan with ideal uniform sampling of the same active knot
    // intervals.
    bool support_valid = false;
    int support_control_point_num = 0;
    int support_interval_num = 0;
    int support_dimension = 0;
    int support_effective_rank = 0;
    int support_weak_direction_num = 0;
    double support_time_span_s = 0.0;
    double support_min_knot_dt_s = 0.0;
    double support_max_knot_dt_s = 0.0;
    double support_quality_min = 0.0;
    double support_condition_number = 0.0;
    double support_score = 0.0;
    bool support_degenerate_state = false;
    int support_enter_counter = 0;
    int support_exit_counter = 0;
    int support_weakest_knot_index = -1;
    double support_weakest_knot_energy_ratio = 0.0;
    double support_weakest_rotation_ratio = 0.0;
    double support_boundary_energy_ratio = 0.0;

    // -1 invalid, 0 healthy, 1 environment geometry, 2 spline support,
    // 3 coupled environment-and-support degeneracy. This is diagnostic only.
    int degeneracy_cause = -1;

    // Ascending order. The state order is [rotation, translation] in the map
    // frame. Rotation columns are normalized by characteristic_range before
    // decomposition.
    Eigen::Matrix<double, 6, 1> eigenvalues =
        Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 1> relative_eigenvalues =
        Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 6> eigenvectors =
        Eigen::Matrix<double, 6, 6>::Identity();
  };

  class ObservabilityAnalyzer
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<ObservabilityAnalyzer> Ptr;

    ObservabilityAnalyzer(const YAML::Node &node,
                          Trajectory::Ptr trajectory,
                          const std::string &output_prefix);

    ~ObservabilityAnalyzer();

    bool Enabled() const { return enabled_; }

    ObservabilityResult AnalyzeAndLog(
        int64_t scan_timestamp_ns,
        const Eigen::aligned_vector<PointCorrespondence> &point_corrs);

    const ObservabilityResult &LastResult() const { return last_result_; }

  private:
    ObservabilityResult Analyze(
        int64_t scan_timestamp_ns,
        const Eigen::aligned_vector<PointCorrespondence> &point_corrs) const;

    double ComputeCharacteristicRange(
        const Eigen::aligned_vector<PointCorrespondence> &point_corrs) const;

    bool BuildPoseJacobian(const PointCorrespondence &pc,
                           double characteristic_range,
                           Eigen::Matrix<double, 1, 6> &jacobian) const;

    void AnalyzeSplineSupport(
        const std::vector<std::pair<int64_t, double>> &weighted_timestamps,
        ObservabilityResult &result) const;

    void WriteCsvHeader();
    void WriteCsvRow(const ObservabilityResult &result);
    void PrintSummary(const ObservabilityResult &result) const;

  private:
    Trajectory::Ptr trajectory_;

    bool enabled_ = false;
    bool output_csv_ = true;
    bool use_correspondence_scale_ = true;
    int min_correspondences_ = 30;
    int analyze_every_n_scans_ = 1;
    int print_every_n_scans_ = 20;
    double relative_eigenvalue_threshold_ = 1e-3;
    double enter_relative_eigenvalue_threshold_ = 3e-3;
    double exit_relative_eigenvalue_threshold_ = 6e-3;
    int enter_consecutive_scans_ = 10;
    int exit_consecutive_scans_ = 10;
    double min_characteristic_range_ = 1.0;
    double max_characteristic_range_ = 100.0;

    bool support_enabled_ = true;
    int support_reference_samples_per_interval_ = 32;
    int support_max_control_points_ = 32;
    double support_enter_quality_threshold_ = 2e-2;
    double support_exit_quality_threshold_ = 5e-2;
    int support_enter_consecutive_scans_ = 10;
    int support_exit_consecutive_scans_ = 10;

    size_t scan_counter_ = 0;
    std::string csv_path_;
    std::ofstream csv_stream_;
    DegeneracyHysteresis degeneracy_hysteresis_;
    DegeneracyHysteresis support_hysteresis_;
    ObservabilityResult last_result_;
  };

} // namespace cocolic
