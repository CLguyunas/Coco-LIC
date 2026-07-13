/*
 * Coco-LIC: Continuous-Time Tightly-Coupled LiDAR-Inertial-Camera Odometry using Non-Uniform B-spline
 *
 * Detector-only extension: this file observes LiDAR geometry but never changes
 * the estimator, map, marginalization prior, or trajectory control points.
 */

#pragma once

#include <Eigen/Core>
#include <lidar/lidar_feature.h>
#include <spline/trajectory.h>
#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>

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
    int weak_direction_num = 6;

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
    double min_characteristic_range_ = 1.0;
    double max_characteristic_range_ = 100.0;

    size_t scan_counter_ = 0;
    std::string csv_path_;
    std::ofstream csv_stream_;
    ObservabilityResult last_result_;
  };

} // namespace cocolic
