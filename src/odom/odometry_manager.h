/*
 * Coco-LIC: Continuous-Time Tightly-Coupled LiDAR-Inertial-Camera Odometry using Non-Uniform B-spline
 * Copyright (C) 2023 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include <ros/ros.h>

#include <odom/msg_manager.h>
#include <odom/odometry_viewer.h>
#include <odom/trajectory_manager.h>

#include <imu/imu_state_estimator.h>
#include <imu/imu_initializer.h>
#include <lidar/lidar_handler.h>
#include <degeneracy/ct_lidar_observability.h>

#include <array>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <mutex>
#include <thread>

#include <camera/r3live.hpp>

namespace cocolic
{

  struct QiConfig
  {
    bool quality_enable = false;
    bool lidar_quality_enable = true;
    bool visual_quality_enable = true;
    bool selection_enable = false;

    double lidar_q_min = 0.5;
    double lidar_q_max = 1.2;
    double visual_q_min = 0.7;
    double visual_q_max = 1.0;
    double visual_point_quality_weight = 0.25;

    int max_lidar_obs = 800;
    int max_visual_obs = 200;
    double selection_d_efficiency = 0.95;
    double selection_min_gain = 1.0e-6;
    double info_prior_eps = 1.0e-6;
    bool output_csv = true;
    bool log_enable = false;
  };

  struct QiLidarObs
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PointCorrespondence correspondence;
    double q = 1.0;
    double residual_pre = 0.0;
    double residual_post = 0.0;
    double base_weight = 1.0;
    double final_weight = 1.0;
    bool selected = false;
    double info_gain = 0.0;
    Eigen::Matrix<double, 6, 6> info =
        Eigen::Matrix<double, 6, 6>::Zero();
  };

  struct QiVisualObs
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int point_id = -1;
    RGB_pts *point_ptr = nullptr;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
    double q = 1.0;
    double residual_pre = 0.0;
    double residual_post = 0.0;
    double frame_quality = 1.0;
    double point_quality = 1.0;
    double base_weight = 1.0;
    double final_weight = 1.0;
    bool selected = false;
    double info_gain = 0.0;
    Eigen::Matrix<double, 6, 6> info =
        Eigen::Matrix<double, 6, 6>::Zero();
  };

  struct QiResidualScale
  {
    double sigma = 1.0;
    bool ready = false;
  };

  enum KnotDensity
  {
    gear1 = 1, // 0.1
    gear2 = 2, // 0.05
    gear3 = 3, // 0.033
    gear4 = 4  // 0.025
  };
  class OdometryManager
  {
  public:
    OdometryManager(const YAML::Node &node, ros::NodeHandle &nh);

    void RunBag();

    void RunInSubscribeMode();

    double SaveOdometry();

    std::vector<int> cp_num_vec;

    int GetKnotDensity(double gyro_norm, double acce_norm)
    {
      if (gyro_norm < 0.0 || acce_norm < 0.0)
      {
        std::cout << RED << "gyro_norm/acce_norm is wrong!" << RESET << std::endl;
      }

      int gyro_density = -1, acce_density = -1;

      acce_norm = std::abs(acce_norm - gravity_norm_);
      // LOG(INFO) << "[acce_norm] " << acce_norm;
      if (acce_norm < 0.5)
      { // [0, 0.5)
        acce_density = KnotDensity::gear1;
      }
      else if (acce_norm < 1.0)
      { // [0.5, 1.0)
        acce_density = KnotDensity::gear2;
      }
      else if (acce_norm < 5.0)
      { // [1.0, 5.0)
        acce_density = KnotDensity::gear3;
      }
      else
      { // [5.0, -)
        acce_density = KnotDensity::gear4;
      }

      // LOG(INFO) << "[gyro_norm] " << gyro_norm;
      if (gyro_norm < 0.5)
      { // [0, 0.5)
        gyro_density = KnotDensity::gear1;
      }
      else if (gyro_norm < 1.0)
      { // [0.5, 1.0)
        gyro_density = KnotDensity::gear2;
      }
      else if (gyro_norm < 5.0)
      { // [1.0, 5.0)
        gyro_density = KnotDensity::gear3;
      }
      else
      { // [5.0, -)
        gyro_density = KnotDensity::gear4;
      }

      return std::max(gyro_density, acce_density);
    };

  protected:
    bool CreateCacheFolder(const std::string &config_path,
                           const std::string &bag_path);

    void SolveLICO();

    void ProcessLICData();

    void ProcessImageData();

    bool PrepareTwoSegMsgs(int seg_idx);

    void UpdateTwoSeg();

    bool PrepareMsgs();

    void UpdateOneSeg();

    void SetInitialState();

    void PublishCloudAndTrajectory();

    void Publish3DGSMappingData(const NextMsgs& cur_msg);

    void BuildVisualObs();
    void PrepareQiVisualObs(int64_t image_timestamp);
    void PrepareQiLidarObs(
        const Eigen::aligned_vector<PointCorrespondence> &point_corrs);
    void SelectQiLidarObs();
    void SelectQiVisualObs();
    void UpdateQiResidualStatistics(int64_t image_timestamp,
                                    bool process_image,
                                    bool optimization_success);
    void WriteQiCsv(int64_t scan_timestamp,
                    int64_t image_timestamp,
                    bool process_image,
                    bool optimization_success);
    void LogQiSummary() const;

    double ComputeLidarResidual(const PointCorrespondence &corr,
                                const SE3d &T_lidar) const;
    double ComputeVisualResidual(const QiVisualObs &obs,
                                 const SE3d &T_cam) const;
    Eigen::Matrix<double, 6, 6> ComputeLidarInfo(
        const PointCorrespondence &corr,
        const SE3d &T_lidar,
        double final_weight) const;
    Eigen::Matrix<double, 6, 6> ComputeVisualInfo(
        const QiVisualObs &obs,
        const SE3d &T_cam,
        double final_weight) const;
    double QiClamp(double value, double low, double high) const;
    double QiMedian(std::vector<double> values) const;
    QiResidualScale QiMadScale(const std::deque<double> &values) const;
    double QiLogDet(const Eigen::Matrix<double, 6, 6> &mat) const;
    double QiConditionNumber(
        const Eigen::Matrix<double, 6, 6> &mat) const;
    double QiDEfficiency(
        const Eigen::Matrix<double, 6, 6> &selected,
        const Eigen::Matrix<double, 6, 6> &full) const;
    double QiMinDirectionRetention(
        const Eigen::Matrix<double, 6, 6> &selected,
        const Eigen::Matrix<double, 6, 6> &full) const;

  protected:
    OdometryMode odometry_mode_;

    MsgManager::Ptr msg_manager_;

    bool is_initialized_;
    IMUInitializer::Ptr imu_initializer_;

    Trajectory::Ptr trajectory_;
    TrajectoryManager::Ptr trajectory_manager_;

    LidarHandler::Ptr lidar_handler_;

    R3LIVE::Ptr camera_handler_;

    CtLidarObservability::Ptr ct_lidar_observability_;

    int64_t t_begin_add_cam_; // 

    OdometryViewer odom_viewer_;

    int update_every_k_knot_;

    /// [nurbs]
    double t_add_;
    int64_t t_add_ns_;
    bool non_uniform_;
    double distance0_;

    int cp_add_num_coarse_;
    int cp_add_num_refine_;

    int lidar_iter_;
    bool use_lidar_scale_;

    std::string cache_path_;

    double pasue_time_;

    TimeStatistics time_summary_;

    struct SysTimeOffset
    {
      SysTimeOffset(double t1, double t2, double t3, double t4)
          : timestamp(t1), t_lidar(t2), t_cam(t3), t_imu(t4) {}
      double timestamp = 0;
      double t_lidar = 0;
      double t_cam = 0;
      double t_imu = 0;
    };
    std::vector<SysTimeOffset> sys_t_offset_vec_;

  private:
    double gravity_norm_;

    int64_t traj_max_time_ns_cur;
    int64_t traj_max_time_ns_next;
    int64_t traj_max_time_ns_next_next;

    int cp_add_num_cur;
    int cp_add_num_next;
    int cp_add_num_next_next;

    bool is_evo_viral_;

    double ave_r_thresh_;
    double ave_a_thresh_;

    VPointCloud sub_map_cur_frame_point_;

    Eigen::aligned_vector<Eigen::Vector3d> v_points_;
    Eigen::aligned_vector<Eigen::Vector2d> px_obss_;

    Eigen::Matrix3d K_;

    bool if_3dgs_;
    int lidar_skip_;

    std::queue<int64_t> time_buf;  // img timestamp
    std::queue<LiDARFeature> lidar_buf;  // lidarfeature in local
    std::queue<cv::Mat> img_buf;  // undistorted
    std::vector<PosCloud::Ptr> lidarpoints;

    QiConfig qi_config_;
    double qi_lidar_weight_ = 1.0;
    double qi_image_weight_ = 1.0;
    double qi_characteristic_length_ = 1.0;
    std::ofstream qi_csv_;

    Eigen::aligned_vector<QiLidarObs> qi_lidar_obs_;
    Eigen::aligned_vector<QiLidarObs> qi_selected_lidar_obs_;
    Eigen::aligned_vector<QiVisualObs> qi_visual_obs_;
    Eigen::aligned_vector<QiVisualObs> qi_selected_visual_obs_;

    Eigen::aligned_vector<PointCorrespondence> qi_selected_point_corrs_;
    Eigen::aligned_vector<Eigen::Vector3d> qi_selected_visual_points_;
    Eigen::aligned_vector<Eigen::Vector2d> qi_selected_visual_pixels_;
    std::vector<double> qi_selected_lidar_weights_;
    std::vector<double> qi_selected_visual_weights_;

    std::deque<double> qi_recent_lidar_residuals_;
    std::deque<double> qi_recent_visual_residuals_;
    std::deque<double> qi_recent_n_pnp_;
    QiResidualScale qi_lidar_scale_;
    QiResidualScale qi_visual_scale_;
    double qi_n0_ = 1.0;
    int64_t qi_last_visual_history_timestamp_ = -1;

    double qi_last_lidar_pre_median_ = 0.0;
    double qi_last_lidar_pre_mad_ = 0.0;
    double qi_last_lidar_post_median_ = 0.0;
    double qi_last_lidar_post_mad_ = 0.0;
    double qi_last_visual_pre_median_ = 0.0;
    double qi_last_visual_pre_mad_ = 0.0;
    double qi_last_visual_post_median_ = 0.0;
    double qi_last_visual_post_mad_ = 0.0;

    double qi_last_lidar_logdet_ = 0.0;
    double qi_last_visual_logdet_ = 0.0;
    double qi_last_lidar_gain_mean_ = 0.0;
    double qi_last_visual_gain_mean_ = 0.0;
    double qi_last_lidar_cond_ = 0.0;
    double qi_last_visual_cond_ = 0.0;
    double qi_last_lidar_d_efficiency_ = 1.0;
    double qi_last_visual_d_efficiency_ = 1.0;
    double qi_last_lidar_min_direction_retention_ = 1.0;
    double qi_last_visual_min_direction_retention_ = 1.0;

    double qi_last_lidar_q_min_ = 1.0;
    double qi_last_lidar_q_mean_ = 1.0;
    double qi_last_lidar_q_max_ = 1.0;
    double qi_last_lidar_scale_min_ = 1.0;
    double qi_last_lidar_scale_mean_ = 1.0;
    double qi_last_lidar_scale_max_ = 1.0;
    double qi_last_lidar_base_weight_min_ = 1.0;
    double qi_last_lidar_base_weight_mean_ = 1.0;
    double qi_last_lidar_base_weight_max_ = 1.0;
    double qi_last_lidar_weight_ratio_min_ = 1.0;
    double qi_last_lidar_weight_ratio_mean_ = 1.0;
    double qi_last_lidar_weight_ratio_max_ = 1.0;
    double qi_last_visual_q_min_ = 1.0;
    double qi_last_visual_q_mean_ = 1.0;
    double qi_last_visual_q_max_ = 1.0;
    double qi_last_visual_weight_ratio_min_ = 1.0;
    double qi_last_visual_weight_ratio_mean_ = 1.0;
    double qi_last_visual_weight_ratio_max_ = 1.0;

    int qi_last_lidar_candidates_ = 0;
    int qi_last_lidar_selected_ = 0;
    int qi_last_visual_candidates_ = 0;
    int qi_last_visual_selected_ = 0;
    int qi_last_n_pnp_ = 0;
    double qi_last_eta_pnp_ = 1.0;
    double qi_last_eta_fmat_ = 1.0;
  };

} // namespace cocolic
