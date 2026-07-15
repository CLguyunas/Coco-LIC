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

#include <eigen_conversions/eigen_msg.h>
#include <odom/odometry_manager.h>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <limits>

#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>

std::fstream rgb_file;
std::fstream img_file;

namespace cocolic
{

  OdometryManager::OdometryManager(const YAML::Node &node, ros::NodeHandle &nh)
      : odometry_mode_(LIO), is_initialized_(false)
  {
    std::string config_path;
    nh.param<std::string>("project_path", config_path, "");
    config_path += "/config";

    std::string lidar_yaml = node["lidar_yaml"].as<std::string>();
    YAML::Node lidar_node = YAML::LoadFile(config_path + lidar_yaml);

    std::string imu_yaml = node["imu_yaml"].as<std::string>();
    YAML::Node imu_node = YAML::LoadFile(config_path + imu_yaml);

    std::string cam_yaml = config_path + node["camera_yaml"].as<std::string>();
    YAML::Node cam_node = YAML::LoadFile(cam_yaml);

    odometry_mode_ = OdometryMode(node["odometry_mode"].as<int>());
    std::cout << "\n🥥 Odometry Mode: ";
    if (odometry_mode_ == LICO)
    {
      std::cout << "LiDAR-Inertial-Camera Odometry 🥥" << std::endl;
    }
    else if (odometry_mode_ == LIO)
    {
      std::cout << "LiDAR-Inertial Odometry 🥥" << std::endl;
    }

    // extrinsic: sensor to imu
    ExtrinsicParam EP_LtoI, EP_CtoI, EP_ItoI, EP_MtoI;
    EP_LtoI.Init(lidar_node["lidar0"]["Extrinsics"]);
    if (odometry_mode_ == LICO)
      EP_CtoI.Init(cam_node["CameraExtrinsics"]);
    if (node["IMUExtrinsics"])
      EP_ItoI.Init(imu_node["IMUExtrinsics"]);
    EP_MtoI.Init(imu_node["MarkerExtrinsics"]);

    trajectory_ = std::make_shared<Trajectory>(-1, 0);
    trajectory_->SetSensorExtrinsics(SensorType::LiDARSensor, EP_LtoI);
    trajectory_->SetSensorExtrinsics(SensorType::CameraSensor, EP_CtoI);
    trajectory_->SetSensorExtrinsics(SensorType::IMUSensor, EP_ItoI);
    trajectory_->SetSensorExtrinsics(SensorType::Marker, EP_MtoI);

    // non-uniform b-spline
    t_add_ = node["t_add"].as<double>();
    t_add_ns_ = t_add_ * S_TO_NS;
    non_uniform_ = node["non_uniform"].as<bool>();
    distance0_ = node["distance0"].as<double>();

    // lidar
    lidar_iter_ = node["lidar_iter"].as<int>();
    use_lidar_scale_ = node["use_lidar_scale"].as<bool>();
    lidar_handler_ = std::make_shared<LidarHandler>(lidar_node, trajectory_);
    std::cout << "\n🍺 The number of multiple LiDARs is " << lidar_node["num_lidars"].as<int>() << "." << std::endl;

    // imu
    imu_initializer_ = std::make_shared<IMUInitializer>(imu_node);
    gravity_norm_ = imu_initializer_->GetGravity().norm();

    // camera
    camera_handler_ = std::make_shared<R3LIVE>(cam_node, EP_CtoI);
    t_begin_add_cam_ = node["t_begin_add_cam"].as<double>() * S_TO_NS;
    v_points_.clear();
    px_obss_.clear();
    double fx = cam_node["cam_fx"].as<double>();
    double fy = cam_node["cam_fy"].as<double>();
    double cx = cam_node["cam_cx"].as<double>();
    double cy = cam_node["cam_cy"].as<double>();
    K_ << fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0;

    // trajectory parameterized by b-spline
    trajectory_manager_ = std::make_shared<TrajectoryManager>(node, config_path, trajectory_);
    trajectory_manager_->use_lidar_scale = use_lidar_scale_;
    trajectory_manager_->SetIntrinsic(K_);

    int division_coarse = node["division_coarse"].as<int>();
    cp_add_num_coarse_ = division_coarse;
    trajectory_manager_->SetDivisionParam(division_coarse, -1);

    odom_viewer_.SetPublisher(nh);

    msg_manager_ = std::make_shared<MsgManager>(node, config_path, nh);  // load rosbag

    bool verbose;
    nh.param<double>("pasue_time", pasue_time_, -1);
    nh.param<bool>("verbose", verbose, false);
    trajectory_manager_->verbose = verbose;

    // evaluation
    is_evo_viral_ = node["is_evo_viral"].as<bool>();
    CreateCacheFolder(config_path, msg_manager_->bag_path_);

    // gaussian-lic
    if_3dgs_ = node["if_3dgs"].as<bool>();
    lidar_skip_ = node["lidar_skip"].as<int>();

    // QI-LIC observation management. The quality and selection switches
    // default to false, preserving the original Coco-LIC path for old configs.
    qi_config_.quality_enable = yaml::GetValue<bool>(
        node, "qi_quality_enable",
        yaml::GetValue<bool>(node, "qim_quality_enable", false));
    qi_config_.lidar_quality_enable = yaml::GetValue<bool>(
        node, "qi_lidar_quality_enable",
        yaml::GetValue<bool>(node, "qim_lidar_quality_enable", true));
    qi_config_.visual_quality_enable = yaml::GetValue<bool>(
        node, "qi_visual_quality_enable",
        yaml::GetValue<bool>(node, "qim_visual_quality_enable", true));
    qi_config_.selection_enable = yaml::GetValue<bool>(
        node, "qi_selection_enable",
        yaml::GetValue<bool>(node, "qim_selection_enable", false));

    qi_config_.lidar_q_min = yaml::GetValue<double>(
        node, "qi_lidar_q_min",
        yaml::GetValue<double>(node, "qim_lidar_q_min", 1.0));
    qi_config_.lidar_q_max = yaml::GetValue<double>(
        node, "qi_lidar_q_max",
        yaml::GetValue<double>(node, "qim_lidar_q_max", 1.2));
    qi_config_.visual_q_min = yaml::GetValue<double>(
        node, "qi_visual_q_min",
        yaml::GetValue<double>(node, "qim_visual_q_min", 0.7));
    qi_config_.visual_q_max = yaml::GetValue<double>(
        node, "qi_visual_q_max",
        yaml::GetValue<double>(node, "qim_visual_q_max", 1.0));
    qi_config_.visual_point_quality_weight = yaml::GetValue<double>(
        node, "qi_visual_point_quality_weight", 0.25);

    qi_config_.max_lidar_obs = yaml::GetValue<int>(
        node, "qi_max_lidar_obs",
        yaml::GetValue<int>(node, "qim_max_lidar_obs", 800));
    qi_config_.max_visual_obs = yaml::GetValue<int>(
        node, "qi_max_visual_obs",
        yaml::GetValue<int>(node, "qim_max_visual_obs", 200));
    qi_config_.selection_info_ratio = yaml::GetValue<double>(
        node, "qi_selection_info_ratio", 0.95);
    qi_config_.selection_min_gain = yaml::GetValue<double>(
        node, "qi_selection_min_gain", 1.0e-6);
    qi_config_.info_prior_eps = yaml::GetValue<double>(
        node, "qi_info_prior_eps",
        yaml::GetValue<double>(node, "qim_info_prior_eps", 1.0e-6));
    qi_config_.log_enable = yaml::GetValue<bool>(
        node, "qi_log_enable",
        yaml::GetValue<bool>(node, "qim_log_enable", false));

    qi_lidar_weight_ = yaml::GetValue<double>(node, "lidar_weight", 1.0);
    qi_image_weight_ = yaml::GetValue<double>(node, "image_weight", 1.0);

    qi_config_.lidar_q_min = std::max(qi_config_.lidar_q_min, 1.0);
    qi_config_.lidar_q_max = std::max(qi_config_.lidar_q_max,
                                      qi_config_.lidar_q_min);
    qi_config_.visual_q_min = std::max(qi_config_.visual_q_min, 1.0e-3);
    qi_config_.visual_q_max = std::max(qi_config_.visual_q_max,
                                       qi_config_.visual_q_min);
    qi_config_.visual_point_quality_weight = QiClamp(
        qi_config_.visual_point_quality_weight, 0.0, 1.0);
    qi_config_.selection_info_ratio = QiClamp(
        qi_config_.selection_info_ratio, 0.0, 1.0);
    qi_config_.selection_min_gain = std::max(
        qi_config_.selection_min_gain, 0.0);
    qi_config_.info_prior_eps = std::max(qi_config_.info_prior_eps, 1.0e-12);
    qi_config_.max_lidar_obs = std::max(qi_config_.max_lidar_obs, 1);
    qi_config_.max_visual_obs = std::max(qi_config_.max_visual_obs, 1);

    lidarpoints.clear();

    std::cout << std::fixed << std::setprecision(4);
    // LOG(INFO) << std::fixed << std::setprecision(4);
  }

  bool OdometryManager::CreateCacheFolder(const std::string &config_path,
                                          const std::string &bag_path)
  {
    boost::filesystem::path path_cfg(config_path);
    boost::filesystem::path path_bag(bag_path);
    if (path_bag.extension() != ".bag")
    {
      return false;
    }
    std::string bag_name_ = path_bag.stem().string();

    std::string cache_path_parent_ = path_cfg.parent_path().string();
    cache_path_ = cache_path_parent_ + "/data/" + bag_name_;
    // boost::filesystem::create_directory(cache_path_);
    return true;
  }


  double OdometryManager::QiClamp(double value, double low, double high) const
  {
    return std::max(low, std::min(high, value));
  }

  double OdometryManager::QiMedian(std::vector<double> values) const
  {
    if (values.empty())
      return 0.0;
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    double med = values[mid];
    if (values.size() % 2 == 0)
    {
      std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
      med = 0.5 * (med + values[mid - 1]);
    }
    return med;
  }

  QiResidualScale OdometryManager::QiMadScale(const std::deque<double> &values) const
  {
    QiResidualScale scale;
    if (values.size() < 5)
      return scale;
    std::vector<double> vals(values.begin(), values.end());
    const double med = QiMedian(vals);
    std::vector<double> devs;
    devs.reserve(vals.size());
    for (double v : vals)
      devs.push_back(std::abs(v - med));
    scale.sigma = std::max(1.4826 * QiMedian(devs), 1.0e-6);
    scale.ready = true;
    return scale;
  }

  Eigen::Vector3d OdometryManager::TransformLidarPoint(
      const PointCorrespondence &corr, const SE3d &T_lidar) const
  {
    return T_lidar * corr.point;
  }

  double OdometryManager::ComputeLidarResidual(
      const PointCorrespondence &corr, const SE3d &T_lidar) const
  {
    const Eigen::Vector3d point_map = TransformLidarPoint(corr, T_lidar);
    if (corr.geo_type == Plane)
      return point_map.dot(corr.geo_plane.head<3>()) + corr.geo_plane[3];
    return ((point_map - corr.geo_point).cross(corr.geo_normal)).norm();
  }

  double OdometryManager::ComputeVisualResidual(
      const QiVisualObs &obs, const SE3d &T_cam) const
  {
    const Eigen::Vector3d point_cam = T_cam.inverse() * obs.point;
    if (point_cam.z() <= 1.0e-6)
      return 1.0e3;
    const Eigen::Vector2d uv(
        K_(0, 0) * point_cam.x() / point_cam.z() + K_(0, 2),
        K_(1, 1) * point_cam.y() / point_cam.z() + K_(1, 2));
    return (obs.pixel - uv).norm();
  }

  Eigen::Matrix<double, 6, 6> OdometryManager::ComputeLidarInfo(
      const PointCorrespondence &corr,
      const Eigen::Vector3d &point_map,
      double final_weight) const
  {
    Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Zero();
    const Eigen::Matrix3d skew = SO3d::hat(point_map);
    if (corr.geo_type == Plane)
    {
      Eigen::Matrix<double, 1, 6> J;
      const Eigen::Vector3d n = corr.geo_plane.head<3>();
      J.block<1, 3>(0, 0) = -n.transpose() * skew;
      J.block<1, 3>(0, 3) = n.transpose();
      info = final_weight * final_weight * J.transpose() * J;
    }
    else
    {
      Eigen::Vector3d dir = corr.geo_normal;
      if (dir.norm() > 1.0e-9)
        dir.normalize();
      const Eigen::Matrix3d P = Eigen::Matrix3d::Identity() - dir * dir.transpose();
      Eigen::Matrix<double, 3, 6> J;
      J.block<3, 3>(0, 0) = P * (-skew);
      J.block<3, 3>(0, 3) = P;
      info = final_weight * final_weight * J.transpose() * J;
    }
    return info;
  }

  Eigen::Matrix<double, 6, 6> OdometryManager::ComputeVisualInfo(
      const QiVisualObs &obs, const SE3d &T_cam, double final_weight) const
  {
    Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Zero();
    const Eigen::Vector3d point_cam = T_cam.inverse() * obs.point;
    if (point_cam.z() <= 1.0e-6)
      return info;

    const double x = point_cam.x();
    const double y = point_cam.y();
    const double z = point_cam.z();
    Eigen::Matrix<double, 2, 3> J_pi;
    J_pi << K_(0, 0) / z, 0.0, -K_(0, 0) * x / (z * z),
        0.0, K_(1, 1) / z, -K_(1, 1) * y / (z * z);
    Eigen::Matrix<double, 3, 6> J_pose;
    J_pose.block<3, 3>(0, 0) = -SO3d::hat(point_cam);
    J_pose.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
    const Eigen::Matrix<double, 2, 6> J = -J_pi * J_pose;
    info = final_weight * final_weight * J.transpose() * J;
    return info;
  }

  double OdometryManager::QiLogDet(const Eigen::Matrix<double, 6, 6> &mat) const
  {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(mat);
    if (solver.info() != Eigen::Success)
      return -1.0e30;
    double logdet = 0.0;
    for (int i = 0; i < 6; ++i)
      logdet += std::log(std::max(solver.eigenvalues()[i], 1.0e-12));
    return logdet;
  }

  double OdometryManager::QiConditionNumber(
      const Eigen::Matrix<double, 6, 6> &mat) const
  {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(mat);
    if (solver.info() != Eigen::Success)
      return 0.0;
    const double min_ev = std::max(solver.eigenvalues()[0], 1.0e-12);
    const double max_ev = std::max(solver.eigenvalues()[5], 1.0e-12);
    return max_ev / min_ev;
  }

  void OdometryManager::BuildVisualObs(int64_t image_timestamp)
  {
    (void)image_timestamp;
    qi_visual_obs_.clear();
    v_points_.clear();
    px_obss_.clear();
    auto &tracked = camera_handler_->op_track.m_map_rgb_pts_in_last_frame_pos;
    for (auto it = tracked.begin(); it != tracked.end(); ++it)
    {
      RGB_pts *rgb_pt = static_cast<RGB_pts *>(it->first);
      if (!rgb_pt)
        continue;
      QiVisualObs obs;
      obs.point_ptr = rgb_pt;
      obs.point_id = rgb_pt->m_pt_index;
      obs.point = Eigen::Vector3d(rgb_pt->get_pos()(0, 0),
                                  rgb_pt->get_pos()(1, 0),
                                  rgb_pt->get_pos()(2, 0));
      obs.pixel = Eigen::Vector2d(it->second.x, it->second.y);
      qi_visual_obs_.push_back(obs);
      v_points_.push_back(obs.point);
      px_obss_.push_back(obs.pixel);
    }
  }

  void OdometryManager::PrepareQiVisualObs(int64_t image_timestamp)
  {
    qi_selected_visual_obs_.clear();
    qi_selected_visual_points_.clear();
    qi_selected_visual_pixels_.clear();
    qi_selected_visual_weights_.clear();
    qi_last_visual_candidates_ = static_cast<int>(qi_visual_obs_.size());
    qi_last_visual_selected_ = qi_last_visual_candidates_;
    if (qi_visual_obs_.empty())
    {
      qi_last_visual_weight_ratio_min_ = 1.0;
      qi_last_visual_weight_ratio_mean_ = 1.0;
      qi_last_visual_weight_ratio_max_ = 1.0;
      return;
    }

    const int n_track = std::max(camera_handler_->op_track.inlier_aft_track, 1);
    const int n_fmat = std::max(camera_handler_->op_track.inlier_aft_fmat, 0);
    const int n_pnp = std::max(camera_handler_->op_track.inlier_aft_pnp, 0);
    qi_last_n_pnp_ = n_pnp;
    qi_last_eta_fmat_ = QiClamp(1.0 * n_fmat / n_track, 0.0, 1.0);
    qi_last_eta_pnp_ = QiClamp(n_fmat > 0 ? 1.0 * n_pnp / n_fmat : 0.0,
                               0.0, 1.0);

    qi_recent_n_pnp_.push_back(static_cast<double>(n_pnp));
    while (qi_recent_n_pnp_.size() > 20)
      qi_recent_n_pnp_.pop_front();
    qi_n0_ = std::max(QiMedian(std::vector<double>(
                            qi_recent_n_pnp_.begin(), qi_recent_n_pnp_.end())),
                      1.0);

    const SE3d T_cam = trajectory_->GetCameraPoseNURBS(image_timestamp);
    std::vector<double> pre_residuals;
    pre_residuals.reserve(qi_visual_obs_.size());
    for (const auto &obs : qi_visual_obs_)
      pre_residuals.push_back(ComputeVisualResidual(obs, T_cam));

    qi_last_visual_pre_median_ = QiMedian(pre_residuals);
    std::vector<double> pre_dev;
    pre_dev.reserve(pre_residuals.size());
    for (double r : pre_residuals)
      pre_dev.push_back(std::abs(r - qi_last_visual_pre_median_));
    qi_last_visual_pre_mad_ = QiMedian(pre_dev);

    const double q_n = static_cast<double>(n_pnp) /
        (static_cast<double>(n_pnp) + qi_n0_);
    const double frame_quality = std::cbrt(QiClamp(
        q_n * qi_last_eta_pnp_ * qi_last_eta_fmat_, 0.0, 1.0));

    double q_min = std::numeric_limits<double>::max();
    double q_max = -std::numeric_limits<double>::max();
    double q_sum = 0.0;
    for (size_t i = 0; i < qi_visual_obs_.size(); ++i)
    {
      auto &obs = qi_visual_obs_[i];
      obs.residual_pre = pre_residuals[i];
      obs.frame_quality = frame_quality;
      obs.point_quality = qi_visual_scale_.ready
          ? std::exp(-obs.residual_pre / qi_visual_scale_.sigma)
          : 1.0;
      const double point_mix =
          (1.0 - qi_config_.visual_point_quality_weight) +
          qi_config_.visual_point_quality_weight * obs.point_quality;
      const double raw_quality = QiClamp(frame_quality * point_mix, 0.0, 1.0);
      obs.q = (qi_config_.quality_enable && qi_config_.visual_quality_enable)
          ? qi_config_.visual_q_min +
                (qi_config_.visual_q_max - qi_config_.visual_q_min) * raw_quality
          : 1.0;
      obs.base_weight = qi_image_weight_;
      obs.final_weight = obs.base_weight * std::sqrt(obs.q);
      obs.info = ComputeVisualInfo(obs, T_cam, obs.final_weight);
      q_min = std::min(q_min, obs.q);
      q_max = std::max(q_max, obs.q);
      q_sum += obs.q;
    }

    qi_last_visual_q_min_ = q_min;
    qi_last_visual_q_max_ = q_max;
    qi_last_visual_q_mean_ = q_sum / qi_visual_obs_.size();
    SelectQiVisualObs();
  }

  void OdometryManager::PrepareQiLidarObs(
      const Eigen::aligned_vector<PointCorrespondence> &point_corrs)
  {
    qi_lidar_obs_.clear();
    qi_selected_lidar_obs_.clear();
    qi_selected_point_corrs_.clear();
    qi_selected_lidar_weights_.clear();
    qi_last_lidar_candidates_ = static_cast<int>(point_corrs.size());
    qi_last_lidar_selected_ = qi_last_lidar_candidates_;
    if (point_corrs.empty())
      return;

    std::vector<double> pre_residuals;
    pre_residuals.reserve(point_corrs.size());
    double q_min = std::numeric_limits<double>::max();
    double q_max = -std::numeric_limits<double>::max();
    double q_sum = 0.0;

    for (const auto &corr : point_corrs)
    {
      QiLidarObs obs;
      obs.correspondence = corr;
      const SE3d T_lidar = trajectory_->GetLidarPoseNURBS(corr.t_point);
      const Eigen::Vector3d point_map = TransformLidarPoint(corr, T_lidar);
      obs.residual_pre = std::abs(ComputeLidarResidual(corr, T_lidar));

      if (qi_config_.quality_enable && qi_config_.lidar_quality_enable &&
          qi_lidar_scale_.ready)
      {
        const double residual_quality = std::exp(
            -obs.residual_pre / qi_lidar_scale_.sigma);
        obs.q = qi_config_.lidar_q_min +
            (qi_config_.lidar_q_max - qi_config_.lidar_q_min) *
                QiClamp(residual_quality, 0.0, 1.0);
      }
      else
      {
        obs.q = 1.0;
      }

      obs.base_weight = qi_lidar_weight_;
      if (use_lidar_scale_)
        obs.base_weight *= corr.scale;
      obs.final_weight = obs.base_weight * std::sqrt(obs.q);
      obs.info = ComputeLidarInfo(corr, point_map, obs.final_weight);

      pre_residuals.push_back(obs.residual_pre);
      q_min = std::min(q_min, obs.q);
      q_max = std::max(q_max, obs.q);
      q_sum += obs.q;
      qi_lidar_obs_.push_back(obs);
    }

    qi_last_lidar_q_min_ = q_min;
    qi_last_lidar_q_max_ = q_max;
    qi_last_lidar_q_mean_ = q_sum / qi_lidar_obs_.size();
    qi_last_lidar_pre_median_ = QiMedian(pre_residuals);
    std::vector<double> pre_dev;
    pre_dev.reserve(pre_residuals.size());
    for (double r : pre_residuals)
      pre_dev.push_back(std::abs(r - qi_last_lidar_pre_median_));
    qi_last_lidar_pre_mad_ = QiMedian(pre_dev);
    SelectQiLidarObs();
  }

  void OdometryManager::SelectQiLidarObs()
  {
    qi_selected_lidar_obs_.clear();
    qi_selected_point_corrs_.clear();
    qi_selected_lidar_weights_.clear();
    if (qi_lidar_obs_.empty())
      return;

    const int budget = std::min(qi_config_.max_lidar_obs,
                                static_cast<int>(qi_lidar_obs_.size()));
    const Eigen::Matrix<double, 6, 6> lambda0 =
        qi_config_.info_prior_eps * Eigen::Matrix<double, 6, 6>::Identity();
    Eigen::Matrix<double, 6, 6> lambda = lambda0;
    Eigen::Matrix<double, 6, 6> lambda_full = lambda0;
    for (const auto &obs : qi_lidar_obs_)
      lambda_full += obs.info;
    const double base_logdet = QiLogDet(lambda0);
    const double full_gain = std::max(QiLogDet(lambda_full) - base_logdet,
                                      1.0e-12);

    double gain_sum = 0.0;
    if (!qi_config_.selection_enable)
    {
      for (auto &obs : qi_lidar_obs_)
      {
        obs.selected = true;
        lambda += obs.info;
        qi_selected_lidar_obs_.push_back(obs);
      }
    }
    else
    {
      std::vector<char> used(qi_lidar_obs_.size(), 0);
      for (int k = 0; k < budget; ++k)
      {
        const double current_logdet = QiLogDet(lambda);
        double best_gain = -1.0e30;
        int best_idx = -1;
        for (size_t i = 0; i < qi_lidar_obs_.size(); ++i)
        {
          if (used[i])
            continue;
          const double gain = QiLogDet(lambda + qi_lidar_obs_[i].info) -
                                      current_logdet;
          if (gain > best_gain)
          {
            best_gain = gain;
            best_idx = static_cast<int>(i);
          }
        }
        if (best_idx < 0)
          break;
        if (k >= 6 && best_gain < qi_config_.selection_min_gain)
          break;

        used[best_idx] = 1;
        qi_lidar_obs_[best_idx].selected = true;
        qi_lidar_obs_[best_idx].info_gain = best_gain;
        lambda += qi_lidar_obs_[best_idx].info;
        gain_sum += best_gain;
        qi_selected_lidar_obs_.push_back(qi_lidar_obs_[best_idx]);

        const double coverage = QiClamp(
            (QiLogDet(lambda) - base_logdet) / full_gain, 0.0, 1.0);
        if (k + 1 >= 6 && coverage >= qi_config_.selection_info_ratio)
          break;
      }
    }

    double scale_min = std::numeric_limits<double>::max();
    double scale_max = -std::numeric_limits<double>::max();
    double scale_sum = 0.0;
    double base_min = std::numeric_limits<double>::max();
    double base_max = -std::numeric_limits<double>::max();
    double base_sum = 0.0;
    double ratio_min = std::numeric_limits<double>::max();
    double ratio_max = -std::numeric_limits<double>::max();
    double ratio_sum = 0.0;

    for (const auto &obs : qi_selected_lidar_obs_)
    {
      qi_selected_point_corrs_.push_back(obs.correspondence);
      qi_selected_lidar_weights_.push_back(obs.final_weight);
      const double ratio = obs.base_weight > 1.0e-12
          ? obs.final_weight / obs.base_weight : 1.0;
      const double scale = obs.correspondence.scale;
      scale_min = std::min(scale_min, scale);
      scale_max = std::max(scale_max, scale);
      scale_sum += scale;
      base_min = std::min(base_min, obs.base_weight);
      base_max = std::max(base_max, obs.base_weight);
      base_sum += obs.base_weight;
      ratio_min = std::min(ratio_min, ratio);
      ratio_max = std::max(ratio_max, ratio);
      ratio_sum += ratio;
    }

    const int selected = static_cast<int>(qi_selected_lidar_obs_.size());
    qi_last_lidar_selected_ = selected;
    qi_last_lidar_scale_min_ = selected > 0 ? scale_min : 1.0;
    qi_last_lidar_scale_mean_ = selected > 0 ? scale_sum / selected : 1.0;
    qi_last_lidar_scale_max_ = selected > 0 ? scale_max : 1.0;
    qi_last_lidar_base_weight_min_ = selected > 0 ? base_min : qi_lidar_weight_;
    qi_last_lidar_base_weight_mean_ = selected > 0 ? base_sum / selected : qi_lidar_weight_;
    qi_last_lidar_base_weight_max_ = selected > 0 ? base_max : qi_lidar_weight_;
    qi_last_lidar_weight_ratio_min_ = selected > 0 ? ratio_min : 1.0;
    qi_last_lidar_weight_ratio_mean_ = selected > 0 ? ratio_sum / selected : 1.0;
    qi_last_lidar_weight_ratio_max_ = selected > 0 ? ratio_max : 1.0;
    qi_last_lidar_logdet_ = QiLogDet(lambda);
    qi_last_lidar_cond_ = QiConditionNumber(lambda);
    qi_last_lidar_gain_mean_ = selected > 0 ? gain_sum / selected : 0.0;
    qi_last_lidar_info_coverage_ = QiClamp(
        (QiLogDet(lambda) - base_logdet) / full_gain, 0.0, 1.0);
  }

  void OdometryManager::SelectQiVisualObs()
  {
    qi_selected_visual_obs_.clear();
    qi_selected_visual_points_.clear();
    qi_selected_visual_pixels_.clear();
    qi_selected_visual_weights_.clear();
    if (qi_visual_obs_.empty())
      return;

    const int budget = std::min(qi_config_.max_visual_obs,
                                static_cast<int>(qi_visual_obs_.size()));
    const Eigen::Matrix<double, 6, 6> lambda0 =
        qi_config_.info_prior_eps * Eigen::Matrix<double, 6, 6>::Identity();
    Eigen::Matrix<double, 6, 6> lambda = lambda0;
    Eigen::Matrix<double, 6, 6> lambda_full = lambda0;
    for (const auto &obs : qi_visual_obs_)
      lambda_full += obs.info;
    const double base_logdet = QiLogDet(lambda0);
    const double full_gain = std::max(QiLogDet(lambda_full) - base_logdet,
                                      1.0e-12);

    double gain_sum = 0.0;
    if (!qi_config_.selection_enable)
    {
      for (auto &obs : qi_visual_obs_)
      {
        obs.selected = true;
        lambda += obs.info;
        qi_selected_visual_obs_.push_back(obs);
      }
    }
    else
    {
      std::vector<char> used(qi_visual_obs_.size(), 0);
      for (int k = 0; k < budget; ++k)
      {
        const double current_logdet = QiLogDet(lambda);
        double best_gain = -1.0e30;
        int best_idx = -1;
        for (size_t i = 0; i < qi_visual_obs_.size(); ++i)
        {
          if (used[i])
            continue;
          const double gain = QiLogDet(lambda + qi_visual_obs_[i].info) -
                                      current_logdet;
          if (gain > best_gain)
          {
            best_gain = gain;
            best_idx = static_cast<int>(i);
          }
        }
        if (best_idx < 0)
          break;
        if (k >= 6 && best_gain < qi_config_.selection_min_gain)
          break;

        used[best_idx] = 1;
        qi_visual_obs_[best_idx].selected = true;
        qi_visual_obs_[best_idx].info_gain = best_gain;
        lambda += qi_visual_obs_[best_idx].info;
        gain_sum += best_gain;
        qi_selected_visual_obs_.push_back(qi_visual_obs_[best_idx]);

        const double coverage = QiClamp(
            (QiLogDet(lambda) - base_logdet) / full_gain, 0.0, 1.0);
        if (k + 1 >= 6 && coverage >= qi_config_.selection_info_ratio)
          break;
      }
    }

    double ratio_min = std::numeric_limits<double>::max();
    double ratio_max = -std::numeric_limits<double>::max();
    double ratio_sum = 0.0;
    for (const auto &obs : qi_selected_visual_obs_)
    {
      qi_selected_visual_points_.push_back(obs.point);
      qi_selected_visual_pixels_.push_back(obs.pixel);
      qi_selected_visual_weights_.push_back(obs.final_weight);
      const double ratio = obs.base_weight > 1.0e-12
          ? obs.final_weight / obs.base_weight : 1.0;
      ratio_min = std::min(ratio_min, ratio);
      ratio_max = std::max(ratio_max, ratio);
      ratio_sum += ratio;
    }

    const int selected = static_cast<int>(qi_selected_visual_obs_.size());
    qi_last_visual_selected_ = selected;
    qi_last_visual_weight_ratio_min_ = selected > 0 ? ratio_min : 1.0;
    qi_last_visual_weight_ratio_mean_ = selected > 0 ? ratio_sum / selected : 1.0;
    qi_last_visual_weight_ratio_max_ = selected > 0 ? ratio_max : 1.0;
    qi_last_visual_logdet_ = QiLogDet(lambda);
    qi_last_visual_cond_ = QiConditionNumber(lambda);
    qi_last_visual_gain_mean_ = selected > 0 ? gain_sum / selected : 0.0;
    qi_last_visual_info_coverage_ = QiClamp(
        (QiLogDet(lambda) - base_logdet) / full_gain, 0.0, 1.0);
  }

  void OdometryManager::UpdateQiResidualStatistics(
      int64_t image_timestamp, bool process_image, bool optimization_success)
  {
    if (!optimization_success)
      return;

    // Estimate the robust residual scale from all current candidates rather
    // than only the information-selected subset. This prevents selection from
    // biasing the quality scale toward geometrically strong observations.
    std::vector<double> lidar_post;
    lidar_post.reserve(qi_lidar_obs_.size());
    for (auto &obs : qi_lidar_obs_)
    {
      const SE3d T_lidar = trajectory_->GetLidarPoseNURBS(
          obs.correspondence.t_point);
      obs.residual_post = std::abs(ComputeLidarResidual(
          obs.correspondence, T_lidar));
      qi_recent_lidar_residuals_.push_back(obs.residual_post);
      lidar_post.push_back(obs.residual_post);
    }
    while (qi_recent_lidar_residuals_.size() > 5000)
      qi_recent_lidar_residuals_.pop_front();
    if (!lidar_post.empty())
    {
      qi_last_lidar_post_median_ = QiMedian(lidar_post);
      std::vector<double> dev;
      dev.reserve(lidar_post.size());
      for (double r : lidar_post)
        dev.push_back(std::abs(r - qi_last_lidar_post_median_));
      qi_last_lidar_post_mad_ = QiMedian(dev);
    }

    if (process_image)
    {
      const SE3d T_cam = trajectory_->GetCameraPoseNURBS(image_timestamp);
      std::vector<double> visual_post;
      visual_post.reserve(qi_visual_obs_.size());
      for (auto &obs : qi_visual_obs_)
      {
        obs.residual_post = ComputeVisualResidual(obs, T_cam);
        qi_recent_visual_residuals_.push_back(obs.residual_post);
        visual_post.push_back(obs.residual_post);
      }
      while (qi_recent_visual_residuals_.size() > 2000)
        qi_recent_visual_residuals_.pop_front();
      if (!visual_post.empty())
      {
        qi_last_visual_post_median_ = QiMedian(visual_post);
        std::vector<double> dev;
        dev.reserve(visual_post.size());
        for (double r : visual_post)
          dev.push_back(std::abs(r - qi_last_visual_post_median_));
        qi_last_visual_post_mad_ = QiMedian(dev);
      }
    }

    qi_lidar_scale_ = QiMadScale(qi_recent_lidar_residuals_);
    qi_visual_scale_ = QiMadScale(qi_recent_visual_residuals_);
  }

  void OdometryManager::LogQiSummary() const
  {
    if (!qi_config_.log_enable)
      return;
    const bool lidar_quality_active =
        qi_config_.quality_enable && qi_config_.lidar_quality_enable;
    const bool visual_quality_active =
        qi_config_.quality_enable && qi_config_.visual_quality_enable;
    std::cout << GREEN << "[QI] LiDAR cand/sel "
              << qi_last_lidar_candidates_ << "/" << qi_last_lidar_selected_
              << " scale[min/mean/max] " << qi_last_lidar_scale_min_ << "/"
              << qi_last_lidar_scale_mean_ << "/" << qi_last_lidar_scale_max_
              << " base_w[min/mean/max] " << qi_last_lidar_base_weight_min_ << "/"
              << qi_last_lidar_base_weight_mean_ << "/" << qi_last_lidar_base_weight_max_
              << " q[min/mean/max] " << qi_last_lidar_q_min_ << "/"
              << qi_last_lidar_q_mean_ << "/" << qi_last_lidar_q_max_
              << " final/base[min/mean/max] " << qi_last_lidar_weight_ratio_min_ << "/"
              << qi_last_lidar_weight_ratio_mean_ << "/" << qi_last_lidar_weight_ratio_max_
              << " sigma_ready " << qi_lidar_scale_.ready
              << " quality " << lidar_quality_active
              << " pre med/mad " << qi_last_lidar_pre_median_ << "/"
              << qi_last_lidar_pre_mad_
              << " post med/mad " << qi_last_lidar_post_median_ << "/"
              << qi_last_lidar_post_mad_
              << " logdet/gain/cond/coverage " << qi_last_lidar_logdet_ << "/"
              << qi_last_lidar_gain_mean_ << "/" << qi_last_lidar_cond_ << "/"
              << qi_last_lidar_info_coverage_ << RESET << std::endl;

    std::cout << GREEN << "[QI] Visual cand/sel "
              << qi_last_visual_candidates_ << "/" << qi_last_visual_selected_
              << " N/eta_pnp/eta_fmat " << qi_last_n_pnp_ << "/"
              << qi_last_eta_pnp_ << "/" << qi_last_eta_fmat_
              << " q[min/mean/max] " << qi_last_visual_q_min_ << "/"
              << qi_last_visual_q_mean_ << "/" << qi_last_visual_q_max_
              << " final/image[min/mean/max] " << qi_last_visual_weight_ratio_min_ << "/"
              << qi_last_visual_weight_ratio_mean_ << "/" << qi_last_visual_weight_ratio_max_
              << " quality " << visual_quality_active
              << " pre med/mad " << qi_last_visual_pre_median_ << "/"
              << qi_last_visual_pre_mad_
              << " post med/mad " << qi_last_visual_post_median_ << "/"
              << qi_last_visual_post_mad_
              << " logdet/gain/cond/coverage " << qi_last_visual_logdet_ << "/"
              << qi_last_visual_gain_mean_ << "/" << qi_last_visual_cond_ << "/"
              << qi_last_visual_info_coverage_ << RESET << std::endl;
  }

  void OdometryManager::RunBag()
  {
    while (ros::ok())
    {
      /// [1] process a newly arrived frame of data: lidar or imu or camera
      msg_manager_->SpinBagOnce();
      if (!msg_manager_->has_valid_msg_)
      {
        break;
      }

      /// [2] static initialization, do not move at the begging!
      if (!is_initialized_)
      {
        while (!msg_manager_->imu_buf_.empty())
        {
          imu_initializer_->FeedIMUData(msg_manager_->imu_buf_.front());
          msg_manager_->imu_buf_.pop_front();
        }

        if (imu_initializer_->StaticInitialIMUState())
        {
          SetInitialState();
          std::cout << "\n🍺 Static initialization succeeds.\n";
          std::cout << "\n🍺 Trajectory start time: " << trajectory_->GetDataStartTime() << " ns.\n";
        }
        else
        {
          continue;
        }
      }

      /// [3] prepare data for the latest time interval delta_t
      static bool is_two_seg_prepared = false;
      static int seg_msg_cnt = 0;
      if (!is_two_seg_prepared)
      {
        if (PrepareTwoSegMsgs(seg_msg_cnt))  // prepare interval0 and interval1
        {
          seg_msg_cnt++;
        }
        if (seg_msg_cnt == 2)  // if interval0 and interval1 are ready
        {
          is_two_seg_prepared = true;
          UpdateTwoSeg();
          trajectory_->InitBlendMat();  // blending matrix is computed by knots of b-spline
        }
        else
        {
          continue;
        }
      }

      /// [4] update trajectory segment in the latest time interval delta_t
      if (PrepareMsgs())
      {
        // decide control point placement in the time interval delta_t by imu 
        UpdateOneSeg();  
        int offset = cp_add_num_cur + cp_add_num_next + cp_add_num_next_next;
        for (int i = 0; i < cp_add_num_cur; i++)
        {
          trajectory_->AddBlendMat(offset - i);  // blending matrix is computed by knots of b-spline
        }
        trajectory_manager_->SetDivision(cp_add_num_cur);
        trajectory_->startIdx = trajectory_->knts.size() - 1 - offset - 2;  // 2 serves as a margin or tolerance
        if (trajectory_->startIdx < 0)
        {
          trajectory_->startIdx = 0;
        }

        // fusing lidar-imu-camera to update the trajectory
        SolveLICO();

        // deep copy
        msg_manager_->cur_msgs = NextMsgs();
        msg_manager_->cur_msgs = msg_manager_->next_msgs;
        msg_manager_->cur_msgs.image = msg_manager_->next_msgs.image.clone();
        msg_manager_->next_msgs = NextMsgs();
        msg_manager_->next_msgs = msg_manager_->next_next_msgs;
        msg_manager_->next_msgs.image = msg_manager_->next_next_msgs.image.clone();
        msg_manager_->next_next_msgs = NextMsgs();

        traj_max_time_ns_cur = traj_max_time_ns_next;
        traj_max_time_ns_next = traj_max_time_ns_next_next;
        cp_add_num_cur = cp_add_num_next;
        cp_add_num_next = cp_add_num_next_next;

        while (msg_manager_->image_buf_.size() > 10)
        {
          msg_manager_->image_buf_.pop_front();
        }
      }
    }
  }

  void OdometryManager::SolveLICO()
  {
    msg_manager_->LogInfo();
    if (msg_manager_->cur_msgs.lidar_timestamp < 0)
    {
      // LOG(INFO) << "CANT SolveLICO!";
    }

    // lic optimization
    ProcessLICData();

    // Keep the current-window optimizer and the marginalization prior
    // consistent in both selected observations and per-observation weights.
    const bool qi_enabled = qi_config_.quality_enable ||
                            qi_config_.selection_enable;
    if (qi_enabled)
    {
      trajectory_manager_->UpdateLICPrior(
          qi_selected_point_corrs_,
          &qi_selected_lidar_weights_,
          &qi_selected_visual_weights_);
    }
    else
    {
      trajectory_manager_->UpdateLICPrior(
          lidar_handler_->GetPointCorrespondence());
    }

    // remove old imu data
    auto &msg = msg_manager_->cur_msgs;
    trajectory_manager_->UpdateLiDARAttribute(msg.lidar_timestamp,
                                              msg.lidar_max_timestamp);
  }

  void OdometryManager::ProcessLICData()
  {
    auto &msg = msg_manager_->cur_msgs;  // fake points with timestamp -1 exist up to now
    msg.CheckData();

    bool process_image = msg.if_have_image && msg.image_timestamp > t_begin_add_cam_;
    if (process_image)
    {
      // LOG(INFO) << "Process " << msg.scan_num << " scans in ["
      //           << msg.lidar_timestamp * NS_TO_S << ", " << msg.lidar_max_timestamp * NS_TO_S << "]"
      //           << "; image_time: " << msg.image_timestamp * NS_TO_S;
    }
    else
    {
      // LOG(INFO) << "Process " << msg.scan_num << " scans in ["
      //           << msg.lidar_timestamp * NS_TO_S << ", " << msg.lidar_max_timestamp * NS_TO_S << "]";
    }

    /// [1] transform the format of lidar pointcloud -> feature_cur_、feature_cur_ds_
    lidar_handler_->FeatureCloudHandler(msg.lidar_timestamp, msg.lidar_max_timestamp,
                                     msg.lidar_corner_cloud, msg.lidar_surf_cloud, msg.lidar_raw_cloud);  // fake points are removed

    /// [2] coarsely optimize trajectory based on prior、imu（served as good initial values）
    trajectory_manager_->PredictTrajectory(msg.lidar_timestamp, msg.lidar_max_timestamp,
                                           traj_max_time_ns_cur, cp_add_num_cur, non_uniform_);

    /// [3] update lidar local map
    int active_idx = trajectory_->numKnots() - 1 - cp_add_num_cur - 2;
    trajectory_->SetActiveTime(trajectory_->knts[active_idx]);
    lidar_handler_->UpdateLidarSubMap();

    /// [4] update visual local map（tracking map points for the current image frame）
    // after upate, m_map_rgb_pts_in_last_frame_pos = m_map_rgb_pts_in_current_frame_pos
    v_points_.clear();
    px_obss_.clear();
    if (process_image)
    {
      SE3d Twc = trajectory_->GetCameraPoseNURBS(msg.image_timestamp);
      camera_handler_->UpdateVisualSubMap(msg.image, msg.image_timestamp * NS_TO_S, Twc.unit_quaternion(), Twc.translation());
      BuildVisualObs(msg.image_timestamp);
      auto &map_rgb_pts_in_last_frame_pos = camera_handler_->op_track.m_map_rgb_pts_in_last_frame_pos;

      if (odom_viewer_.pub_track_img_.getNumSubscribers() != 0 || odom_viewer_.pub_sub_visual_map_.getNumSubscribers() != 0)
      {
        cv::Mat img_debug = camera_handler_->img_pose_->m_img.clone();
        VPointCloud visual_sub_map_debug;  // optical flow + ransac *2 -> 3d association（red）
        visual_sub_map_debug.clear();

        for (auto it = map_rgb_pts_in_last_frame_pos.begin(); it != map_rgb_pts_in_last_frame_pos.end(); it++)
        {
          RGB_pts *rgb_pt = ((RGB_pts *)it->first);
          cv::circle(img_debug, it->second, 2, cv::Scalar(0, 255, 0), -1, 8);  // optical flow + ransac *2 -> 2d association（green）
          VPoint temp_map;
          temp_map.x = rgb_pt->get_pos()(0, 0);
          temp_map.y = rgb_pt->get_pos()(1, 0);
          temp_map.z = rgb_pt->get_pos()(2, 0);
          temp_map.intensity = 0.;
          visual_sub_map_debug.push_back(temp_map);
        }

        cv_bridge::CvImage out_msg;
        out_msg.header.stamp = ros::Time::now();
        out_msg.encoding = sensor_msgs::image_encodings::BGR8;
        out_msg.image = img_debug;
        odom_viewer_.PublishTrackImg(out_msg.toImageMsg());
        odom_viewer_.PublishSubVisualMap(visual_sub_map_debug);
      }
    }

    /// [5] finely optimize trajectory based on prior、lidar、imu、camera
    bool qi_enabled = qi_config_.quality_enable || qi_config_.selection_enable;
    if (qi_enabled && process_image)
    {
      PrepareQiVisualObs(msg.image_timestamp);
    }
    else
    {
      qi_selected_visual_obs_.clear();
      qi_selected_visual_points_.clear();
      qi_selected_visual_pixels_.clear();
      qi_selected_visual_weights_.clear();
    }

    if (qi_enabled)
    {
      qi_selected_lidar_obs_.clear();
      qi_selected_point_corrs_.clear();
      qi_selected_lidar_weights_.clear();
    }

    bool last_opt_success = false;
    for (int iter = 0; iter < lidar_iter_; ++iter)
    {
      lidar_handler_->GetLoamFeatureAssociation();

      if (qi_enabled)
      {
        PrepareQiLidarObs(lidar_handler_->GetPointCorrespondence());
        if (process_image)
        {
          last_opt_success = trajectory_manager_->UpdateTrajectoryWithLIC(
              iter, msg.image_timestamp,
              qi_selected_point_corrs_, qi_selected_visual_points_, qi_selected_visual_pixels_, 8,
              &qi_selected_lidar_weights_, &qi_selected_visual_weights_);
        }
        else
        {
          last_opt_success = trajectory_manager_->UpdateTrajectoryWithLIC(
              iter, msg.image_timestamp,
              qi_selected_point_corrs_, {}, {}, 8,
              &qi_selected_lidar_weights_, nullptr);
          trajectory_manager_->SetProcessCurImg(false);
        }
      }
      else if (process_image)
      {
        last_opt_success = trajectory_manager_->UpdateTrajectoryWithLIC(
            iter, msg.image_timestamp,
            lidar_handler_->GetPointCorrespondence(), v_points_, px_obss_, 8);
      }
      else
      {
        last_opt_success = trajectory_manager_->UpdateTrajectoryWithLIC(
            iter, msg.image_timestamp,
            lidar_handler_->GetPointCorrespondence(), {}, {}, 8);
        trajectory_manager_->SetProcessCurImg(false);
      }
    }
    if (qi_enabled)
    {
      UpdateQiResidualStatistics(msg.image_timestamp, process_image,
                                 last_opt_success);
      LogQiSummary();
    }
    PublishCloudAndTrajectory();

    /// [6] update visual global map
    PosCloud::Ptr cloud_undistort = PosCloud::Ptr(new PosCloud);
    auto latest_feature_before_active_time = lidar_handler_->GetFeatureCurrent();
    PosCloud::Ptr cloud_distort = latest_feature_before_active_time.surface_features;
    if (cloud_distort->size() != 0)
    {
      trajectory_->UndistortScanInG(*cloud_distort, latest_feature_before_active_time.timestamp, *cloud_undistort);
      camera_handler_->UpdateVisualGlobalMap(cloud_undistort, latest_feature_before_active_time.time_max * NS_TO_S);
    }

    /// [7] associate new map points for the current image frame
    if (process_image)
    {
      SE3d Twc = trajectory_->GetCameraPoseNURBS(msg.image_timestamp);
      camera_handler_->AssociateNewPointsToCurrentImg(Twc.unit_quaternion(), Twc.translation());

      if (odom_viewer_.pub_undistort_scan_in_cur_img_.getNumSubscribers() != 0)
      {
        cv::Mat img_debug = camera_handler_->img_pose_->m_img.clone();
        {
          for (int i = 0; i < cloud_undistort->points.size(); i++)
          {
            auto pt = cloud_undistort->points[i];
            Eigen::Vector3d pt_e(pt.x, pt.y, pt.z);
            Eigen::Matrix3d Rwc = Twc.unit_quaternion().toRotationMatrix();
            Eigen::Vector3d twc = Twc.translation();
            Eigen::Vector3d pt_cam = Rwc.transpose() * pt_e - Rwc.transpose() * twc;
            double X = pt_cam.x(), Y = pt_cam.y(), Z = pt_cam.z();
            cv::Point2f pix(K_(0, 0) * X / Z + K_(0, 2), K_(1, 1) * Y / Z + K_(1, 2));
            cv::circle(img_debug, pix, 2, cv::Scalar(0, 0, 255), -1, 8);
          }
          cv_bridge::CvImage out_msg;
          out_msg.header.stamp = ros::Time::now();
          out_msg.encoding = sensor_msgs::image_encodings::BGR8;
          out_msg.image = img_debug;
          odom_viewer_.PublishUndistortScanInCurImg(out_msg.toImageMsg());
        }
      }

      if (odom_viewer_.pub_old_and_new_added_points_in_cur_img_.getNumSubscribers() != 0)
      {
        cv::Mat img_debug = camera_handler_->img_pose_->m_img.clone();
        auto obss = camera_handler_->op_track.m_map_rgb_pts_in_last_frame_pos;
        for (auto it = obss.begin(); it != obss.end(); it++)
        {
          cv::Point2f pix = it->second;
          cv::circle(img_debug, pix, 2, cv::Scalar(0, 255, 0), -1, 8);
        }

        cv_bridge::CvImage out_msg;
        out_msg.header.stamp = ros::Time::now();
        out_msg.encoding = sensor_msgs::image_encodings::BGR8;
        out_msg.image = img_debug;
        odom_viewer_.PublishOldAndNewAddedPointsInCurImg(out_msg.toImageMsg());
      }
    }

    /// [new] for Gaussian-LIC
    if (process_image && if_3dgs_)
    {
      Publish3DGSMappingData(msg);
    }

    /// [8] visualize tf in rviz
    auto pose = trajectory_->GetLidarPoseNURBS(msg.lidar_timestamp);
    auto pose_debug = trajectory_->GetCameraPoseNURBS(msg.lidar_timestamp);
    odom_viewer_.PublishTF(pose.unit_quaternion(), pose.translation(), "lidar",
                           "map");
    odom_viewer_.PublishTF(pose_debug.unit_quaternion(), pose_debug.translation(), "camera",
                           "map");
    odom_viewer_.PublishTF(trajectory_manager_->GetGlobalFrame(),
                           Eigen::Vector3d::Zero(), "map", "global");
  }

  bool OdometryManager::PrepareTwoSegMsgs(int seg_idx)
  {
    if (!is_initialized_)
      return false;

    int64_t data_start_time = trajectory_->GetDataStartTime();
    for (auto &data : msg_manager_->lidar_buf_)
    {
      if (!data.is_time_wrt_traj_start)
      {
        data.ToRelativeMeasureTime(data_start_time);                             // 
        msg_manager_->lidar_max_timestamps_[data.lidar_id] = data.max_timestamp; // 
      }
    }
    for (auto &data : msg_manager_->image_buf_)
    {
      if (!data.is_time_wrt_traj_start)
      {
        data.ToRelativeMeasureTime(data_start_time);
        msg_manager_->image_max_timestamp_ = data.timestamp;
      }
    }
    msg_manager_->RemoveBeginData(data_start_time, 0);

    // 
    // 
    int64_t traj_max_time_ns = trajectory_->maxTimeNsNURBS() + t_add_ns_ * (seg_idx + 1);

    // 
    // 
    bool have_msg = false;
    if (seg_idx == 0)
    {
      int64_t traj_last_max_time_ns = trajectory_->maxTimeNsNURBS();
      have_msg = msg_manager_->GetMsgs(msg_manager_->cur_msgs, traj_last_max_time_ns, traj_max_time_ns, data_start_time);
    }
    if (seg_idx == 1)
    {
      int64_t traj_last_max_time_ns = trajectory_->maxTimeNsNURBS() + t_add_ns_;
      have_msg = msg_manager_->GetMsgs(msg_manager_->next_msgs, traj_last_max_time_ns, traj_max_time_ns, data_start_time);
    }

    // 
    if (have_msg)
    {
      while (!msg_manager_->imu_buf_.empty())
      {
        trajectory_manager_->AddIMUData(msg_manager_->imu_buf_.front());
        msg_manager_->imu_buf_.pop_front();
      }
      if (seg_idx == 0)
      {
        traj_max_time_ns_cur = traj_max_time_ns;
      }
      if (seg_idx == 1)
      {
        traj_max_time_ns_next = traj_max_time_ns;
      }
      return true;
    }
    else
    {
      return false;
    }
  }

  void OdometryManager::UpdateTwoSeg()
  {
    auto imu_datas = trajectory_manager_->GetIMUData();

    /// update the first seg
    {
      int cp_add_num = cp_add_num_coarse_;
      Eigen::Vector3d aver_r = Eigen::Vector3d::Zero(), aver_a = Eigen::Vector3d::Zero();
      double var_r = 0, var_a = 0;
      int cnt = 0;
      for (int i = 0; i < imu_datas.size(); i++)
      {
        if (imu_datas[i].timestamp < trajectory_->maxTimeNsNURBS() ||
            imu_datas[i].timestamp >= traj_max_time_ns_cur)
          continue;
        cnt++;
        aver_r += imu_datas[i].gyro;
        aver_a += imu_datas[i].accel;
      }
      aver_r /= cnt;
      aver_a /= cnt;
      for (int i = 0; i < imu_datas.size(); i++)
      {
        if (imu_datas[i].timestamp < trajectory_->maxTimeNURBS() ||
            imu_datas[i].timestamp >= traj_max_time_ns_cur)
          continue;
        var_r += (imu_datas[i].gyro - aver_r).transpose() * (imu_datas[i].gyro - aver_r);
        var_a += (imu_datas[i].accel - aver_a).transpose() * (imu_datas[i].accel - aver_a);
      }
      var_r = sqrt(var_r / (cnt - 1));
      var_a = sqrt(var_a / (cnt - 1));
      // LOG(INFO) << "[aver_r_first] " << aver_r.norm() << " | [aver_a_first] " << aver_a.norm();
      // LOG(INFO) << "[var_r_first] " << var_r << " | [var_a_first] " << var_a;

      if (non_uniform_)
      {
        cp_add_num = GetKnotDensity(aver_r.norm(), aver_a.norm());
      }
      // LOG(INFO) << "[cp_add_num_first] " << cp_add_num;
      cp_num_vec.push_back(cp_add_num);

      int64_t step = (traj_max_time_ns_cur - trajectory_->maxTimeNsNURBS()) / cp_add_num;
      // LOG(INFO) << "[extend_step_first] " << step;
      for (int i = 0; i < cp_add_num - 1; i++)
      {
        int64_t time = trajectory_->maxTimeNsNURBS() + step * (i + 1);
        trajectory_->AddKntNs(time);
      }
      trajectory_->AddKntNs(traj_max_time_ns_cur);

      cp_add_num_cur = cp_add_num;
    }

    /// update the second seg
    {
      int cp_add_num = cp_add_num_coarse_;
      Eigen::Vector3d aver_r = Eigen::Vector3d::Zero(), aver_a = Eigen::Vector3d::Zero();
      double var_r = 0, var_a = 0;
      int cnt = 0;
      for (int i = 0; i < imu_datas.size(); i++)
      {
        if (imu_datas[i].timestamp < traj_max_time_ns_cur ||
            imu_datas[i].timestamp >= traj_max_time_ns_next)
          continue;
        cnt++;
        aver_r += imu_datas[i].gyro;
        aver_a += imu_datas[i].accel;
      }
      aver_r /= cnt;
      aver_a /= cnt;
      for (int i = 0; i < imu_datas.size(); i++)
      {
        if (imu_datas[i].timestamp < traj_max_time_ns_cur ||
            imu_datas[i].timestamp >= traj_max_time_ns_next)
          continue;
        var_r += (imu_datas[i].gyro - aver_r).transpose() * (imu_datas[i].gyro - aver_r);
        var_a += (imu_datas[i].accel - aver_a).transpose() * (imu_datas[i].accel - aver_a);
      }
      var_r = sqrt(var_r / (cnt - 1));
      var_a = sqrt(var_a / (cnt - 1));
      // LOG(INFO) << "[aver_r_second] " << aver_r.norm() << " | [aver_a_second] " << aver_a.norm();
      // LOG(INFO) << "[var_r_second] " << var_r << " | [var_a_second] " << var_a;

      if (non_uniform_)
      {
        cp_add_num = GetKnotDensity(aver_r.norm(), aver_a.norm());
      }
      // LOG(INFO) << "[cp_add_num_second] " << cp_add_num;
      cp_num_vec.push_back(cp_add_num);

      int64_t step = (traj_max_time_ns_next - traj_max_time_ns_cur) / cp_add_num;
      // LOG(INFO) << "[extend_step_second] " << step;
      for (int i = 0; i < cp_add_num - 1; i++)
      {
        int64_t time = traj_max_time_ns_cur + step * (i + 1);
        trajectory_->AddKntNs(time);
      }
      trajectory_->AddKntNs(traj_max_time_ns_next);

      cp_add_num_next = cp_add_num;
    }
  }

  bool OdometryManager::PrepareMsgs()
  {
    if (!is_initialized_)
      return false;

    int64_t data_start_time = trajectory_->GetDataStartTime();
    for (auto &data : msg_manager_->lidar_buf_)
    {
      if (!data.is_time_wrt_traj_start)
      {
        data.ToRelativeMeasureTime(data_start_time);                             // 
        msg_manager_->lidar_max_timestamps_[data.lidar_id] = data.max_timestamp; // 
      }
    }
    for (auto &data : msg_manager_->image_buf_)
    {
      if (!data.is_time_wrt_traj_start)
      {
        data.ToRelativeMeasureTime(data_start_time);
        msg_manager_->image_max_timestamp_ = data.timestamp;
      }
    }
    msg_manager_->RemoveBeginData(data_start_time, 0);

    int64_t traj_max_time_ns = traj_max_time_ns_next + t_add_ns_;

    int64_t traj_last_max_time_ns = traj_max_time_ns_next;
    bool have_msg = msg_manager_->GetMsgs(msg_manager_->next_next_msgs, traj_last_max_time_ns, traj_max_time_ns, data_start_time);

    if (have_msg)
    {
      while (!msg_manager_->imu_buf_.empty())
      {
        trajectory_manager_->AddIMUData(msg_manager_->imu_buf_.front());
        msg_manager_->imu_buf_.pop_front();
      }
      traj_max_time_ns_next_next = traj_max_time_ns;
      return true;
    }
    else
    {
      return false;
    }
  }

  void OdometryManager::UpdateOneSeg()
  {
    auto imu_datas = trajectory_manager_->GetIMUData();

    /// update the first seg
    {
      int cp_add_num = cp_add_num_coarse_;
      Eigen::Vector3d aver_r = Eigen::Vector3d::Zero(), aver_a = Eigen::Vector3d::Zero();
      double var_r = 0, var_a = 0;
      int cnt = 0;
      for (int i = 0; i < imu_datas.size(); i++)
      {
        if (imu_datas[i].timestamp < traj_max_time_ns_next ||
            imu_datas[i].timestamp >= traj_max_time_ns_next_next)
          continue;
        cnt++;
        aver_r += imu_datas[i].gyro;
        aver_a += imu_datas[i].accel;
      }
      aver_r /= cnt;
      aver_a /= cnt;
      for (int i = 0; i < imu_datas.size(); i++)
      {
        if (imu_datas[i].timestamp < traj_max_time_ns_next ||
            imu_datas[i].timestamp >= traj_max_time_ns_next_next)
          continue;
        var_r += (imu_datas[i].gyro - aver_r).transpose() * (imu_datas[i].gyro - aver_r);
        var_a += (imu_datas[i].accel - aver_a).transpose() * (imu_datas[i].accel - aver_a);
      }
      var_r = sqrt(var_r / (cnt - 1));
      var_a = sqrt(var_a / (cnt - 1));
      // LOG(INFO) << "[aver_r_new] " << aver_r.norm() << " | [aver_a_new] " << aver_a.norm();
      // LOG(INFO) << "[var_r_new] " << var_r << " | [var_a_new] " << var_a;

      if (non_uniform_)
      {
        cp_add_num = GetKnotDensity(aver_r.norm(), aver_a.norm());
      }
      // LOG(INFO) << "[cp_add_num_new] " << cp_add_num;
      cp_num_vec.push_back(cp_add_num);

      int64_t step = (traj_max_time_ns_next_next - traj_max_time_ns_next) / cp_add_num;
      // LOG(INFO) << "[extend_step_new] " << step;
      for (int i = 0; i < cp_add_num - 1; i++)
      {
        int64_t time = traj_max_time_ns_next + step * (i + 1);
        trajectory_->AddKntNs(time);
      }
      trajectory_->AddKntNs(traj_max_time_ns_next_next);

      cp_add_num_next_next = cp_add_num;
    }
  }

  void OdometryManager::SetInitialState()
  {
    if (is_initialized_)
    {
      assert(trajectory_->GetDataStartTime() > 0 && "data start time < 0");
      std::cout << RED << "[Error] system state has been initialized" << RESET << std::endl;
      return;
    }

    is_initialized_ = true;

    if (imu_initializer_->InitialDone())
    {
      SystemState sys_state = imu_initializer_->GetIMUState(); // I0toG
      trajectory_manager_->SetSystemState(sys_state, distance0_);

      trajectory_manager_->AddIMUData(imu_initializer_->GetIMUData().back());
      msg_manager_->imu_buf_.clear();
    }
    assert(trajectory_->GetDataStartTime() > 0 && "data start time < 0");
  }

  void OdometryManager::PublishCloudAndTrajectory()
  {
    odom_viewer_.PublishDenseCloud(trajectory_, lidar_handler_->GetFeatureMapDs(),
                                   lidar_handler_->GetFeatureCurrent());

    odom_viewer_.PublishSplineTrajectory(
        trajectory_, 0.0, trajectory_->maxTimeNURBS(), 0.1);
  }

  void OdometryManager::Publish3DGSMappingData(const NextMsgs& msg)
  {
    time_buf.push(msg.image_timestamp);
    lidar_buf.push(lidar_handler_->GetFeatureCurrent());
    img_buf.push(camera_handler_->img_pose_->m_img);

    while(1)
    {
      int64_t active_time = trajectory_->GetActiveTime();
      if (time_buf.front() < active_time && lidar_buf.front().time_max < active_time)
      {
        auto time = time_buf.front();
        auto lidar = lidar_buf.front();
        auto img = img_buf.front();
        time_buf.pop();
        lidar_buf.pop();
        img_buf.pop();

        PosCloud::Ptr cloud_undistort_ds = PosCloud::Ptr(new PosCloud);
        // PosCloud::Ptr cloud_distort_ds = lidar.surface_features;
        PosCloud::Ptr cloud_distort_ds = lidar.full_cloud;
        if (cloud_distort_ds->size() != 0)
        {
          trajectory_->UndistortScanInG(*cloud_distort_ds, lidar.timestamp, *cloud_undistort_ds);
          lidarpoints.push_back(cloud_undistort_ds);
        }

        // image
        odom_viewer_.Publish3DGSImage(img, time + trajectory_->GetDataStartTime());

        auto pose_cam = trajectory_->GetCameraPoseNURBS(time);
        auto inv_pose_cam = pose_cam.inverse();
        auto cam_K = camera_handler_->m_camera_intrinsic;
        double fx = cam_K(0, 0), fy = cam_K(1, 1);
        double cx = cam_K(0, 2), cy = cam_K(1, 2);
        int H = camera_handler_->img_pose_->m_img.rows;
        int W = camera_handler_->img_pose_->m_img.cols;

        // depth
        cv::Mat depthmap = cv::Mat::zeros(H, W, CV_32FC1);
        for (int j = std::max(0, int(lidarpoints.size()) - 5); j < lidarpoints.size(); j++)
        {
          auto lidarpoint = lidarpoints[j];
          for (int i = 0; i < lidarpoint->size(); i++)
          {
            auto pt = lidarpoint->points[i];
            Eigen::Vector3d pt_w = Eigen::Vector3d(pt.x, pt.y, pt.z);
            Eigen::Vector3d pt_c = inv_pose_cam.unit_quaternion().toRotationMatrix() * pt_w + inv_pose_cam.translation();
            double depth = pt_c(2);
            pt_c /= pt_c(2);
            double u = fx * pt_c(0) + cx;
            double v = fy * pt_c(1) + cy;
            int i_u = std::round(u), i_v = std::round(v);
            if (depth <= 0) continue;
            if (!((i_u >= 0 && i_u < W && i_v >= 0 && i_v < H))) continue;

            float& current_depth = depthmap.at<float>(i_v, i_u);
            if (current_depth == 0 || depth < current_depth) 
            {
                current_depth = depth;
            }
          }
        }
        while (lidarpoints.size() > 5)
        {
          lidarpoints.erase(lidarpoints.begin());
        }
        odom_viewer_.Publish3DGSDepth(depthmap, time + trajectory_->GetDataStartTime());

        // pose
        odom_viewer_.Publish3DGSPose(pose_cam.unit_quaternion(), pose_cam.translation(), time + trajectory_->GetDataStartTime());

        // points
        int filter_cnt = 0;
        int skip = lidar_skip_;
        Eigen::aligned_vector<Eigen::Vector3d> new_points;
        Eigen::aligned_vector<Eigen::Vector3i> new_colors;
        for (int i = 0; i < cloud_undistort_ds->points.size(); i += skip)
        {
          auto pt = cloud_undistort_ds->points[i];
          Eigen::Vector3d pt_w = Eigen::Vector3d(pt.x, pt.y, pt.z);
          Eigen::Vector3d pt_c = inv_pose_cam.unit_quaternion().toRotationMatrix() * pt_w + inv_pose_cam.translation();
          if (pt_c(2) < 0.01) 
          {
            filter_cnt++;
            continue;
          }
          pt_c /= pt_c(2);
          double u = fx * pt_c(0) + cx;
          double v = fy * pt_c(1) + cy;
          if (u < 0 || u > W - 1) 
          {
            filter_cnt++;
            continue;
          }
          new_points.push_back(Eigen::Vector3d(pt.x, pt.y, pt.z));

          int i_u = std::round(u), i_v = std::round(v);
          int blue = 0, green = 0, red = 0;
          if (i_u >= 0 && i_u < W && i_v >= 0 && i_v < H)
          {
            int u0 = std::floor(u), v0 = std::floor(v);
            int u1 = std::min(u0 + 1, W - 1), v1 = std::min(v0 + 1, H - 1);
            double du = u - u0, dv = v - v0;

            cv::Vec3b c00 = camera_handler_->img_pose_->m_img.at<cv::Vec3b>(v0, u0);
            cv::Vec3b c10 = camera_handler_->img_pose_->m_img.at<cv::Vec3b>(v0, u1);
            cv::Vec3b c01 = camera_handler_->img_pose_->m_img.at<cv::Vec3b>(v1, u0);
            cv::Vec3b c11 = camera_handler_->img_pose_->m_img.at<cv::Vec3b>(v1, u1);

            Eigen::Vector3d color00(c00[0], c00[1], c00[2]);
            Eigen::Vector3d color10(c10[0], c10[1], c10[2]);
            Eigen::Vector3d color01(c01[0], c01[1], c01[2]);
            Eigen::Vector3d color11(c11[0], c11[1], c11[2]);

            Eigen::Vector3d interpolated_color = 
                (1 - du) * (1 - dv) * color00 + 
                du * (1 - dv) * color10 + 
                (1 - du) * dv * color01 + 
                du * dv * color11;
            blue = std::round(interpolated_color.x());
            green = std::round(interpolated_color.y());
            red = std::round(interpolated_color.z());
          }
          new_colors.push_back(Eigen::Vector3i(red, green, blue));
        }
        odom_viewer_.Publish3DGSPoints(new_points, new_colors, time + trajectory_->GetDataStartTime());
      }
      else break;
    }
  }

  double OdometryManager::SaveOdometry()
  {
    std::string descri;
    if (odometry_mode_ == LICO)
      descri = "LICO";
    else if (odometry_mode_ == LIO)
      descri = "LIO";

    if (msg_manager_->NumLiDAR() > 1)
      descri = descri + "2";

    ros::Time timer;
    std::string time_full_str = std::to_string(timer.now().toNSec());
    std::string t_str = "_" + time_full_str.substr(time_full_str.size() - 4);

    int idx = -1;
    int64_t true_maxtime = trajectory_->maxTimeNsNURBS();
    for (int i = trajectory_->knts.size() - 1; i >= 0; i--)
    {
      if (true_maxtime == trajectory_->knts[i])
      {
        idx = i;
        break;
      }
    }
    idx -= 1;
    int64_t maxtime = trajectory_->knts[idx];
    maxtime = trajectory_->maxTimeNsNURBS() - 0.1 * S_TO_NS;

    trajectory_->ToTUMTxt(cache_path_ + "_" + descri + ".txt", maxtime, is_evo_viral_,
                          0.01);  // 100Hz pose querying

    // int sum_cp = std::accumulate(cp_num_vec.begin(), cp_num_vec.end(), 0);
    // std::cout << GREEN << "ave_cp_num " << sum_cp * 1.0 / cp_num_vec.size() << RESET << std::endl;

    return trajectory_->maxTimeNURBS();
  }

} // namespace cocolic
