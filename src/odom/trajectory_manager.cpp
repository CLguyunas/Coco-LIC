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

#include <odom/factor/analytic_diff/image_feature_factor.h>
#include <odom/factor/analytic_diff/trajectory_value_factor.h>
#include <odom/trajectory_manager.h>
#include <ros/assert.h>
#include <utils/log_utils.h>

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <fstream>
std::fstream myfile_t_ba;
namespace cocolic
{

  TrajectoryManager::TrajectoryManager(const YAML::Node &node,
                                       const std::string &config_path,
                                       Trajectory::Ptr trajectory)
      : verbose(false),
        cur_img_time_(-1),
        process_cur_img_(false),
        opt_weight_(OptWeight(node)),
        trajectory_(trajectory),
        lidar_marg_info(nullptr),
        cam_marg_info(nullptr)
  {
    std::string imu_yaml = node["imu_yaml"].as<std::string>();
    YAML::Node imu_node = YAML::LoadFile(config_path + imu_yaml);
    imu_state_estimator_ = std::make_shared<ImuStateEstimator>(imu_node);

    if_use_init_bg_ = imu_node["if_use_init_bg"].as<bool>();

    lidar_prior_ctrl_id = std::make_pair(0, 0);

    InitFactorInfo(trajectory_->GetSensorEP(CameraSensor),
                   trajectory_->GetSensorEP(LiDARSensor),
                   opt_weight_.image_weight, opt_weight_.local_velocity_info_vec);

    division_ = 0;
    use_marg_ = true;

    opt_cnt = 0;
    t_opt_sum = 0.0;

    v_points_.clear();
    px_obss_.clear();
  }

  void TrajectoryManager::InitFactorInfo(
      const ExtrinsicParam &Ep_CtoI, const ExtrinsicParam &Ep_LtoI,
      const double image_feature_weight,
      const Eigen::Vector3d &local_velocity_weight)
  {
    if (image_feature_weight > 1e-5)
    {
      Eigen::Matrix2d sqrt_info =
          image_feature_weight * Eigen::Matrix2d::Identity();

      analytic_derivative::ImageFeatureFactor::SetParam(Ep_CtoI.so3, Ep_CtoI.p);
      analytic_derivative::ImageFeatureFactor::sqrt_info = sqrt_info;

      analytic_derivative::Image3D2DFactor::SetParam(Ep_CtoI.so3, Ep_CtoI.p);
      analytic_derivative::Image3D2DFactor::sqrt_info = sqrt_info;

      analytic_derivative::ImageFeatureOnePoseFactor::SetParam(Ep_CtoI.so3,
                                                               Ep_CtoI.p);
      analytic_derivative::ImageFeatureOnePoseFactor::sqrt_info = sqrt_info;

      analytic_derivative::ImageDepthFactor::sqrt_info = sqrt_info;

      analytic_derivative::EpipolarFactor::SetParam(Ep_CtoI.so3, Ep_CtoI.p);
    }
    analytic_derivative::LoamFeatureOptMapPoseFactor::SetParam(Ep_LtoI.so3,
                                                               Ep_LtoI.p);
    analytic_derivative::RalativeLoamFeatureFactor::SetParam(Ep_LtoI.so3,
                                                             Ep_LtoI.p);
  }

  void TrajectoryManager::SetSystemState(const SystemState &sys_state, double distance0)
  {
    gravity_ = sys_state.g;

    SetOriginalPose(sys_state.q, sys_state.p);

    trajectory_->AddKntNs(0.0 * S_TO_NS);       // add knot t3   （t0、t1、t2 have been added in the constructor）
    trajectory_->AddKntNs(distance0 * S_TO_NS); // add knot t4
    trajectory_->SetMaxTimeNsNURBS(trajectory_->knts.back());

    SO3d R0(sys_state.q);
    for (size_t i = 0; i < trajectory_->numKnots(); i++)
    {
      trajectory_->setKnotSO3(R0, i);
    }
    // LOG(INFO) << "[debug numKnots] " << trajectory_->numKnots(); // 4

    tparam_.last_bias_time = trajectory_->maxTimeNsNURBS();
    tparam_.cur_bias_time = trajectory_->maxTimeNsNURBS();

    // TODO
    all_imu_bias_[tparam_.last_bias_time] = sys_state.bias;
    if (!if_use_init_bg_)
    {
      all_imu_bias_[tparam_.last_bias_time].gyro_bias = Eigen::Vector3d::Zero();
      all_imu_bias_[tparam_.last_bias_time].accel_bias = Eigen::Vector3d::Zero();
    }
  }

  void TrajectoryManager::SetOriginalPose(Eigen::Quaterniond q,
                                          Eigen::Vector3d p)
  {
    original_pose_.orientation.setQuaternion(q);
    original_pose_.position = p;
  }

  void TrajectoryManager::AddIMUData(const IMUData &data)
  {
    if (trajectory_->GetDataStartTime() < 0)
    {
      trajectory_->SetDataStartTime(data.timestamp);
    }
    imu_data_.emplace_back(data);
    imu_data_.back().timestamp -= trajectory_->GetDataStartTime();

    imu_state_estimator_->FeedIMUData(imu_data_.back());
  }

  void TrajectoryManager::AddPoseData(const PoseData &data)
  {
    pose_data_.emplace_back(data);
    pose_data_.back().timestamp -= trajectory_->GetDataStartTime();
  }

  void TrajectoryManager::RemoveIMUData(int64_t t_window_min)
  {
    if (t_window_min < 0)
      return;

    // https://stackoverflow.com/questions/991335/
    // how-to-erase-delete-pointers-to-objects-stored-in-a-vector
    for (auto iter = imu_data_.begin(); iter != imu_data_.end();)
    {
      if (iter->timestamp < t_window_min)
      {
        iter = imu_data_.erase(iter);
      }
      else
      {
        break;
      }
    }
  }

  void TrajectoryManager::RemovePoseData(int64_t t_window_min)
  {
    if (t_window_min < 0)
      return;

    // https://stackoverflow.com/questions/991335/
    // how-to-erase-delete-pointers-to-objects-stored-in-a-vector
    for (auto iter = pose_data_.begin(); iter != pose_data_.end();)
    {
      if (iter->timestamp < t_window_min)
      {
        iter = pose_data_.erase(iter);
      }
      else
      {
        break;
      }
    }
  }

  void TrajectoryManager::UpdateIMUInlio()
  {
    int64_t t_min = opt_min_t_ns;
    int64_t t_max = opt_max_t_ns;

    for (auto iter = imu_data_.begin(); iter != imu_data_.end(); ++iter)
    {
      if (iter->timestamp >= t_min)
      {
        if (iter->timestamp >= t_max)
        {
          continue;
        }
        tparam_.lio_imu_idx[0] = std::distance(imu_data_.begin(), iter);
        tparam_.lio_imu_time[0] = iter->timestamp;
        break;
      }
    }

    for (auto rter = imu_data_.rbegin(); rter != imu_data_.rend(); ++rter)
    {
      if (rter->timestamp < t_max)
      {
        tparam_.lio_imu_idx[1] =
            std::distance(imu_data_.begin(), rter.base()) - 1;
        tparam_.lio_imu_time[1] = rter->timestamp;
        break;
      }
    }
  }

  void TrajectoryManager::PredictTrajectory(int64_t scan_time_min, int64_t scan_time_max,
                                            int64_t traj_max_time_ns, int knot_add_num, bool non_uniform)
  {
    if (imu_data_.empty() || imu_data_.size() == 1)
    {
      // LOG(ERROR) << "[AppendWithIMUData] IMU data empty! ";
      return;
    }

    /// newly added interval：[opt_min_t_ns, opt_max_t_ns)
    opt_min_t_ns = trajectory_->maxTimeNsNURBS();

    /// extend trajectory by adding control points
    trajectory_->SetMaxTimeNsNURBS(traj_max_time_ns);
    opt_max_t_ns = trajectory_->maxTimeNsNURBS();
    SE3d last_knot = trajectory_->getLastKnot();
    trajectory_->extendKnotsTo(knot_add_num, last_knot);

    ////// color control point for visualization
    int intensity = 0;
    if (knot_add_num == 1)
    {
      intensity = 100;
    }
    else if (knot_add_num == 2)
    {
      intensity = 200;
    }
    else if (knot_add_num == 3)
    {
      intensity = 300;
    }
    else if (knot_add_num == 4)
    {
      intensity = 400;
    }
    for (int i = 0; i < knot_add_num; i++)
    {
      trajectory_->intensity_map[trajectory_->numKnots() + i] = intensity;
    }
    ////// color control point for visualization

    // LOG(INFO) << "[max_time_ns] " << opt_max_t_ns;
    // LOG(INFO) << "[numKnots aft extension] " << trajectory_->numKnots();

    tparam_.last_bias_time = tparam_.cur_bias_time;  // opt_min_t_ns
    tparam_.cur_bias_time = opt_max_t_ns;
    // LOG(INFO) << "[last_bias_time] " << tparam_.last_bias_time << " "
    //           << "[cur_bias_time] " << tparam_.cur_bias_time;
    tparam_.UpdateCurScan(scan_time_min, scan_time_max);
    UpdateIMUInlio();  // determine the imu data involved in this optimization

    /// optimization
    InitTrajWithPropagation();
  }

  void TrajectoryManager::InitTrajWithPropagation()
  {
    TrajectoryEstimatorOptions option;
    option.lock_ab = true;
    option.lock_wb = true;
    option.lock_g = true;
    option.lock_tran = false; // note
    option.show_residual_summary = verbose;
    TrajectoryEstimator::Ptr estimator(
        new TrajectoryEstimator(trajectory_, option, "Init Traj"));

    estimator->SetFixedIndex(3);

    // [0] prior factor
    if (true && lidar_marg_info)
    {
      estimator->AddMarginalizationFactor(lidar_marg_info,
                                          lidar_marg_parameter_blocks);
    }

    // [1] imu factor
    double *para_bg = all_imu_bias_.rbegin()->second.gyro_bias.data();
    double *para_ba = all_imu_bias_.rbegin()->second.accel_bias.data();
    for (int i = tparam_.lio_imu_idx[0]; i <= tparam_.lio_imu_idx[1]; ++i)
    {
      if (imu_data_.at(i).timestamp < opt_min_t_ns)
        continue;
      if (imu_data_.at(i).timestamp >= opt_max_t_ns)
        continue;
      estimator->AddIMUMeasurementAnalyticNURBS(imu_data_.at(i),
                                                para_bg, para_ba,
                                                gravity_.data(), //(0, 0, 9.8)
                                                opt_weight_.imu_info_vec);
    }

    ceres::Solver::Summary summary = estimator->Solve(50, false);
    static int init_cnt = 0;
    init_cnt++;
    // LOG(INFO) << init_cnt << " TrajInitSolver " << summary.BriefReport();
    // LOG(INFO) << init_cnt << " TrajInit Successful/Unsuccessful steps: "
    //           << summary.num_successful_steps << "/"
    //           << summary.num_unsuccessful_steps;
  }

  void TrajectoryManager::ConfigureCasrIntervention(
      const CasrInterventionConfig &config)
  {
    casr_intervention_config_ = config;
    casr_reference_snapshot_ = CasrReferenceSnapshot();
    last_casr_intervention_report_ = CasrInterventionReport();
  }

  void TrajectoryManager::CaptureCasrInterventionReference(
      int64_t scan_timestamp_ns)
  {
    casr_reference_snapshot_ = CasrReferenceSnapshot();
    if (!casr_intervention_config_.enabled || scan_timestamp_ns < 0)
    {
      return;
    }

    const int trajectory_control_point_num =
        static_cast<int>(trajectory_->numKnots());
    if (trajectory_control_point_num <= 0)
    {
      return;
    }
    const int retained_control_point_num = std::min(
        trajectory_control_point_num,
        casr_intervention_config_.max_control_points + SplineOrder);
    casr_reference_snapshot_.control_point_start_index =
        trajectory_control_point_num - retained_control_point_num;
    casr_reference_snapshot_.scan_timestamp_ns = scan_timestamp_ns;
    casr_reference_snapshot_.rotations.reserve(
        static_cast<size_t>(retained_control_point_num));
    casr_reference_snapshot_.positions.reserve(
        static_cast<size_t>(retained_control_point_num));
    for (int i = 0; i < retained_control_point_num; ++i)
    {
      const size_t knot_index = static_cast<size_t>(
          casr_reference_snapshot_.control_point_start_index + i);
      casr_reference_snapshot_.rotations.emplace_back(
          trajectory_->getKnotSO3(knot_index));
      casr_reference_snapshot_.positions.emplace_back(
          trajectory_->getKnotPos(knot_index));
    }
    casr_reference_snapshot_.valid = true;
  }

  bool TrajectoryManager::ExtractCasrReference(
      const CasrInterventionPlan &plan,
      int64_t scan_timestamp_ns,
      Eigen::aligned_vector<SO3d> &reference_rotations,
      Eigen::aligned_vector<Eigen::Vector3d> &reference_positions) const
  {
    reference_rotations.clear();
    reference_positions.clear();
    if (!casr_reference_snapshot_.valid ||
        casr_reference_snapshot_.scan_timestamp_ns != scan_timestamp_ns ||
        plan.control_point_start_index <
            casr_reference_snapshot_.control_point_start_index)
    {
      return false;
    }
    const int local_start = plan.control_point_start_index -
                            casr_reference_snapshot_.control_point_start_index;
    if (local_start < 0 ||
        local_start + plan.control_point_num >
            static_cast<int>(casr_reference_snapshot_.rotations.size()) ||
        local_start + plan.control_point_num >
            static_cast<int>(casr_reference_snapshot_.positions.size()))
    {
      return false;
    }

    reference_rotations.reserve(static_cast<size_t>(plan.control_point_num));
    reference_positions.reserve(static_cast<size_t>(plan.control_point_num));
    for (int i = 0; i < plan.control_point_num; ++i)
    {
      reference_rotations.emplace_back(
          casr_reference_snapshot_.rotations[
              static_cast<size_t>(local_start + i)]);
      reference_positions.emplace_back(
          casr_reference_snapshot_.positions[
              static_cast<size_t>(local_start + i)]);
    }
    return true;
  }

  bool TrajectoryManager::BuildCasrContinuityProjector(
      const CasrInterventionPlan &plan,
      const Eigen::aligned_vector<SO3d> &reference_rotations,
      Eigen::MatrixXd &affine_nullspace_projector,
      CasrInterventionReport &report) const
  {
    affine_nullspace_projector.resize(0, 0);
    report.continuity_operator_valid = false;
    report.continuity_operator_rank = 0;
    report.continuity_symmetry_error = 0.0;
    report.continuity_idempotence_error = 0.0;
    if (plan.recovery_mechanism ==
        CasrRecoveryMechanism::PropagationReference)
    {
      return true;
    }
    if (plan.control_point_num < 3 || plan.control_point_start_index < 0 ||
        plan.control_point_start_index + plan.control_point_num >
            static_cast<int>(trajectory_->knts.size()))
    {
      return false;
    }

    std::vector<double> knot_times_seconds;
    knot_times_seconds.reserve(static_cast<size_t>(plan.control_point_num));
    for (int i = 0; i < plan.control_point_num; ++i)
    {
      const int knot_index = plan.control_point_start_index + i;
      knot_times_seconds.emplace_back(
          trajectory_->knts[static_cast<size_t>(knot_index)] * NS_TO_S);
    }
    affine_nullspace_projector =
        analytic_derivative::BuildCasrAffineNullspaceProjector(
            knot_times_seconds, reference_rotations);
    const int dimension = 6 * plan.control_point_num;
    if (affine_nullspace_projector.rows() != dimension ||
        affine_nullspace_projector.cols() != dimension ||
        !affine_nullspace_projector.allFinite())
    {
      affine_nullspace_projector.resize(0, 0);
      return false;
    }

    report.continuity_symmetry_error =
        (affine_nullspace_projector -
         affine_nullspace_projector.transpose())
            .cwiseAbs()
            .maxCoeff();
    report.continuity_idempotence_error =
        (affine_nullspace_projector * affine_nullspace_projector -
         affine_nullspace_projector)
            .cwiseAbs()
            .maxCoeff();
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
        affine_nullspace_projector);
    if (solver.info() != Eigen::Success ||
        !solver.eigenvalues().allFinite())
    {
      affine_nullspace_projector.resize(0, 0);
      return false;
    }
    report.continuity_operator_rank =
        static_cast<int>((solver.eigenvalues().array() >
                          dso_fixed::kContinuityProjectorTolerance)
                             .count());
    report.continuity_operator_valid =
        report.continuity_symmetry_error <=
            dso_fixed::kContinuityProjectorTolerance &&
        report.continuity_idempotence_error <=
            dso_fixed::kContinuityProjectorTolerance &&
        report.continuity_operator_rank > 0;
    if (!report.continuity_operator_valid)
    {
      affine_nullspace_projector.resize(0, 0);
    }
    return report.continuity_operator_valid;
  }

  bool TrajectoryManager::ComputeCasrSourceCoordinates(
      const CasrInterventionPlan &plan,
      const Eigen::MatrixXd &recovery_basis,
      const Eigen::MatrixXd &affine_nullspace_projector,
      const Eigen::aligned_vector<SO3d> &reference_rotations,
      const Eigen::aligned_vector<Eigen::Vector3d> &reference_positions,
      Eigen::VectorXd &environment_coordinates,
      Eigen::VectorXd &support_coordinates,
      double *total_increment_norm,
      double *orthogonal_increment_norm,
      double *max_rotation_increment_rad,
      double *max_translation_increment_m) const
  {
    environment_coordinates.resize(0);
    support_coordinates.resize(0);
    if (plan.control_point_num <= 0 || plan.recovery_rank <= 0 ||
        recovery_basis.rows() != 6 * plan.control_point_num ||
        recovery_basis.cols() != plan.recovery_rank ||
        static_cast<int>(reference_rotations.size()) !=
            plan.control_point_num ||
        static_cast<int>(reference_positions.size()) !=
            plan.control_point_num)
    {
      return false;
    }

    Eigen::VectorXd scaled_increment =
        Eigen::VectorXd::Zero(6 * plan.control_point_num);
    double max_rotation = 0.0;
    double max_translation = 0.0;
    for (int i = 0; i < plan.control_point_num; ++i)
    {
      const size_t knot_index = static_cast<size_t>(
          plan.control_point_start_index + i);
      const Eigen::Vector3d rotation_increment =
          (reference_rotations[static_cast<size_t>(i)].inverse() *
           trajectory_->getKnotSO3(knot_index))
              .log();
      const Eigen::Vector3d translation_increment =
          trajectory_->getKnotPos(knot_index) -
          reference_positions[static_cast<size_t>(i)];
      if (!rotation_increment.allFinite() ||
          !translation_increment.allFinite())
      {
        return false;
      }
      scaled_increment.segment<3>(6 * i) =
          plan.characteristic_range * rotation_increment;
      scaled_increment.segment<3>(6 * i + 3) = translation_increment;
      max_rotation = std::max(max_rotation, rotation_increment.norm());
      max_translation =
          std::max(max_translation, translation_increment.norm());
    }

    environment_coordinates = recovery_basis.transpose() * scaled_increment;
    if (!environment_coordinates.allFinite())
    {
      return false;
    }
    if (affine_nullspace_projector.rows() == scaled_increment.size() &&
        affine_nullspace_projector.cols() == scaled_increment.size())
    {
      support_coordinates = recovery_basis.transpose() *
                            affine_nullspace_projector * scaled_increment;
      if (!support_coordinates.allFinite())
      {
        return false;
      }
    }
    else
    {
      support_coordinates = Eigen::VectorXd::Zero(plan.recovery_rank);
    }

    if (total_increment_norm)
    {
      *total_increment_norm = scaled_increment.norm();
    }
    if (orthogonal_increment_norm)
    {
      *orthogonal_increment_norm =
          (scaled_increment -
           recovery_basis * environment_coordinates)
              .norm();
    }
    if (max_rotation_increment_rad)
    {
      *max_rotation_increment_rad = max_rotation;
    }
    if (max_translation_increment_m)
    {
      *max_translation_increment_m = max_translation;
    }
    return true;
  }

  void TrajectoryManager::MeasureCasrIncrement(
      const CasrInterventionPlan &plan,
      const Eigen::MatrixXd &recovery_basis,
      const Eigen::MatrixXd &affine_nullspace_projector,
      const Eigen::aligned_vector<SO3d> &reference_rotations,
      const Eigen::aligned_vector<Eigen::Vector3d> &reference_positions,
      CasrMeasurementStage stage,
      CasrInterventionReport &report) const
  {
    Eigen::VectorXd environment_coordinates;
    Eigen::VectorXd support_coordinates;
    double total_norm = 0.0;
    double orthogonal_norm = 0.0;
    double max_rotation_increment_rad = 0.0;
    double max_translation_increment_m = 0.0;
    if (!ComputeCasrSourceCoordinates(
            plan, recovery_basis, affine_nullspace_projector,
            reference_rotations, reference_positions,
            environment_coordinates, support_coordinates, &total_norm,
            &orthogonal_norm, &max_rotation_increment_rad,
            &max_translation_increment_m))
    {
      return;
    }
    const double projected_norm = environment_coordinates.norm();
    const double environment_norm = environment_coordinates.norm();
    const double support_norm = support_coordinates.norm();
    double factor_residual_norm = 0.0;
    if (plan.sqrt_information_weights.size() == environment_coordinates.size())
    {
      const double environment_weighted =
          (plan.sqrt_information_weights.array() *
           environment_coordinates.array()).matrix().squaredNorm();
      const double support_weighted =
          (plan.sqrt_information_weights.array() *
           support_coordinates.array()).matrix().squaredNorm();
      if (plan.recovery_mechanism ==
          CasrRecoveryMechanism::SplineIncrementContinuity)
      {
        factor_residual_norm = std::sqrt(support_weighted);
      }
      else if (plan.recovery_mechanism ==
               CasrRecoveryMechanism::CoupledSourceConsensus)
      {
        factor_residual_norm =
            std::sqrt(0.5 * (environment_weighted + support_weighted));
      }
      else
      {
        factor_residual_norm = std::sqrt(environment_weighted);
      }
    }
    else
    {
      factor_residual_norm =
          plan.sqrt_information_weight * projected_norm;
    }
    if (stage == CasrMeasurementStage::PostSolve)
    {
      report.post_total_increment_norm = total_norm;
      report.post_projected_increment_norm = projected_norm;
      report.post_orthogonal_increment_norm = orthogonal_norm;
      report.post_factor_residual_norm = factor_residual_norm;
      report.post_environment_residual_norm = environment_norm;
      report.post_support_residual_norm = support_norm;
      report.max_rotation_increment_rad = max_rotation_increment_rad;
      report.max_translation_increment_m = max_translation_increment_m;
    }
    else if (stage == CasrMeasurementStage::Counterfactual)
    {
      report.counterfactual_total_increment_norm = total_norm;
      report.counterfactual_projected_increment_norm = projected_norm;
      report.counterfactual_orthogonal_increment_norm = orthogonal_norm;
      report.counterfactual_factor_residual_norm = factor_residual_norm;
      report.counterfactual_environment_residual_norm = environment_norm;
      report.counterfactual_support_residual_norm = support_norm;
    }
    else
    {
      report.pre_total_increment_norm = total_norm;
      report.pre_projected_increment_norm = projected_norm;
      report.pre_orthogonal_increment_norm = orthogonal_norm;
      report.pre_factor_residual_norm = factor_residual_norm;
      report.pre_environment_residual_norm = environment_norm;
      report.pre_support_residual_norm = support_norm;
    }
  }

  bool TrajectoryManager::UpdateTrajectoryWithLIC(
      int lidar_iter, int64_t img_time_stamp,
      const Eigen::aligned_vector<PointCorrespondence> &point_corrs,
      const Eigen::aligned_vector<Eigen::Vector3d> &pnp_3ds,
      const Eigen::aligned_vector<Eigen::Vector2d> &pnp_2ds,
      const int iteration,
      const CasrShadowResult *casr_result,
      double casr_characteristic_range,
      int64_t casr_scan_timestamp_ns)
  {
    last_casr_intervention_report_ = CasrInterventionReport();
    last_casr_intervention_report_.enabled =
        casr_intervention_config_.enabled;
    last_casr_intervention_report_.apply_to_estimator =
        casr_intervention_config_.apply_to_estimator;
    last_casr_intervention_report_.scan_timestamp_ns =
        std::max<int64_t>(0, casr_scan_timestamp_ns);
    last_casr_intervention_report_.base_information_weight =
        casr_intervention_config_.base_information_weight;
    last_casr_intervention_report_.max_effective_information_weight =
        casr_intervention_config_.max_effective_information_weight;
    last_casr_intervention_report_.curvature_matching_enabled =
        casr_intervention_config_.curvature_matching_enabled;
    last_casr_intervention_report_.counterfactual_enabled =
        casr_intervention_config_.counterfactual_validation;
    if (point_corrs.empty() || imu_data_.empty() || imu_data_.size() == 1)
    {
      if (casr_result)
      {
        last_casr_intervention_report_.route = casr_result->route;
        last_casr_intervention_report_.data_source =
            casr_result->data_source;
        last_casr_intervention_report_.state =
            casr_intervention_config_.enabled
                ? CasrInterventionState::InvalidInput
                : CasrInterventionState::Disabled;
      }
      // LOG(WARNING) << " input empty data " << point_corrs.size() << ", "
      //              << imu_data_.size();
      return false;
    }

    // LOG(INFO) << "[point_corrs size] " << point_corrs.size();
    // LOG(INFO) << "[opt_domain]: "
    //           << "[" << opt_min_t_ns * NS_TO_S << ", " << opt_max_t_ns * NS_TO_S << ")";

    IMUBias last_bias = all_imu_bias_.rbegin()->second;
    all_imu_bias_[tparam_.cur_bias_time] = last_bias;
    std::map<int, double *> para_bg_vec;
    std::map<int, double *> para_ba_vec;
    {
      auto &bias0 = all_imu_bias_[tparam_.last_bias_time]; // bi
      para_bg_vec[0] = bias0.gyro_bias.data();
      para_ba_vec[0] = bias0.accel_bias.data();

      auto &bias1 = all_imu_bias_[tparam_.cur_bias_time]; // bj
      para_bg_vec[1] = bias1.gyro_bias.data();
      para_ba_vec[1] = bias1.accel_bias.data();
    }

    TrajectoryEstimatorOptions option;
    option.lock_ab = false;
    option.lock_wb = false;
    option.lock_g = true;
    option.show_residual_summary = verbose;
    TrajectoryEstimator::Ptr estimator(
        new TrajectoryEstimator(trajectory_, option, "Before LIO"));

    estimator->SetFixedIndex(3);

    // [0] prior factor
    if (true && lidar_marg_info)
    {
      estimator->AddMarginalizationFactor(lidar_marg_info,
                                          lidar_marg_parameter_blocks);
    }

    // [1] lidar factor
    SO3d S_LtoI = trajectory_->GetSensorEP(LiDARSensor).so3;
    Eigen::Vector3d p_LinI = trajectory_->GetSensorEP(LiDARSensor).p;
    SO3d S_GtoM = SO3d(Eigen::Quaterniond::Identity());
    Eigen::Vector3d p_GinM = Eigen::Vector3d::Zero();

    for (const auto &v : point_corrs)
    {
      if (v.t_point < opt_min_t_ns)
        continue;
      if (v.t_point >= opt_max_t_ns)
        continue;
      if (v.t_point < tparam_.last_scan[1])
        continue;
      if (use_lidar_scale)
      {
        estimator->AddLoamMeasurementAnalyticNURBS(v, S_GtoM, p_GinM, S_LtoI, p_LinI,
                                                   opt_weight_.lidar_weight * v.scale);
      }
      else
      {
        estimator->AddLoamMeasurementAnalyticNURBS(v, S_GtoM, p_GinM, S_LtoI, p_LinI,
                                                   opt_weight_.lidar_weight);
      }
    }

    // [2] imu factor
    for (int i = tparam_.lio_imu_idx[0]; i < tparam_.lio_imu_idx[1]; ++i)
    {
      if (imu_data_.at(i).timestamp < opt_min_t_ns)
        continue;
      if (imu_data_.at(i).timestamp >= opt_max_t_ns)
        continue;
      estimator->AddIMUMeasurementAnalyticNURBS(imu_data_.at(i), para_bg_vec[0],
                                                para_ba_vec[0], gravity_.data(),
                                                opt_weight_.imu_info_vec);
    }

    /// [3] bias factor
    Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 6> noise_covariance = Eigen::Matrix<double, 6, 6>::Zero();
    noise_covariance.block<3, 3>(0, 0) = (opt_weight_.imu_noise.sigma_wb_discrete * opt_weight_.imu_noise.sigma_wb_discrete) * Eigen::Matrix3d::Identity();
    noise_covariance.block<3, 3>(3, 3) = (opt_weight_.imu_noise.sigma_ab_discrete * opt_weight_.imu_noise.sigma_ab_discrete) * Eigen::Matrix3d::Identity();
    for (int i = tparam_.lio_imu_idx[0] + 1; i < tparam_.lio_imu_idx[1]; ++i)
    {
      if (imu_data_.at(i - 1).timestamp < opt_min_t_ns)
        continue;
      if (imu_data_.at(i).timestamp >= opt_max_t_ns)
        continue;
      double dt = (imu_data_[i].timestamp - imu_data_[i - 1].timestamp) * NS_TO_S;
      Eigen::Matrix<double, 6, 6> F = Eigen::Matrix<double, 6, 6>::Zero();
      F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
      F.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
      Eigen::Matrix<double, 6, 6> G = Eigen::Matrix<double, 6, 6>::Zero();
      G.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * dt;
      G.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * dt;
      covariance = F * covariance * F.transpose() + G * noise_covariance * G.transpose();
    }

    Eigen::Matrix<double, 6, 6> sqrt_info_mat = Eigen::LLT<Eigen::Matrix<double, 6, 6>>(covariance.inverse()).matrixL().transpose();
    sqrt_info_ << sqrt_info_mat(0, 0), sqrt_info_mat(1, 1), sqrt_info_mat(2, 2), sqrt_info_mat(3, 3), sqrt_info_mat(4, 4), sqrt_info_mat(5, 5);
    estimator->AddBiasFactor(para_bg_vec[0], para_bg_vec[1], para_ba_vec[0],
                             para_ba_vec[1], 1, sqrt_info_);

    /// [4] pnp factor
    v_points_.clear();
    px_obss_.clear();
    if (pnp_3ds.size() != 0)
    {
      v_points_ = pnp_3ds;
      px_obss_ = pnp_2ds;
      process_cur_img_ = true;
      cur_img_time_ = img_time_stamp;
      for (int i = 0; i < pnp_3ds.size(); i++)
      {
        estimator->AddPnPMeasurementAnalyticNURBS(
            pnp_3ds[i], pnp_2ds[i],
            img_time_stamp,
            trajectory_->GetSensorEP(CameraSensor).so3,
            trajectory_->GetSensorEP(CameraSensor).p,
            K_, opt_weight_.image_weight);
      }
    }
    else
    {
      process_cur_img_ = false;
    }

    CasrInterventionPlan casr_plan;
    Eigen::MatrixXd casr_recovery_basis;
    Eigen::MatrixXd casr_continuity_projector;
    Eigen::aligned_vector<SO3d> casr_reference_rotations;
    Eigen::aligned_vector<Eigen::Vector3d> casr_reference_positions;
    bool casr_measurement_ready = false;
    const auto sync_casr_plan_to_report = [&]() {
      last_casr_intervention_report_.state = casr_plan.state;
      last_casr_intervention_report_.eligible = casr_plan.eligible;
      last_casr_intervention_report_.control_point_start_index =
          casr_plan.control_point_start_index;
      last_casr_intervention_report_.control_point_num =
          casr_plan.control_point_num;
      last_casr_intervention_report_.recovery_rank =
          casr_plan.recovery_rank;
      last_casr_intervention_report_.recovery_mechanism =
          casr_plan.recovery_mechanism;
      last_casr_intervention_report_.recovery_reference =
          casr_plan.recovery_reference;
      last_casr_intervention_report_.requested_activation_strength =
          casr_plan.requested_activation_strength;
      last_casr_intervention_report_.used_activation_strength =
          casr_plan.used_activation_strength;
      last_casr_intervention_report_.characteristic_range =
          casr_plan.characteristic_range;
      last_casr_intervention_report_.effective_information_weight =
          casr_plan.effective_information_weight;
      last_casr_intervention_report_.sqrt_information_weight =
          casr_plan.sqrt_information_weight;
      last_casr_intervention_report_.curvature_valid =
          casr_plan.curvature_valid;
      last_casr_intervention_report_.curvature_tangent_dimension =
          casr_plan.curvature_tangent_dimension;
      last_casr_intervention_report_.reference_curvature_max =
          casr_plan.reference_curvature_max;
      last_casr_intervention_report_.target_curvature =
          casr_plan.target_curvature;
      last_casr_intervention_report_.recovery_curvature_min =
          casr_plan.recovery_curvature_min;
      last_casr_intervention_report_.recovery_curvature_median =
          casr_plan.recovery_curvature_median;
      last_casr_intervention_report_.recovery_curvature_max =
          casr_plan.recovery_curvature_max;
      last_casr_intervention_report_.added_information_min =
          casr_plan.added_information_min;
      last_casr_intervention_report_.added_information_median =
          casr_plan.added_information_median;
      last_casr_intervention_report_.added_information_max =
          casr_plan.added_information_max;
    };
    if (casr_result)
    {
      casr_plan = BuildCasrInterventionPlan(
          casr_intervention_config_, *casr_result,
          casr_characteristic_range,
          static_cast<int>(trajectory_->numKnots()));
      last_casr_intervention_report_.route = casr_result->route;
      last_casr_intervention_report_.data_source = casr_result->data_source;
      sync_casr_plan_to_report();

      if (casr_plan.eligible)
      {
        if (!ExtractCasrReference(
                casr_plan, casr_scan_timestamp_ns,
                casr_reference_rotations, casr_reference_positions))
        {
          casr_plan.eligible = false;
          casr_plan.state = CasrInterventionState::MissingReference;
          sync_casr_plan_to_report();
        }
        else
        {
          if (!BuildCasrContinuityProjector(
                  casr_plan, casr_reference_rotations,
                  casr_continuity_projector,
                  last_casr_intervention_report_))
          {
            casr_plan.eligible = false;
            casr_plan.state =
                CasrInterventionState::InvalidTemporalSupport;
          }
          if (casr_plan.eligible &&
              casr_plan.recovery_mechanism !=
                  CasrRecoveryMechanism::PropagationReference)
          {
            const Eigen::MatrixXd continuity_basis =
                casr_continuity_projector *
                casr_result->recovery_knot_basis;
            const Eigen::JacobiSVD<Eigen::MatrixXd> continuity_svd(
                continuity_basis, Eigen::ComputeThinU |
                                      Eigen::ComputeThinV);
            if (continuity_svd.singularValues().size() !=
                    casr_plan.recovery_rank ||
                !continuity_svd.singularValues().allFinite() ||
                continuity_svd.singularValues().minCoeff() <=
                    dso_fixed::kContinuityProjectorTolerance)
            {
              casr_plan.eligible = false;
              casr_plan.state =
                  CasrInterventionState::InvalidTemporalSupport;
            }
          }
          if (casr_intervention_config_.curvature_matching_enabled)
          {
            CasrCurvatureEstimate curvature_estimate;
            estimator->EvaluateCasrProjectedCurvature(
                casr_plan.control_point_start_index,
                casr_plan.characteristic_range,
                casr_result->recovery_knot_basis,
                curvature_estimate);
            FinalizeCasrInterventionPlanWithCurvature(
                casr_intervention_config_, curvature_estimate, casr_plan);
          }
          casr_recovery_basis =
              casr_result->recovery_knot_basis *
              casr_plan.recovery_basis_rotation;
          casr_measurement_ready =
              casr_recovery_basis.rows() ==
                  casr_result->recovery_knot_basis.rows() &&
              casr_recovery_basis.cols() == casr_plan.recovery_rank &&
              casr_recovery_basis.allFinite();
          if (!casr_measurement_ready && casr_plan.eligible)
          {
            casr_plan.eligible = false;
            casr_plan.state = CasrInterventionState::InvalidBasis;
          }
          sync_casr_plan_to_report();
          if (casr_measurement_ready)
          {
            MeasureCasrIncrement(
                casr_plan, casr_recovery_basis,
                casr_continuity_projector,
                casr_reference_rotations, casr_reference_positions,
                CasrMeasurementStage::PreSolve,
                last_casr_intervention_report_);
          }
        }
      }
    }

    TicToc t_opt;
    static int loam_cnt = 0;
    ceres::Solver::Summary summary;
    ceres::Solver::Summary counterfactual_summary;
    TrajectoryEstimator::ParameterSnapshot initial_snapshot;
    TrajectoryEstimator::ParameterSnapshot counterfactual_solution;
    ceres::ResidualBlockId casr_residual_block_id = nullptr;
    bool counterfactual_attempted = false;
    bool initial_state_available = false;
    bool skip_casr_solve = false;
    bool committed_state_valid = true;
    const bool factor_requested =
        casr_plan.eligible && casr_intervention_config_.apply_to_estimator &&
        casr_measurement_ready;

    if (factor_requested)
    {
      // Snapshot every parameter block before either branch. Both solves must
      // start from this exact state, including biases and visual variables.
      initial_snapshot = estimator->CaptureParameterSnapshot();
      initial_state_available = !initial_snapshot.empty();
      if (!initial_state_available)
      {
        casr_plan.eligible = false;
        casr_plan.state = CasrInterventionState::InvalidInput;
        sync_casr_plan_to_report();
      }
    }

    if (factor_requested && initial_state_available &&
        casr_intervention_config_.counterfactual_validation)
    {
      counterfactual_attempted = true;
      counterfactual_summary = estimator->Solve(iteration, false);
      last_casr_intervention_report_.counterfactual_solver_usable =
          counterfactual_summary.IsSolutionUsable();
      last_casr_intervention_report_.counterfactual_solver_successful_steps =
          counterfactual_summary.num_successful_steps;
      last_casr_intervention_report_.counterfactual_solver_unsuccessful_steps =
          counterfactual_summary.num_unsuccessful_steps;
      MeasureCasrIncrement(
          casr_plan, casr_recovery_basis, casr_continuity_projector,
          casr_reference_rotations,
          casr_reference_positions,
          CasrMeasurementStage::Counterfactual,
          last_casr_intervention_report_);
      if (counterfactual_summary.IsSolutionUsable())
      {
        counterfactual_solution = estimator->CaptureParameterSnapshot();
      }
      bool coupled_consensus_blocked = false;
      if (counterfactual_summary.IsSolutionUsable() &&
          casr_plan.recovery_mechanism ==
              CasrRecoveryMechanism::CoupledSourceConsensus)
      {
        Eigen::VectorXd environment_coordinates;
        Eigen::VectorXd support_coordinates;
        if (ComputeCasrSourceCoordinates(
                casr_plan, casr_recovery_basis,
                casr_continuity_projector, casr_reference_rotations,
                casr_reference_positions, environment_coordinates,
                support_coordinates))
        {
          const CasrSourceConsensus consensus =
              EvaluateCasrSourceConsensus(environment_coordinates,
                                          support_coordinates);
          last_casr_intervention_report_.source_consensus_evaluated =
              consensus.evaluated;
          last_casr_intervention_report_.source_consensus_sufficient =
              consensus.sufficient;
          last_casr_intervention_report_.source_consensus_consistent =
              consensus.consistent;
          last_casr_intervention_report_.source_consensus_cosine =
              consensus.cosine;
          coupled_consensus_blocked =
              !consensus.sufficient || !consensus.consistent;
          if (coupled_consensus_blocked)
          {
            casr_plan.eligible = false;
            casr_plan.state = consensus.sufficient
                                  ? CasrInterventionState::SourceConsensusConflict
                                  : CasrInterventionState::SourceConsensusInsufficient;
            sync_casr_plan_to_report();
          }
        }
        else
        {
          coupled_consensus_blocked = true;
          casr_plan.eligible = false;
          casr_plan.state =
              CasrInterventionState::SourceConsensusInsufficient;
          sync_casr_plan_to_report();
        }
      }
      if (!estimator->RestoreParameterSnapshot(initial_snapshot))
      {
        skip_casr_solve = true;
        casr_plan.eligible = false;
        casr_plan.state = CasrInterventionState::SolverFailure;
        sync_casr_plan_to_report();
        if (!counterfactual_solution.empty())
        {
          committed_state_valid =
              estimator->RestoreParameterSnapshot(counterfactual_solution);
        }
        else
        {
          committed_state_valid = false;
        }
        summary = counterfactual_summary;
      }
      else if (coupled_consensus_blocked)
      {
        skip_casr_solve = true;
        if (!counterfactual_solution.empty())
        {
          committed_state_valid =
              estimator->RestoreParameterSnapshot(counterfactual_solution);
        }
        else
        {
          committed_state_valid = false;
        }
        summary = counterfactual_summary;
      }
    }

    if (factor_requested && initial_state_available && !skip_casr_solve)
    {
      casr_residual_block_id = estimator->AddCasrCauseDrivenIntervention(
          casr_plan.recovery_mechanism,
          casr_plan.control_point_start_index, casr_recovery_basis,
          casr_continuity_projector,
          casr_reference_rotations, casr_reference_positions,
          casr_plan.characteristic_range,
          casr_plan.sqrt_information_weights);
      last_casr_intervention_report_.factor_added =
          casr_residual_block_id != nullptr;
      if (!last_casr_intervention_report_.factor_added)
      {
        casr_plan.eligible = false;
        casr_plan.state = CasrInterventionState::InvalidBasis;
        sync_casr_plan_to_report();
        if (counterfactual_attempted)
        {
          if (!counterfactual_solution.empty())
          {
            committed_state_valid =
                estimator->RestoreParameterSnapshot(counterfactual_solution);
          }
          else
          {
            committed_state_valid = false;
          }
          summary = counterfactual_summary;
        }
        else
        {
          summary = estimator->Solve(iteration, false);
        }
        skip_casr_solve = true;
      }
    }

    if (last_casr_intervention_report_.factor_added && !skip_casr_solve)
    {
      summary = estimator->Solve(iteration, false);
      last_casr_intervention_report_.primary_solver_usable =
          summary.IsSolutionUsable();
      last_casr_intervention_report_.primary_solver_successful_steps =
          summary.num_successful_steps;
      last_casr_intervention_report_.primary_solver_unsuccessful_steps =
          summary.num_unsuccessful_steps;
      last_casr_intervention_report_.applied = summary.IsSolutionUsable();

      if (!summary.IsSolutionUsable())
      {
        const bool removed =
            estimator->RemoveResidualBlock(casr_residual_block_id);
        last_casr_intervention_report_.fallback_attempted = true;
        if (removed && counterfactual_attempted &&
            !counterfactual_solution.empty())
        {
          const bool baseline_restored =
              estimator->RestoreParameterSnapshot(counterfactual_solution);
          last_casr_intervention_report_.fallback_solver_usable =
              baseline_restored && counterfactual_summary.IsSolutionUsable();
          if (baseline_restored)
          {
            summary = counterfactual_summary;
          }
        }
        else if (removed && initial_state_available &&
                 estimator->RestoreParameterSnapshot(initial_snapshot))
        {
          summary = estimator->Solve(iteration, false);
          last_casr_intervention_report_.fallback_solver_usable =
              summary.IsSolutionUsable();
          if (!summary.IsSolutionUsable())
          {
            estimator->RestoreParameterSnapshot(initial_snapshot);
          }
        }
        casr_plan.state =
            last_casr_intervention_report_.fallback_solver_usable
                ? CasrInterventionState::SolverFailureRecovered
                : CasrInterventionState::SolverFailure;
        last_casr_intervention_report_.state = casr_plan.state;
        if (!last_casr_intervention_report_.fallback_solver_usable &&
            initial_state_available)
        {
          committed_state_valid =
              estimator->RestoreParameterSnapshot(initial_snapshot);
        }
      }
      else
      {
        last_casr_intervention_report_.state =
            CasrInterventionState::Applied;
      }
    }
    else if (!counterfactual_attempted && !skip_casr_solve)
    {
      summary = estimator->Solve(iteration, false);
      last_casr_intervention_report_.primary_solver_usable =
          summary.IsSolutionUsable();
      last_casr_intervention_report_.primary_solver_successful_steps =
          summary.num_successful_steps;
      last_casr_intervention_report_.primary_solver_unsuccessful_steps =
          summary.num_unsuccessful_steps;
    }

    last_casr_intervention_report_.solver_usable =
        committed_state_valid && summary.IsSolutionUsable();
    last_casr_intervention_report_.solver_successful_steps =
        summary.num_successful_steps;
    last_casr_intervention_report_.solver_unsuccessful_steps =
        summary.num_unsuccessful_steps;
    if (casr_measurement_ready && !casr_reference_rotations.empty())
    {
      MeasureCasrIncrement(
          casr_plan, casr_recovery_basis,
          casr_continuity_projector,
          casr_reference_rotations, casr_reference_positions,
          CasrMeasurementStage::PostSolve,
          last_casr_intervention_report_);
      if (counterfactual_attempted &&
          last_casr_intervention_report_.counterfactual_solver_usable &&
          last_casr_intervention_report_.applied)
      {
        const double denominator_floor =
            casr_intervention_config_.counterfactual_ratio_denominator_floor;
        last_casr_intervention_report_.projected_casr_over_counterfactual =
            last_casr_intervention_report_.post_projected_increment_norm /
            std::max(denominator_floor,
                     last_casr_intervention_report_
                         .counterfactual_projected_increment_norm);
        last_casr_intervention_report_.orthogonal_casr_over_counterfactual =
            last_casr_intervention_report_.post_orthogonal_increment_norm /
            std::max(denominator_floor,
                     last_casr_intervention_report_
                         .counterfactual_orthogonal_increment_norm);
      }
    }
    double opt_time = t_opt.toc();
    // LOG(INFO) << "[t_opt] " << opt_time << std::endl;
    // LOG(INFO) << "LoamSolver " << summary.BriefReport();
    // LOG(INFO) << ++loam_cnt << " UpdateLio Successful/Unsuccessful steps: "
    //           << summary.num_successful_steps << "/"
    //           << summary.num_unsuccessful_steps;

    opt_cnt++;
    t_opt_sum += opt_time;

    // LOG(INFO) << "[gyro_bias_new] " << all_imu_bias_.rbegin()->second.gyro_bias.x() << " "
    //           << all_imu_bias_.rbegin()->second.gyro_bias.y() << " "
    //           << all_imu_bias_.rbegin()->second.gyro_bias.z();
    // LOG(INFO) << "[acce_bias_new] " << all_imu_bias_.rbegin()->second.accel_bias.x() << " "
    //           << all_imu_bias_.rbegin()->second.accel_bias.y() << " "
    //           << all_imu_bias_.rbegin()->second.accel_bias.z();

    return true;
  }

  void TrajectoryManager::UpdateLiDARAttribute(double scan_time_min,
                                               double scan_time_max)
  {
    if (trajectory_->maxTimeNsNURBS() > 25 * S_TO_NS)
    {
      int64_t t = trajectory_->maxTimeNsNURBS() - 15 * S_TO_NS;
      RemoveIMUData(t);
      RemovePoseData(t);
    }
  }

  void TrajectoryManager::UpdateLICPrior(
      const Eigen::aligned_vector<PointCorrespondence> &point_corrs)
  {
    TrajectoryEstimatorOptions option;
    option.is_marg_state = true;

    TrajectoryEstimator::Ptr estimator(
        new TrajectoryEstimator(trajectory_, option));  // AddControlPoint

    // construct a new prior
    MarginalizationInfo *marginalization_info = new MarginalizationInfo();

    // prepare the control points and biases to be marginalized
    int lhs_idx = trajectory_->numKnots() - 1 - division_ - 2;  // retain the last 3 control points in this optimization; remember, cubic spline is adopted
    int rhs_idx = trajectory_->numKnots() - 4;

    auto &last_bias = all_imu_bias_[tparam_.last_bias_time];  // marginalize the bias bi
    auto &cur_bias = all_imu_bias_[tparam_.cur_bias_time];
    std::vector<double *> drop_param;
    for (int i = lhs_idx; i <= rhs_idx; i++)
    {
      drop_param.emplace_back(trajectory_->getKnotSO3(i).data());
      drop_param.emplace_back(trajectory_->getKnotPos(i).data());
    }
    drop_param.emplace_back(last_bias.gyro_bias.data());
    drop_param.emplace_back(last_bias.accel_bias.data());

    // [0] prior factor marginalization
    if (lidar_marg_info)
    {
      std::vector<int> drop_set;
      for (int i = 0; i < lidar_marg_parameter_blocks.size(); i++)
      {
        for (auto const &dp : drop_param)
        {
          if (lidar_marg_parameter_blocks[i] == dp)
          {
            drop_set.emplace_back(i);
            break;
          }
        }
      }

      if (!drop_set.empty())
      {
        MarginalizationFactor *cost_function = new MarginalizationFactor(lidar_marg_info);
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(RType_Prior, cost_function, NULL,
                                                                       lidar_marg_parameter_blocks, drop_set);
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }
    }

    // [1] imu factor marginalization
    for (int i = tparam_.lio_imu_idx[0]; i < tparam_.lio_imu_idx[1]; ++i)
    {
      if (imu_data_.at(i).timestamp < opt_min_t_ns)
        continue;
      if (imu_data_.at(i).timestamp >= opt_max_t_ns)
        continue;
      int64_t time_ns = imu_data_.at(i).timestamp;
      std::pair<int, double> su; // i u
      trajectory_->GetIdxT(time_ns, su);
      Eigen::Matrix4d blending_matrix = trajectory_->blending_mats[su.first - 3];
      Eigen::Matrix4d cumulative_blending_matrix = trajectory_->cumu_blending_mats[su.first - 3];
      std::vector<double *> vec;
      estimator->AddControlPointsNURBS(su.first - 3, vec);
      estimator->AddControlPointsNURBS(su.first - 3, vec, true);
      vec.emplace_back(last_bias.gyro_bias.data());
      vec.emplace_back(last_bias.accel_bias.data());

      std::vector<int> drop_set;
      for (int i = 0; i < vec.size(); i++)
      {
        for (auto const &dp : drop_param)
        {
          if (vec[i] == dp)
          {
            drop_set.emplace_back(i);
            break;
          }
        }
      }

      if (!drop_set.empty())
      {
        ceres::CostFunction *cost_function = new analytic_derivative::IMUFactorNURBS(
            time_ns, imu_data_.at(i), gravity_, opt_weight_.imu_info_vec, trajectory_->knts, su,
            blending_matrix, cumulative_blending_matrix);
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(RType_IMU, cost_function, NULL,
                                                                       vec, drop_set);
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }
    }

    // [2] lidar factor marginalization
    SO3d S_LtoI = trajectory_->GetSensorEP(LiDARSensor).so3;
    Eigen::Vector3d p_LinI = trajectory_->GetSensorEP(LiDARSensor).p;
    SO3d S_GtoM = SO3d(Eigen::Quaterniond::Identity());
    Eigen::Vector3d p_GinM = Eigen::Vector3d::Zero();
    for (const auto &v : point_corrs)
    {
      if (v.t_point < opt_min_t_ns)
        continue;
      if (v.t_point >= opt_max_t_ns)
        continue;
      if (v.t_point < tparam_.last_scan[1])
        continue;
      int64_t time_ns = v.t_point;
      std::pair<int, double> su; // i and u
      trajectory_->GetIdxT(time_ns, su);
      Eigen::Matrix4d blending_matrix = trajectory_->blending_mats[su.first - 3];
      Eigen::Matrix4d cumulative_blending_matrix = trajectory_->cumu_blending_mats[su.first - 3];
      std::vector<double *> vec;
      estimator->AddControlPointsNURBS(su.first - 3, vec);
      estimator->AddControlPointsNURBS(su.first - 3, vec, true);

      std::vector<int> drop_set;
      for (int i = 0; i < vec.size(); i++)
      {
        for (auto const &dp : drop_param)
        {
          if (vec[i] == dp)
          {
            drop_set.emplace_back(i);
            break;
          }
        }
      }

      if (!drop_set.empty())
      {
        double weight = opt_weight_.lidar_weight;
        if (use_lidar_scale)
        {
          weight *= v.scale;
        }
        ceres::CostFunction *cost_function = new analytic_derivative::LoamFeatureFactorNURBS(
            time_ns, v, su, blending_matrix, cumulative_blending_matrix,
            S_GtoM, p_GinM, S_LtoI, p_LinI, weight);
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(RType_LiDAR, cost_function, NULL,
                                                                       vec, drop_set);
        int num_residuals = cost_function->num_residuals();
        Eigen::MatrixXd residuals;
        residuals.setZero(num_residuals, 1);
        cost_function->Evaluate(vec.data(), residuals.data(), nullptr);
        double dist = (residuals / weight).norm();
        if (dist < 0.05)
        // if (dist < 0.01)
        {
          marginalization_info->addResidualBlockInfo(residual_block_info);
        }
      }
    }

    // [3] bias factor marginalization
    std::vector<double *> vec;
    vec.emplace_back(last_bias.gyro_bias.data());
    vec.emplace_back(cur_bias.gyro_bias.data());
    vec.emplace_back(last_bias.accel_bias.data());
    vec.emplace_back(cur_bias.accel_bias.data());

    std::vector<int> drop_set;
    drop_set.emplace_back(0); // bgi
    drop_set.emplace_back(2); // bai

    analytic_derivative::BiasFactor *cost_function = new analytic_derivative::BiasFactor(1, sqrt_info_);
    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(RType_Bias, cost_function, NULL,
                                                                   vec, drop_set);
    marginalization_info->addResidualBlockInfo(residual_block_info);

    /// [4] pnp factor marginalization
    if (process_cur_img_ && v_points_.size() != 0)
    {
      int64_t time_ns = cur_img_time_;
      std::pair<int, double> su; // i和u
      trajectory_->GetIdxT(time_ns, su);
      Eigen::Matrix4d blending_matrix = trajectory_->blending_mats[su.first - 3];
      Eigen::Matrix4d cumulative_blending_matrix = trajectory_->cumu_blending_mats[su.first - 3];
      std::vector<double *> vec;
      estimator->AddControlPointsNURBS(su.first - 3, vec);
      estimator->AddControlPointsNURBS(su.first - 3, vec, true);

      std::vector<int> drop_set;
      for (int i = 0; i < vec.size(); i++)
      {
        for (auto const &dp : drop_param)
        {
          if (vec[i] == dp)
          {
            drop_set.emplace_back(i);
            break;
          }
        }
      }

      if (!drop_set.empty())
      {
        Eigen::Matrix3d K;
        for (int i = 0; i < v_points_.size(); i++)
        {
          ceres::CostFunction *cost_function = new analytic_derivative::PnPFactorNURBS(
              time_ns, su,
              blending_matrix, cumulative_blending_matrix,
              v_points_[i], px_obss_[i],
              trajectory_->GetSensorEP(CameraSensor).so3,
              trajectory_->GetSensorEP(CameraSensor).p,
              K_, opt_weight_.image_weight);
          ceres::LossFunction *loss_function = NULL;
          loss_function = new ceres::CauchyLoss(10.0); // adopted from vins-mono
          ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(RType_Image, cost_function, loss_function,
                                                                         vec, drop_set);
          marginalization_info->addResidualBlockInfo(residual_block_info);
        }
      }
    }

    marginalization_info->preMarginalize();
    marginalization_info->marginalize();
    if (lidar_marg_info)
    {
      lidar_marg_info = nullptr;
    }
    lidar_marg_info.reset(marginalization_info);
    lidar_marg_parameter_blocks = marginalization_info->getParameterBlocks();
  }

} // namespace cocolic
