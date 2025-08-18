#pragma once

#include <steam.hpp>
#include <steam/problem/cost_term/gyro_super_cost_term.hpp>
#include <steam/problem/cost_term/imu_super_cost_term.hpp>
#include <steam/problem/cost_term/p2p_const_vel_super_cost_term.hpp>
#include <steam/problem/cost_term/preintegrated_accel_cost_term.hpp>
#include <odometry.hpp>
#include <utils/stopwatch.hpp>

#include <fstream>
#include <set>

namespace stateestimate {
    class lidarodom : public Odometry {
        public:
            using Matrix12d = Eigen::Matrix<double, 12, 12>;
            enum class LOSS_FUNC { L2, DCS, CAUCHY, GM };

            struct Neighborhood {
                Eigen::Vector3d center = Eigen::Vector3d::Zero();
                Eigen::Vector3d normal = Eigen::Vector3d::Zero();
                Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
                double a2D = 1.0;  // Planarity coefficient
            };

            struct Options : public Odometry::Options {
                // ----------------------------------------------------------------------------------
                // Sensor and Vehicle Configuration
                // ----------------------------------------------------------------------------------
                
                /// Fixed transformation from the robot's body frame to the sensor's frame (e.g., LiDAR). This is the extrinsic calibration.
                Eigen::Matrix<double, 4, 4> Tb2s = Eigen::Matrix<double, 4, 4>::Identity();
                
                // ----------------------------------------------------------------------------------
                // Continuous-Time Trajectory Model Parameters
                // ----------------------------------------------------------------------------------

                /// Diagonal elements of the continuous-time motion model's process noise covariance matrix ($Q_c$). Controls the uncertainty of motion.
                Eigen::Matrix<double, 6, 1> qc_diag = Eigen::Matrix<double, 6, 1>::Ones();
                /// Number of additional states (knots) to add between the start and end of a scan for the continuous-time trajectory.
                int num_extra_states = 0;
                
                // ----------------------------------------------------------------------------------
                // Point-to-Plane (P2P) ICP Parameters
                // ----------------------------------------------------------------------------------

                /// Exponent applied to the planarity score to weight point-to-plane correspondences. Higher values give more weight to very planar surfaces.
                double power_planarity = 2.0;
                /// Maximum distance for a point-to-plane correspondence to be considered a valid match.
                double p2p_max_dist = 0.5;
                /// Type of robust loss function (e.g., L2, Cauchy) to use for point-to-plane errors to reduce the impact of outliers.
                LOSS_FUNC p2p_loss_func = LOSS_FUNC::CAUCHY;
                /// The scale parameter (sigma) for the chosen robust loss function for P2P errors.
                double p2p_loss_sigma = 0.1;

                // ----------------------------------------------------------------------------------
                // Radial Velocity Parameters (for Doppler LiDAR)
                // ----------------------------------------------------------------------------------

                /// Whether to use radial velocity measurements from the sensor (if available) as constraints in the optimization.
                bool use_rv = false;
                /// Whether to merge point-to-plane and radial velocity factors into a single, more efficient cost term.
                bool merge_p2p_rv = false;
                /// Maximum radial velocity error (in m/s) to be considered a valid measurement.
                double rv_max_error = 2.0;
                /// Type of robust loss function to use for radial velocity errors.
                LOSS_FUNC rv_loss_func = LOSS_FUNC::CAUCHY;
                /// Inverse covariance (information) value for radial velocity measurements.
                double rv_cov_inv = 0.1;
                /// The scale parameter (sigma) for the chosen robust loss function for radial velocity errors.
                double rv_loss_threshold = 0.05;

                // ----------------------------------------------------------------------------------
                // Optimization and Solver Parameters
                // ----------------------------------------------------------------------------------

                /// Enable/disable verbose logging from the optimization solver.
                bool verbose = false;
                /// Maximum number of iterations for the optimization solver in each step.
                int max_iterations = 5;
                /// Threshold to switch from parallel to sequential processing for small workloads to avoid overhead.
                int sequential_threshold = 500;
                /// Number of threads to use for parallelizable tasks.
                unsigned int num_threads = 4;
                /// Number of frames to wait before adding a frame's points to the map, allowing its pose to converge first.
                int delay_adding_points = 4;
                /// Whether to re-interpolate the entire trajectory using the final optimized state values for higher accuracy output.
                bool use_final_state_value = false;
                /// If true, the ICP loop can terminate early if the state change is below a threshold.
                bool break_icp_early = true;
                /// Whether to remove voxels from the map after a certain number of frames have passed (their 'lifetime').
                bool swf_inside_icp_at_begin = true;
                /// Whether to use a line search algorithm within the Gauss-Newton solver to find a better step size.
                bool use_line_search = false;
                /// Whether to use a line search algorithm within the Gauss-Newton solver to find a better step size.
                bool use_elastic_initialization = false;
                /// Whether to use a line search algorithm within the Gauss-Newton solver to find a better step size.
                double keyframe_translation_threshold_m = 0.0;
                /// Whether to use a line search algorithm within the Gauss-Newton solver to find a better step size.
                double keyframe_rotation_threshold_deg = 0.0;
                /// Whether to use a line search algorithm within the Gauss-Newton solver to find a better step size.
                bool use_pointtopoint_factors = false;

                // ----------------------------------------------------------------------------------
                // IMU Parameters
                // ----------------------------------------------------------------------------------

                /// Magnitude of the gravity vector. A positive value suggests a North-East-Down (NED) or similar z-down coordinate system.
                double gravity = 9.8042; 
                /// Whether to use IMU data to constrain the motion model.
                bool use_imu = false;
                /// Whether to use the accelerometer part of the IMU data (in addition to the gyroscope).
                bool use_accel = false;
                /// Diagonal elements of the measurement noise covariance for the accelerometer ($R_{acc}$).
                Eigen::Matrix<double, 3, 1> r_imu_acc = Eigen::Matrix<double, 3, 1>::Zero();
                /// Diagonal elements of the measurement noise covariance for the gyroscope ($R_{gyro}$).
                Eigen::Matrix<double, 3, 1> r_imu_ang = Eigen::Matrix<double, 3, 1>::Zero();
                /// Initial uncertainty (covariance, $P_0$) for the accelerometer bias.
                Eigen::Matrix<double, 3, 1> p0_bias_accel = Eigen::Matrix<double, 3, 1>::Ones();
                /// Process noise (covariance, $Q$) for the accelerometer bias random walk model (how much it can drift over time).
                Eigen::Matrix<double, 3, 1> q_bias_accel = Eigen::Matrix<double, 3, 1>::Ones();
                /// Initial uncertainty (covariance, $P_0$) for the gyroscope bias.
                double p0_bias_gyro = 0.0001;
                /// Process noise (covariance, $Q$) for the gyroscope bias random walk model.
                double q_bias_gyro = 0.0001;
                /// Type of robust loss function for accelerometer errors.
                std::string acc_loss_func = "CAUCHY";
                /// Scale parameter (sigma) for the accelerometer robust loss function.
                double acc_loss_sigma = 1.0;
                /// Type of robust loss function for gyroscope errors.
                std::string gyro_loss_func = "L2";
                /// Scale parameter (sigma) for the gyroscope robust loss function.
                double gyro_loss_sigma = 1.0;

                // ----------------------------------------------------------------------------------
                // IMU-Map Transformation (Ti2m) Parameters
                // ----------------------------------------------------------------------------------

                /// If true, the IMU-to-Map extrinsic ($T_{i2m}$) is only optimized at the beginning and then held fixed.
                bool Ti2m_init_only = true;
                /// Initial covariance for the $T_{i2m}$ estimation.
                Eigen::Matrix<double, 6, 1> Ti2m_init_cov = Eigen::Matrix<double, 6, 1>::Ones();
                /// Process noise for the IMU-to-Map extrinsic ($T_{mi}$) if it's continuously estimated.
                Eigen::Matrix<double, 6, 1> qg_diag = Eigen::Matrix<double, 6, 1>::Ones();

                // ----------------------------------------------------------------------------------
                // Initial State Priors (for the very first frame)
                // ----------------------------------------------------------------------------------
                /// Programmatically set initial pose mean. Use setInitialPose() to populate this.
                Eigen::Matrix<double, 4, 4> Tm2b_init = Eigen::Matrix<double, 4, 4>::Identity();

                // ----------------------------------------------------------------------------------
                // Map Management
                // ----------------------------------------------------------------------------------

                /// Whether to remove voxels from the map after a certain number of frames have passed (their 'lifetime').
                bool filter_lifetimes = false;
                
            };

            static Options parse_json_options(const std::string& json_path);

            lidarodom(const std::string& json_path);
            ~lidarodom();

            Trajectory trajectory() override;
            RegistrationSummary registerFrame(const DataFrame& frame) override;
            void getResults(const std::string& timestamp) override;
            void initializeInitialPose(const Eigen::Matrix4d& T) override;

        private:
            inline double AngularDistance(const Eigen::Matrix3d& rota, const Eigen::Matrix3d& rotb);
            void sub_sample_frame(std::vector<Point3D>& frame, double size_voxel);
            void grid_sampling(const std::vector<Point3D>& frame, std::vector<Point3D>& keypoints, double size_voxel_subsampling);
            Neighborhood compute_neighborhood_distribution(const ArrayVector3d& points); 
            
            void initializeTimestamp(int index_frame, const DataFrame& const_frame);
            Eigen::Matrix<double, 6, 1> initialize_gravity(const std::vector<steam::IMUData>& imu_data_vec);
            void initializeMotion(int index_frame);
            std::vector<Point3D> initializeFrame(int index_frame, const std::vector<Point3D>& const_frame);
            void updateMap(int index_frame, int update_frame);
            bool icp(int index_frame, std::vector<Point3D>& keypoints, const std::vector<steam::IMUData>& imu_data_vec);

        private:
            Options options_;
            steam::se3::SE3StateVar::Ptr Tb2s_var_ = nullptr;  // robot to sensor transformation as a slam variable

            struct TrajectoryVar {
                TrajectoryVar() = default;
                TrajectoryVar(const steam::traj::Time& t, 
                            const steam::se3::SE3StateVar::Ptr& T,
                            const steam::vspace::VSpaceStateVar<6>::Ptr& w, 
                            const steam::vspace::VSpaceStateVar<6>::Ptr& b, 
                            const steam::se3::SE3StateVar::Ptr& Ti2m)
                    : time(t), Tm2b(T), wb2m_inr(w), imu_biases(b), Ti2m(Ti2m) {}
                steam::traj::Time time;
                steam::se3::SE3StateVar::Ptr Tm2b;
                steam::vspace::VSpaceStateVar<6>::Ptr wb2m_inr;
                steam::vspace::VSpaceStateVar<6>::Ptr imu_biases;
                steam::se3::SE3StateVar::Ptr Ti2m;
            };
            std::vector<TrajectoryVar> trajectory_vars_;
            size_t to_marginalize_ = 0;
            std::map<double, std::pair<Matrix12d, Matrix12d>> interp_mats_;
            steam::SlidingWindowFilter::Ptr sliding_window_filter_;

            Eigen::Vector3d t_prev_ = Eigen::Vector3d::Zero();
            Eigen::Matrix3d r_prev_ = Eigen::Matrix3d::Identity();

            SLAM_REGISTER_ODOMETRY("SLAM_LIDAR_ODOM", lidarodom);
    };
}    // namespace stateestimate
