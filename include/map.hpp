#pragma once

#include <vector>
#include <queue>
#include <tuple>
#include <cmath> // For std::floor, std::max, std::abs
#include <limits> // For std::numeric_limits
#include <algorithm> // For std::max

#include <Eigen/Dense>
#include <robin_map.h>
#include <iomanip> // for set precission


// Your project's custom headers
#include "voxel.hpp"
#include "point.hpp"

namespace stateestimate {

    class Map {
        public:
            // --- Types and Structs ---
            
            // ########################################################################
            // setInitialPose
            // ########################################################################
            // --- Constructors and Basic Setup ---
            Map() = default;

            // ########################################################################
            // setInitialPose
            // ########################################################################
            explicit Map(int default_lifetime): default_lifetime_(default_lifetime) {}

            // ########################################################################
            // pointcloud
            // ########################################################################
            // --- Point Cloud and Size Queries ---
            // Extracts all points from the map into a single vector. Low overhead.
            [[nodiscard]] ArrayVector3d pointcloud() const {
                ArrayVector3d points;
                points.reserve(size());
                for (const auto& voxel : voxel_map_) {
                    for (int i = 0; i < voxel.second.NumPoints(); ++i) points.push_back(voxel.second.points[i]);
                }
                return points;
            }

            // ########################################################################
            // size
            // ########################################################################
            // Returns the total number of points in the map in O(1) time.
            [[nodiscard]] size_t size() const {
                size_t map_size = 0;
                for (auto &voxel : voxel_map_) {
                    map_size += (voxel.second).NumPoints();
                }
                return map_size;
            }
            
            // ########################################################################
            // remove
            // ########################################################################
            // Removes voxels whose representative point is farther than 'distance' from 'location'.
            void remove(const Eigen::Vector3d& location, double distance) {
                std::vector<Voxel> voxels_to_erase;
                const double sq_distance = distance * distance;

                for (auto &pair : voxel_map_) {
                    Eigen::Vector3d pt = pair.second.points[0];
                    if ((pt - location).squaredNorm() > (sq_distance)) {
                        voxels_to_erase.push_back(pair.first);
                    }
                }

                for (auto &voxel : voxels_to_erase) voxel_map_.erase(voxel);
            }

            // ########################################################################
            // update_and_filter_lifetimes
            // ########################################################################
            // Decrements lifetimes of all voxels and removes those that have expired.
            void update_and_filter_lifetimes() {
                std::vector<Voxel> voxels_to_erase;
                for (VoxelHashMap::iterator it = voxel_map_.begin(); it != voxel_map_.end(); it++) {
                    auto &voxel_block = (it.value());
                    voxel_block.life_time -= 1;
                    if (voxel_block.life_time <= 0) voxels_to_erase.push_back(it->first);
                }
                // Sequentially erase
                for (auto &voxel : voxels_to_erase) voxel_map_.erase(voxel);
            }

            // ########################################################################
            // setDefaultLifeTime
            // ########################################################################
            void setDefaultLifeTime(int default_lifetime) { default_lifetime_ = default_lifetime; }

            // ########################################################################
            // clear
            // ########################################################################
            void clear() { voxel_map_.clear(); }

            // ########################################################################
            // setInitialPose
            // ########################################################################
            // --- Neighbor Search ---
            using pair_distance_t = std::tuple<double, Eigen::Vector3d, Voxel>;

            // ########################################################################
            // setInitialPose
            // ########################################################################
            struct Comparator {
                bool operator()(const pair_distance_t& left, const pair_distance_t& right) const {
                    return std::get<0>(left) < std::get<0>(right); // Max-heap: top is largest distance
                }
            };

            // ########################################################################
            // setInitialPose
            // ########################################################################
            void add(const std::vector<Point3D> &points, double voxel_size, int max_num_points_in_voxel,
            double min_distance_points, int min_num_points = 0) {
                for (const auto &point : points)
                add(point.pt, voxel_size, max_num_points_in_voxel, min_distance_points, min_num_points);
            }

            void add(const ArrayVector3d &points, double voxel_size, int max_num_points_in_voxel, double min_distance_points) {
                for (const auto &point : points) add(point, voxel_size, max_num_points_in_voxel, min_distance_points);
            }

            void add(const Eigen::Vector3d &point, double voxel_size, int max_num_points_in_voxel, double min_distance_points,
                    int min_num_points = 0) {
                Voxel voxel_key(
                    static_cast<int32_t>(std::floor(point.x() / voxel_size)),
                    static_cast<int32_t>(std::floor(point.y() / voxel_size)),
                    static_cast<int32_t>(std::floor(point.z() / voxel_size))
                );

                VoxelHashMap::iterator search = voxel_map_.find(voxel_key);
                if (search != voxel_map_.end()) {
                auto &voxel_block = (search.value());

                if (!voxel_block.IsFull()) {
                    double sq_dist_min_to_points = 10 * voxel_size * voxel_size;
                    for (int i(0); i < voxel_block.NumPoints(); ++i) {
                        auto &_point = voxel_block.points[i];
                        double sq_dist = (_point - point).squaredNorm();
                        if (sq_dist < sq_dist_min_to_points) {
                            sq_dist_min_to_points = sq_dist;
                        }
                    }
                    if (sq_dist_min_to_points > (min_distance_points * min_distance_points)) {
                        if (min_num_points <= 0 || voxel_block.NumPoints() >= min_num_points) {
                            voxel_block.AddPoint(point);
                        }
                    }
                }
                voxel_block.life_time = default_lifetime_;
                } else {
                    if (min_num_points <= 0) {
                        VoxelBlock block(max_num_points_in_voxel);
                        block.AddPoint(point);
                        block.life_time = default_lifetime_;
                        voxel_map_[voxel_key] = std::move(block);
                    }
                }
            }
            // ########################################################################
            // setInitialPose
            // ########################################################################
            using priority_queue_t = std::priority_queue<pair_distance_t, std::vector<pair_distance_t>, Comparator>;

            // ########################################################################
            // searchNeighbors
            // ########################################################################
            ArrayVector3d searchNeighbors(const Eigen::Vector3d& point, int nb_voxels_visited, double size_voxel_map,
                                                    int max_num_neighbors, int threshold_voxel_capacity = 1, std::vector<Voxel>* voxels = nullptr) {
                // Clear the optional output vector if it's provided.
                if (voxels != nullptr) voxels->reserve(max_num_neighbors);
                
                // Determine the voxel containing the query point.
                const Voxel center = Voxel::Coordinates(point, size_voxel_map);
                // A max-priority queue to keep track of the K nearest neighbors. The top element is the farthest.
                priority_queue_t priority_queue;
                // The squared distance to the farthest neighbor found so far.
                double max_dist_sq = std::numeric_limits<double>::infinity();

                // A helper lambda to process all points within a single voxel.
                auto process_voxel = [&](const Voxel& voxel) {
                    auto search = voxel_map_.find(voxel);
                    // Skip if the voxel doesn't exist or is too sparse.
                    if (search == voxel_map_.end() || search->second.NumPoints() < threshold_voxel_capacity) {
                        return;
                    }
                    const auto& block = search->second;

                    // Iterate through all points in the voxel and update the KNN priority queue.
                    for (const auto& neighbor : block.points) {
                        double dist_sq = (neighbor - point).squaredNorm();
                        // If the queue isn't full, just add the new point.
                        if (priority_queue.size() < static_cast<size_t>(max_num_neighbors)) {
                            priority_queue.emplace(dist_sq, neighbor, voxel);
                            // If the queue just became full, update the max distance.
                            if (priority_queue.size() == static_cast<size_t>(max_num_neighbors)) {
                                max_dist_sq = std::get<0>(priority_queue.top());
                            }
                        } else if (dist_sq < max_dist_sq) {
                            // If the new point is closer than the farthest one in the queue, replace it.
                            priority_queue.pop();
                            priority_queue.emplace(dist_sq, neighbor, voxel);
                            max_dist_sq = std::get<0>(priority_queue.top());
                        }
                    }
                };

                const double half_size = size_voxel_map / 2.0;

                // --- Main Search Loop: Traverse concentric shells of voxels ---
                for (int d = 0; d <= nb_voxels_visited; ++d) {
                    // --- Pruning Strategy 1: Shell-level early exit ---
                    // Calculate the minimum possible distance to the current shell of voxels.
                    double shell_min_dist = (d > 0 ? (d - 1) * size_voxel_map : 0.0);
                    // If this minimum distance is greater than our farthest neighbor, we can stop searching.
                    if (shell_min_dist * shell_min_dist > max_dist_sq) break;

                    // Iterate through the voxels forming the surface of a cube with radius 'd'.
                    for (int dx = -d; dx <= d; ++dx) {
                        for (int dy = -d; dy <= d; ++dy) {
                            for (int dz = -d; dz <= d; ++dz) {
                                // Process only the boundary of the cube to avoid redundant checks.
                                if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != d) continue;

                                Voxel voxel{center.x + dx, center.y + dy, center.z + dz};

                                // --- Pruning Strategy 2: Voxel-level distance check ---
                                // Calculate the minimum squared distance from the query point to this voxel's bounding box.
                                double vx_center = voxel.x * size_voxel_map + half_size;
                                double vy_center = voxel.y * size_voxel_map + half_size;
                                double vz_center = voxel.z * size_voxel_map + half_size;

                                double dx_min = std::max(0.0, std::abs(point.x() - vx_center) - half_size);
                                double dy_min = std::max(0.0, std::abs(point.y() - vy_center) - half_size);
                                double dz_min = std::max(0.0, std::abs(point.z() - vz_center) - half_size);
                                double voxel_min_dist_sq = dx_min * dx_min + dy_min * dy_min + dz_min * dz_min;

                                // If the closest point in this voxel is farther than our farthest neighbor, skip the whole voxel.
                                if (voxel_min_dist_sq > max_dist_sq) continue;

                                process_voxel(voxel);
                            }
                        }
                    }
                }

                // --- Result Extraction ---
                // Copy the results from the priority queue into an output vector.
                const auto size = priority_queue.size();
                ArrayVector3d closest_neighbors(size);
                if (voxels) voxels->resize(size);

                // Popping from the max-heap gives elements from farthest to nearest, so we fill the output vector backwards.
                for (size_t i = size; i > 0; --i) {
                    closest_neighbors[i - 1] = std::get<1>(priority_queue.top());
                    if (voxels) (*voxels)[i - 1] = std::get<2>(priority_queue.top());
                    priority_queue.pop();
                }
                return closest_neighbors;
            }

            // ########################################################################
            // dumpResult
            // ########################################################################
            void getMap(std::ostream& os, int precision = 12) const {
                os << std::fixed << std::setprecision(precision);
                for (const auto& [_, block] : voxel_map_) {
                    for (const auto& point : block.points) {
                        os << point.x() << " " << point.y() << " " << point.z() << "\n";
                    }
                }
            }

        private:
            // --- Private Members ---
            VoxelHashMap voxel_map_;
            int default_lifetime_ = 20;
            
    };
}
    