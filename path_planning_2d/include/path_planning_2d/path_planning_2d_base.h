/*
 *This file is part of path_planning_2d
 *
 *    path_planning_2d is free software: you can redistribute it and/or
 *    modify it under the terms of the GNU General Public License as
 *    published by the Free Software Foundation, either version 3 of
 *    the License, or (at your option) any later version.
 *
 *    path_planning_2d is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with path_planning_2d. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef PATH_PLANNING_2D_BASE_H
#define PATH_PLANNING_2D_BASE_H

#include <cstdint>
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>

#include <ros/ros.h>
#include <dummy_simulator/Belief.h>

namespace path_planning_2d {

class PathPlanning2dBase {
public:

  typedef boost::shared_ptr<PathPlanning2dBase> Ptr;
  typedef boost::shared_ptr<const PathPlanning2dBase> ConstPtr;

  // Constructors.
  PathPlanning2dBase(ros::NodeHandle& n): nh(n) { return; }
  PathPlanning2dBase(const PathPlanning2dBase&) = delete;
  //PathPlanning2dBase operator=(const PathPlanning2dBase&) = delete;

  // Destructor.
  ~PathPlanning2dBase() { return; }

  // Initialize the class.
  virtual bool initialize() = 0;

protected:

  // Load parameters from the parameter server.
  virtual bool loadParameters() = 0;

  // Create ROS interface for the class.
  virtual bool createRosIO() = 0;

  // Callback function for the robot belief.
  virtual void beliefCallback(
      const dummy_simulator::BeliefConstPtr& belief) = 0;

  // Convert an image to occupancy grid map.
  virtual void loadMapFromFile() = 0;

protected:

  // Map related variables.
  std::string map_path;
  uint32_t map_width, map_height;
  double map_resolution;
  uint8_t* grid_map;

  // Goal location
  //std::vector<int> goal;
  int32_t goal[2];

  // Discount factor for the infinite horizon cost function.
  float discount_factor;

  //double total_planning_time = 0.0;
  //uint32_t total_planning_steps = 0;

  // ROS node handle
  ros::NodeHandle nh;

  std::string fixed_frame_id;
  std::string robot_frame_id;

  // ROS publisher and subscribers.
  ros::Publisher control_pub;
  ros::Subscriber belief_sub;

  // File saving planning time.
  FILE* planning_time_fid = nullptr;
};

typedef PathPlanning2dBase::Ptr PathPlanning2dBasePtr;
typedef PathPlanning2dBase::ConstPtr PathPlanning2dBaseConstPtr;
} // End namespace path_planning_2d.



#endif

