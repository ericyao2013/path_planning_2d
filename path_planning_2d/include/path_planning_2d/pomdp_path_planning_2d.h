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

#ifndef POMDP_PATH_PLANNING_2D_H
#define POMDP_PATH_PLANNING_2D_H

#define RED     "\x1B[31m"
#define GREEN   "\x1B[32m"
#define YELLOW  "\x1B[33m"
#define BLUE    "\x1B[34m"
#define MAGENTA "\x1B[35m"
#define CYAN    "\x1B[36m"
#define WHITE   "\x1B[37m"
#define RESET   "\x1B[0m"

#include <std_srvs/Trigger.h>
#include "path_planning_2d_base.h"
#include "search_tree.h"

namespace path_planning_2d {

class PomdpPathPlanning2d : public PathPlanning2dBase {
public:
  typedef boost::shared_ptr<PomdpPathPlanning2d> Ptr;
  typedef boost::shared_ptr<const PomdpPathPlanning2d> ConstPtr;

  // Constructors.
  PomdpPathPlanning2d(ros::NodeHandle& n);
  PomdpPathPlanning2d(const PomdpPathPlanning2d&) = delete;
  PomdpPathPlanning2d operator=(const PomdpPathPlanning2d&) = delete;

  // Destructor.
  ~PomdpPathPlanning2d();

  // Initialize the class.
  virtual bool initialize();

private:

  // Load parameters from the parameter server.
  virtual bool loadParameters();

  // Create ROS interface for the class.
  virtual bool createRosIO();

  // Callback function for odometry msgs.
  virtual void beliefCallback(
      const dummy_simulator::BeliefConstPtr& belief);

  // Convert an image to occupancy grid map.
  // The function also checks if goal is at a occupied cell.
  virtual void loadMapFromFile();

  // Callback function for save data service.
  bool saveDataCallback(
      std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res);

  // Callback function for resetting online search tree.
  bool resetSearchTreeCallback(
      std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res);

  // Whether the offline data should be read from files.
  bool read_from_file = false;

  // Tree structure for online search.
  SearchTree* search_tree = nullptr;
  int32_t max_search_tree_depth = 5;
  int32_t max_online_iteration = 5;

  ros::ServiceServer save_data_server;
  ros::ServiceServer reset_search_tree_server;
};

typedef PomdpPathPlanning2d::Ptr PomdpPathPlanning2dPtr;
typedef PomdpPathPlanning2d::ConstPtr PomdpPathPlanning2dConstPtr;
} // End namespace path_planning_2d.



#endif


