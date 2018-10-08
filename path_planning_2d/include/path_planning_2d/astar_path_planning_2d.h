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

#ifndef ASTAR_PATH_PLANNING_2D_H
#define ASTAR_PATH_PLANNING_2D_H

#include <jps_planner/jps_planner/jps_planner.h>
#include "path_planning_2d_base.h"

namespace path_planning_2d {

class AStarPathPlanning2d: public PathPlanning2dBase {
public:
  typedef boost::shared_ptr<AStarPathPlanning2d> Ptr;
  typedef boost::shared_ptr<const AStarPathPlanning2d> ConstPtr;

  // Constructors
  AStarPathPlanning2d(ros::NodeHandle& n);
  AStarPathPlanning2d(const AStarPathPlanning2d&) = delete;
  AStarPathPlanning2d operator=(const AStarPathPlanning2d&) = delete;

  // Destructor.
  ~AStarPathPlanning2d();

  // Initialize the class.
  bool initialize();

private:

  // Load paramters from the parameter server.
  virtual bool loadParameters();

  // Create ROS interface for the class.
  virtual bool createRosIO();

  // Callback function for the belief msgs.
  virtual void beliefCallback(
      const dummy_simulator::BeliefConstPtr& belief);

  // Convert an image to occupancy grid map.
  // The function also checks if goal is at a occupied cell.
  virtual void loadMapFromFile();

  // A* path planner.
  JPSPlanner2D* astar_planner = nullptr;

  // ROS publisher and subscribers.
  ros::Publisher optimal_action_pub;
};

typedef AStarPathPlanning2d::Ptr AStarPathPlanning2dPtr;
typedef AStarPathPlanning2d::ConstPtr AStarPathPlanning2dConstPtr;
} // End namespace path_planning_2d.

#endif
