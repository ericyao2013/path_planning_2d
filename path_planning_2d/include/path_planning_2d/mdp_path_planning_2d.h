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

#ifndef MDP_PATH_PLANNING_2D_H
#define MDP_PATH_PLANNING_2D_H


#include "path_planning_2d_base.h"

namespace path_planning_2d {

class MdpPathPlanning2d : public PathPlanning2dBase {
public:
  typedef boost::shared_ptr<MdpPathPlanning2d> Ptr;
  typedef boost::shared_ptr<const MdpPathPlanning2d> ConstPtr;

  // Constructors.
  MdpPathPlanning2d(ros::NodeHandle& n);
  MdpPathPlanning2d(const MdpPathPlanning2d&) = delete;
  MdpPathPlanning2d operator=(const MdpPathPlanning2d&) = delete;

  // Destructor.
  ~MdpPathPlanning2d();

  // Initialize the class.
  virtual bool initialize();

private:

  // Load parameters from the parameter server.
  virtual bool loadParameters();

  // Create ROS interface for the class.
  virtual bool createRosIO();

  // Callback function for belief msgs.
  virtual void beliefCallback(
      const dummy_simulator::BeliefConstPtr& belief);

  // Convert an image to occupancy grid map.
  // The function also checks if goal is at a occupied cell.
  virtual void loadMapFromFile();

  // Publish the optimal cost and optimal action.
  void publishSolution();

  // Solve the MDP using value iteration.
  void valueIteration();

  // Solve the MDP using policy iteration.
  void policyIteration();


  // Optimal solution for the MDP.
  float* optimal_cost;
  uint8_t* optimal_action;

  // ROS publisher and subscribers.
  ros::Publisher optimal_cost_pub;
  ros::Publisher optimal_action_pub;
};

typedef MdpPathPlanning2d::Ptr MdpPathPlanning2dPtr;
typedef MdpPathPlanning2d::ConstPtr MdpPathPlanning2dConstPtr;
} // End namespace path_planning_2d.



#endif

