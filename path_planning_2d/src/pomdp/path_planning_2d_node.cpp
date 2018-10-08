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

#include <ros/ros.h>
#include <path_planning_2d/pomdp_path_planning_2d.h>

namespace pp2 = path_planning_2d;

int main(int argc, char** argv) {
  ros::init(argc, argv, "path_planner");
  ros::NodeHandle nh("~");

  pp2::PomdpPathPlanning2dPtr pomdp_pp2_ptr(
      new pp2::PomdpPathPlanning2d(nh));
  if (!pomdp_pp2_ptr->initialize()) {
    ROS_ERROR("Cannot initialize the 2-D POMDP path planner...");
    return -1;
  }

  ros::spin();

  return 0;
}

