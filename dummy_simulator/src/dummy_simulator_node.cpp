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
#include <dummy_simulator/dummy_simulator.h>

using namespace std;
using namespace ros;
namespace ds = dummy_simulator;

int main(int argc, char** argv) {
  ros::init(argc, argv, "dummy_simulator");
  ros::NodeHandle nh("~");

  ds::DummySimulatorPtr dummy_sim_ptr(new ds::DummySimulator(nh));
  if (!dummy_sim_ptr->initialize()) {
    ROS_ERROR("Cannot initialize the dummy simulator...");
    return -1;
  }

  ros::spin();

  return 0;
}
