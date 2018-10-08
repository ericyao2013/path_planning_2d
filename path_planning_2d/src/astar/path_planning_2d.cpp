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

#include <cmath>
#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>

#include <std_msgs/Byte.h>

#include <path_planning_2d/astar_path_planning_2d.h>

using namespace cv;
using namespace ros;

namespace path_planning_2d {

AStarPathPlanning2d::AStarPathPlanning2d(ros::NodeHandle& n):
  PathPlanning2dBase(n) {
  return;
}

AStarPathPlanning2d::~AStarPathPlanning2d() {
  free(grid_map);
  free(astar_planner);
  fclose(planning_time_fid);
  return;
}

bool AStarPathPlanning2d::initialize() {
  // Load parameters.
  if (!loadParameters()) {
    ROS_WARN("Cannot load all required parameters...");
    return false;
  }

  // Load a map from the file.
  loadMapFromFile();
  if (grid_map[goal[1]*map_width+goal[0]] > 0) {
    ROS_ERROR("The assigned goal (%d %d) is at a occupied cell...",
        goal[0], goal[1]);
    return false;
  }

  // Initialize the A* planner.
  std::vector<int8_t> compitable_map(map_height*map_width);
  for (size_t i = 0; i < map_height*map_width; ++i)
    compitable_map[i] = 100*grid_map[i];

  Eigen::Vector2d origin(0.0f, 0.0f);
  Eigen::Vector2i dimension(map_width, map_height);

  std::shared_ptr<JPS::OccMapUtil> map_util(new JPS::OccMapUtil);
  map_util->setMap(origin, dimension, compitable_map, map_resolution);

  astar_planner = new JPSPlanner2D(false);
  astar_planner->setMapUtil(map_util);
  astar_planner->updateMap();

  // Create ROS I/O.
  if (!createRosIO()) {
    ROS_WARN("Cannot load all ROS I/O...");
    return false;
  }

  planning_time_fid = fopen("planning_time", "a+");

  return true;
}

bool AStarPathPlanning2d::loadParameters() {
  if (!nh.getParam("map_path", map_path)) return false;
  if (!nh.getParam("goal_x", goal[0])) return false;
  if (!nh.getParam("goal_y", goal[1])) return false;
  if (!nh.getParam("discount_factor", discount_factor)) return false;
  if (!nh.getParam("map_resolution", map_resolution)) return false;

  nh.param<std::string>("fixed_frame_id",
      fixed_frame_id, std::string("map"));
  nh.param<std::string>("robot_frame_id",
      robot_frame_id, std::string("robot"));
  return true;
}

bool AStarPathPlanning2d::createRosIO() {
  control_pub = nh.advertise<std_msgs::Byte>("control", 1);
  belief_sub = nh.subscribe("belief", 1,
      &AStarPathPlanning2d::beliefCallback, this);

  return true;
}

void AStarPathPlanning2d::beliefCallback(
    const dummy_simulator::BeliefConstPtr& belief) {

  float belief_mode = 0.0f;
  int32_t belief_mode_idx = 0;

  for (size_t i = 0; i < belief->belief.size(); ++i) {
    if (belief->belief[i] > belief_mode) {
      belief_mode_idx = i;
      belief_mode = belief->belief[i];
    }
  }

  uint32_t curr_location[2] = {0};
  curr_location[0] = belief_mode_idx%map_width;
  curr_location[1] = belief_mode_idx/map_width;

  if (curr_location[0]==goal[0] && curr_location[1]==goal[1]) {
    std_msgs::Byte action_msg;
    action_msg.data = 4;
    control_pub.publish(action_msg);
    return;
  }

  Eigen::Vector2d start(
      (curr_location[0]+0.5)*map_resolution,
      (curr_location[1]+0.5)*map_resolution);
  Eigen::Vector2d target(
      goal[0]*map_resolution, goal[1]*map_resolution);

  ros::Time start_time = ros::Time::now();
  astar_planner->plan(start, target, 1, false);
  const auto path = astar_planner->getRawPath();

  double planning_time = (ros::Time::now()-start_time).toSec();
  printf("planning time: %f\n", planning_time);

  // Convert the next point in the path into the corresponding action.
  uint32_t next_location[2] = {0};
  next_location[0] = static_cast<uint32_t>(path[1](0)/map_resolution);
  next_location[1] = static_cast<uint32_t>(path[1](1)/map_resolution);
  uint8_t action = (next_location[1]-curr_location[1]+1)*3 +
    (next_location[0]-curr_location[0]+1);

  std_msgs::Byte action_msg;
  action_msg.data = action;
  control_pub.publish(action_msg);

  //fprintf(planning_time_fid, "%15.8f\n", planning_time);

  return;
}

void AStarPathPlanning2d::loadMapFromFile() {
  Mat img = imread(map_path, IMREAD_GRAYSCALE);
  map_height = img.rows;
  map_width = img.cols;

  Mat grid_map_img;
  cv::threshold(img, grid_map_img, 250.0, 1.0, THRESH_BINARY_INV);

  grid_map = (unsigned char*)malloc(
      sizeof(unsigned char)*map_height*map_width);
  memcpy(grid_map, grid_map_img.data,
      sizeof(unsigned char)*map_height*map_width);

  return;
}

} // End namespace path_planning_2d.
