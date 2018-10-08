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

#include <cstdio>
#include <cmath>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <std_msgs/Byte.h>
#include <path_planning_2d/helper_cuda/helper_cuda.h>
#include <path_planning_2d/helper_cuda/helper_device_check.h>
#include <path_planning_2d/pomdp_path_planning_2d.h>

using namespace std;
using namespace cv;

// External functions.
void allocateDeviceMemoryOfModel(
    const uint32_t, const uint32_t);
void freeDeviceMemoryOfModel();
void generateModelData(
    const uint32_t, const uint32_t, const uint8_t* const, const int32_t* const);
void saveModelDataToFile(const uint32_t, const uint32_t);
bool loadModelDataFromFile(const uint32_t, const uint32_t);

void allocateDeviceMemoryOfFIB(
    const uint32_t, const uint32_t);
void freeDeviceMemoryOfFIB();
void fastInformedBound(
    const uint32_t, const uint32_t, const float);
void saveFibDataToFile(const uint32_t, const uint32_t);
bool loadFibDataFromFile(const uint32_t, const uint32_t);

void allocateDeviceMemoryOfPBVI(
    const uint32_t, const uint32_t, const uint32_t);
void freeDeviceMemoryOfPBVI();
void pointBasedValueIteration(
    const uint32_t, const uint32_t, const vector<float>&, const float);
void savePbviDataToFile(const uint32_t, const uint32_t);
bool loadPbviDataFromFile(const uint32_t, const uint32_t);


namespace path_planning_2d {

PomdpPathPlanning2d::PomdpPathPlanning2d(ros::NodeHandle& n):
  PathPlanning2dBase(n) {
  return;
}

PomdpPathPlanning2d::~PomdpPathPlanning2d() {

  freeDeviceMemoryOfModel();
  freeDeviceMemoryOfFIB();
  freeDeviceMemoryOfPBVI();
  free(grid_map);

  delete search_tree;

  fclose(planning_time_fid);

  return;
}

bool PomdpPathPlanning2d::initialize() {

  // Load parameters.
  if (!loadParameters()) {
    ROS_WARN("Cannot load all required parameters...");
    return false;
  }

  // Check the GPU Device.
  //cudaOutputDeviceProperties();

  // Load a map from the file.
  loadMapFromFile();
  if (grid_map[goal[1]*map_width+goal[0]] > 0) {
    ROS_ERROR("The assigned goal (%d %d) is at a occupied cell...",
        goal[0], goal[1]);
    return false;
  }

  // Generate the initial belief assuming uniform
  // distribution over the free cells.
  float sum = 0.0f;
  for (size_t i = 0; i < map_height*map_width; ++i)
    sum += 1.0f - grid_map[i];

  vector<float> initial_belief(map_height*map_width);
  for (size_t i = 0; i < initial_belief.size(); ++i)
    initial_belief[i] = (1.0f-grid_map[i]) / sum;

  if (!read_from_file) {
    // Generate model data.
    printf(WHITE "Generating model data..." RESET "\n");
    allocateDeviceMemoryOfModel(map_height, map_width);
    generateModelData(map_height, map_width, grid_map, goal);

    // Compute the upper bound using FIB.
    printf(WHITE "Fast Informed Bound (FIB)..." RESET "\n");
    allocateDeviceMemoryOfFIB(map_height, map_width);
    fastInformedBound(map_height, map_width, discount_factor);

    // Compute the lower bound using PBVI.
    printf(WHITE "Point based Value Iteration (PBVI)..." RESET "\n");
    const uint32_t belief_set_size = 500;
    allocateDeviceMemoryOfPBVI(map_height, map_width, belief_set_size);
    pointBasedValueIteration(
        map_height, map_width, initial_belief, discount_factor);

  } else {
    // Load model data.
    printf(WHITE "Load model data..." RESET "\n");
    allocateDeviceMemoryOfModel(map_height, map_width);
    if (!loadModelDataFromFile(map_height, map_width)) return false;

    // Compute the upper bound using FIB.
    printf(WHITE "Load Fast Informed Bound (FIB) data..." RESET "\n");
    allocateDeviceMemoryOfFIB(map_height, map_width);
    if (!loadFibDataFromFile(map_height, map_width)) return false;

    // Compute the lower bound using PBVI.
    printf(WHITE "Load Point based Value Iteration (PBVI) data..." RESET "\n");
    const uint32_t belief_set_size = 500;
    allocateDeviceMemoryOfPBVI(map_height, map_width, belief_set_size);
    if (!loadPbviDataFromFile(map_height, map_width)) return false;
  }

  // Set the static variables for online tree search classes.
  QNode::height = map_height;
  QNode::width = map_width;
  QNode::gamma = discount_factor;

  VNode::height = map_height;
  VNode::width = map_width;
  VNode::gamma = discount_factor;

  SearchTree::height = map_height;
  SearchTree::width = map_width;

  // Create ROS I/O.
  if (!createRosIO()) {
    ROS_WARN("Cannot load all ROS I/O");
    return false;
  }

  planning_time_fid = fopen("planning_time", "a+");

  return true;
}

bool PomdpPathPlanning2d::loadParameters() {
  if (!nh.getParam("map_path", map_path)) return false;
  if (!nh.getParam("goal_x", goal[0])) return false;
  if (!nh.getParam("goal_y", goal[1])) return false;
  if (!nh.getParam("discount_factor", discount_factor)) return false;
  if (!nh.getParam("map_resolution", map_resolution)) return false;
  if (!nh.getParam("read_data_from_file", read_from_file)) return false;
  if (!nh.getParam("max_search_tree_depth", max_search_tree_depth)) return false;
  if (!nh.getParam("max_online_iteration", max_online_iteration)) return false;

  nh.param<string>("fixed_frame_id",
      fixed_frame_id, string("map"));
  nh.param<string>("robot_frame_id",
      robot_frame_id, string("robot"));

  return true;
}

bool PomdpPathPlanning2d::createRosIO() {
  control_pub = nh.advertise<std_msgs::Byte>("control", 1);
  belief_sub = nh.subscribe("belief", 1,
      &PomdpPathPlanning2d::beliefCallback, this);

  save_data_server = nh.advertiseService(
      "save_data", &PomdpPathPlanning2d::saveDataCallback, this);
  reset_search_tree_server = nh.advertiseService(
      "reset_search_tree", &PomdpPathPlanning2d::resetSearchTreeCallback, this);

  return true;
}

void PomdpPathPlanning2d::beliefCallback(
    const dummy_simulator::BeliefConstPtr& msg) {

  ros::Time start;

  const uint8_t action = msg->action;
  const uint8_t observation =
    (msg->measurement[3]<<3) + (msg->measurement[2]<<2) +
    (msg->measurement[1]<<1) + msg->measurement[0];
  const float* const belief = msg->belief.data();

  ros::Time start_time = ros::Time::now();

  if (search_tree == nullptr) {
    search_tree = new SearchTree(belief);
  } else {
    search_tree->update(action, observation);
  }

  // Expand the search tree.
  uint8_t update_counter = 0;
  while (search_tree->getDepth() < max_search_tree_depth &&
      update_counter++ < max_online_iteration) {
    search_tree->expand();
  }

  // Find the action for the next step.
  uint8_t new_action = 0;
  float new_reward = 0.0f;
  search_tree->getOptimalAction(new_action, new_reward);

  double planning_time = (ros::Time::now()-start_time).toSec();
  printf("planning time: %f\n", planning_time);
  printf("Search tree depth: %u\n", search_tree->getDepth());
  //fprintf(planning_time_fid, "%f\n", planning_time);

  // Publish the new action.
  std_msgs::BytePtr control_msg(new std_msgs::Byte);
  control_msg->data = new_action;
  control_pub.publish(control_msg);

  return;
}

void PomdpPathPlanning2d::loadMapFromFile() {
  Mat img = imread(map_path, IMREAD_GRAYSCALE);
  map_height = img.rows;
  map_width = img.cols;

  Mat grid_map_img;
  threshold(img, grid_map_img, 250.0, 1.0, THRESH_BINARY_INV);

  grid_map = (unsigned char*)malloc(
      sizeof(unsigned char)*map_height*map_width);
  memcpy(grid_map, grid_map_img.data,
      sizeof(unsigned char)*map_height*map_width);

  return;
}

bool PomdpPathPlanning2d::saveDataCallback(
    std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res) {

  printf(BLUE "Saving model data...\n" RESET);
  saveModelDataToFile(map_height, map_width);

  printf(BLUE "Saving FIB data...\n" RESET);
  saveFibDataToFile(map_height, map_width);

  printf(BLUE "Saving PBVI data...\n" RESET);
  savePbviDataToFile(map_height, map_width);

  res.success = true;
  return true;
}

bool PomdpPathPlanning2d::resetSearchTreeCallback(
    std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res) {

  delete search_tree;
  search_tree = nullptr;

  return true;
}
} // End namespace path_planning_2d.
