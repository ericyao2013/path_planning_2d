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

#include <std_msgs/Byte.h>
#include <visualization_msgs/Marker.h>

#include <path_planning_2d/mdp_path_planning_2d.h>
#include <path_planning_2d/helper_cuda/helper_cuda.h>
#include <path_planning_2d/helper_cuda/helper_device_check.h>

using namespace cv;
using namespace ros;

// External variables.
//extern uint8_t* dev_map_img;
extern uint8_t* dev_map;
extern float* dev_trans_prob;
extern float* dev_stage_cost;
extern float* dev_optimal_cost1;
extern float* dev_optimal_cost2;
extern uint8_t* dev_optimal_action;

// External functions.
void allocateDeviceMemory(const uint32_t, const uint32_t);
void freeDeviceMemory();
//__global__ void cudaConvertImgToMap(
//    uint32_t, uint32_t, uint8_t*, uint8_t*);
__global__ void cudaGenerateModelData(
    uint32_t, uint32_t, uint32_t, uint32_t, uint8_t*, float*, float*);
__global__ void cudaOneStepValueIteration(
    uint32_t, uint32_t, float, float*, float*, float*, float*, uint8_t*);
__global__ void cudaOneStepPolicyEvaluation(
    uint32_t, uint32_t, float, float*, float*, float*, float*, uint8_t*);
__global__ void cudaPolicyImprovment(
    uint32_t, uint32_t, float, float*, float*, float*, uint8_t*);

namespace path_planning_2d {

MdpPathPlanning2d::MdpPathPlanning2d(ros::NodeHandle& n):
  PathPlanning2dBase(n) {
  return;
}

MdpPathPlanning2d::~MdpPathPlanning2d() {
  freeDeviceMemory();
  free(grid_map);
  free(optimal_cost);
  free(optimal_action);
  return;
}

bool MdpPathPlanning2d::initialize() {
  // Load paramters.
  if (!loadParameters()) {
    ROS_WARN("Cannot load all required parameters...");
    return false;
  }

  // Check the GPU Device.
  cudaOutputDeviceProperties();

  // Load a map from the file.
  loadMapFromFile();
  if (grid_map[goal[1]*map_width+goal[0]] > 0) {
    ROS_ERROR("The assigned goal (%d %d) is at a occupied cell...",
        goal[0], goal[1]);
    return false;
  }

  // Allocate memory on device.
  allocateDeviceMemory(map_height, map_width);

  // Upload the map to device.
  checkCudaErrors(cudaMemcpy(dev_map, grid_map,
        sizeof(uint8_t)*map_height*map_width, cudaMemcpyHostToDevice));

  dim3 blockPerGrid(
      static_cast<int>(std::ceil(static_cast<float>(map_width)/8.0)),
      static_cast<int>(std::ceil(static_cast<float>(map_height)/8.0)));
  dim3 threadPerBlock(8, 8);

  // Generate the model data.
  cudaGenerateModelData<<<blockPerGrid, threadPerBlock>>>(
      map_height, map_width, goal[0], goal[1],
      dev_map, dev_trans_prob, dev_stage_cost);
  checkCudaErrors(cudaDeviceSynchronize());

  float* stage_cost = (float*)malloc(sizeof(float)*map_width*map_height*9);
  checkCudaErrors(cudaMemcpy(stage_cost, dev_stage_cost,
        sizeof(float)*map_width*map_height*9, cudaMemcpyDeviceToHost));

  std::printf("Solve MDP with value iteration...\n");
  valueIteration();

  //std::printf("Solve MDP with policy iteration...\n");
  //policyIteration();

  // Download the optimal solution.
  optimal_cost = (float*)malloc(
      sizeof(float)*map_height*map_width);
  optimal_action = (uint8_t*)malloc(
      sizeof(uint8_t)*map_height*map_width);
  checkCudaErrors(cudaMemcpy(optimal_cost, dev_optimal_cost1,
        sizeof(float)*map_height*map_width, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(optimal_action, dev_optimal_action,
        sizeof(uint8_t)*map_height*map_width, cudaMemcpyDeviceToHost));

  // Create ROS I/O.
  if (!createRosIO()) {
    ROS_WARN("Cannot load all ROS I/O...");
    return false;
  }

  // Publish the solution of MDP.
  publishSolution();

  std::printf("Initialization finished...\n");

  return true;
}

bool MdpPathPlanning2d::loadParameters() {
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

bool MdpPathPlanning2d::createRosIO() {
  control_pub = nh.advertise<std_msgs::Byte>("control", 1);
  optimal_cost_pub = nh.advertise<
    visualization_msgs::Marker>("optimal_cost", 1, true);
  optimal_action_pub = nh.advertise<
    visualization_msgs::Marker>("optimal_action", 1, true);

  belief_sub = nh.subscribe("belief", 1,
      &MdpPathPlanning2d::beliefCallback, this);
  return true;
}

void MdpPathPlanning2d::beliefCallback(
    const dummy_simulator::BeliefConstPtr& belief) {

  float belief_mode = 0.0f;
  int32_t belief_mode_idx = 0;

  for (int i = 0; i < belief->belief.size(); ++i) {
    if (belief->belief[i] > belief_mode) {
      belief_mode_idx = i;
      belief_mode = belief->belief[i];
    }
  }

  ros::Time start_time = ros::Time::now();
  uint8_t action = optimal_action[belief_mode_idx];

  std_msgs::Byte action_msg;
  action_msg.data = action;
  control_pub.publish(action_msg);

  return;
}

void MdpPathPlanning2d::loadMapFromFile() {
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

void MdpPathPlanning2d::valueIteration() {
  dim3 blockPerGrid(
      static_cast<int>(std::ceil(static_cast<float>(map_width)/8.0)),
      static_cast<int>(std::ceil(static_cast<float>(map_height)/8.0)));
  dim3 threadPerBlock(8, 8);

  namedWindow("Cost Function", cv::WINDOW_NORMAL);
  namedWindow("Cost Change", cv::WINDOW_NORMAL);

  Mat prev_optimal_cost = Mat::zeros(map_height, map_width, CV_32FC1);
  Mat curr_optimal_cost = Mat::zeros(map_height, map_width, CV_32FC1);

  int total_iterations = 0;
  double cost_inf_norm = 0.0f;
  double max_optimal_cost = 5.0/(1.0-discount_factor);

  do {
    // Perform value iteration 100 times.
    ros::Time start = ros::Time::now();
    for (int i = 0; i < 50; ++i) {
      cudaOneStepValueIteration<<<blockPerGrid, threadPerBlock>>>(
          map_height, map_width, discount_factor,
          dev_trans_prob, dev_stage_cost, dev_optimal_cost1,
          dev_optimal_cost2, dev_optimal_action);
      checkCudaErrors(cudaDeviceSynchronize());
      cudaOneStepValueIteration<<<blockPerGrid, threadPerBlock>>>(
          map_height, map_width, discount_factor,
          dev_trans_prob, dev_stage_cost, dev_optimal_cost2,
          dev_optimal_cost1, dev_optimal_action);
      checkCudaErrors(cudaDeviceSynchronize());
    }
    total_iterations += 100;
    std::printf("time of VI (%d-%d): %f\n", total_iterations-100,
        total_iterations, (ros::Time::now()-start).toSec());

    // Download the latest optimal cost.
    checkCudaErrors(cudaMemcpy(curr_optimal_cost.data, dev_optimal_cost1,
          sizeof(float)*map_height*map_width, cudaMemcpyDeviceToHost));

    // Compute the inf-norm of the cost change.
    Mat optimal_cost_diff;
    absdiff(prev_optimal_cost, curr_optimal_cost, optimal_cost_diff);
    minMaxIdx(optimal_cost_diff, NULL, &cost_inf_norm, NULL, NULL);
    curr_optimal_cost.copyTo(prev_optimal_cost);
    std::printf("Inf-norm: %f\n", cost_inf_norm);

    // Show the curr optimal cost.
    Mat optimal_cost_img = 1.0f - curr_optimal_cost/max_optimal_cost;
    imshow("Cost Function", optimal_cost_img);
    waitKey(5);

    // Show the change of optimal cost.
    Mat optimal_cost_diff_img = optimal_cost_diff / cost_inf_norm;
    imshow("Cost Change", optimal_cost_diff_img);
    waitKey(5);

  } while (cost_inf_norm > max_optimal_cost*1e-3);

  // Save the optimal value function.
  destroyAllWindows();

  return;
}

void MdpPathPlanning2d::policyIteration() {
  dim3 blockPerGrid(
      static_cast<int>(std::ceil(static_cast<float>(map_width)/8.0)),
      static_cast<int>(std::ceil(static_cast<float>(map_height)/8.0)));
  dim3 threadPerBlock(8, 8);

  namedWindow("Cost Function", cv::WINDOW_NORMAL);
  namedWindow("Cost Change", cv::WINDOW_NORMAL);
  namedWindow("Action Change", cv::WINDOW_NORMAL);

  Mat prev_optimal_cost = Mat::zeros(map_height, map_width, CV_32FC1);
  Mat curr_optimal_cost = Mat::zeros(map_height, map_width, CV_32FC1);

  Mat prev_optimal_action = Mat::zeros(map_height, map_width, CV_8UC1);
  Mat curr_optimal_action = Mat::zeros(map_height, map_width, CV_8UC1);

  int total_iterations = 0;
  double cost_inf_norm = 0.0f;
  double max_optimal_cost = 5.0/(1.0-discount_factor);

  do {
    // Policy evaluation.
    ros::Time start = ros::Time::now();
    for (int i = 0; i < 25; ++i) {
      cudaOneStepPolicyEvaluation<<<blockPerGrid, threadPerBlock>>>(
          map_height, map_width, discount_factor,
          dev_trans_prob, dev_stage_cost, dev_optimal_cost1,
          dev_optimal_cost2, dev_optimal_action);
      checkCudaErrors(cudaDeviceSynchronize());
      cudaOneStepPolicyEvaluation<<<blockPerGrid, threadPerBlock>>>(
          map_height, map_width, discount_factor,
          dev_trans_prob, dev_stage_cost, dev_optimal_cost2,
          dev_optimal_cost1, dev_optimal_action);
      checkCudaErrors(cudaDeviceSynchronize());
    }
    total_iterations += 50;
    std::printf("time of VI (%d-%d): %f\n", total_iterations-50,
        total_iterations, (ros::Time::now()-start).toSec());

    // Download the latest optimal cost.
    checkCudaErrors(cudaMemcpy(curr_optimal_cost.data, dev_optimal_cost1,
          sizeof(float)*map_height*map_width, cudaMemcpyDeviceToHost));

    // Compute the inf-norm of the cost change.
    Mat optimal_cost_diff;
    absdiff(prev_optimal_cost, curr_optimal_cost, optimal_cost_diff);
    minMaxIdx(optimal_cost_diff, NULL, &cost_inf_norm, NULL, NULL);
    curr_optimal_cost.copyTo(prev_optimal_cost);
    std::printf("Inf-norm: %f\n", cost_inf_norm);

    // Show the curr optimal cost.
    Mat optimal_cost_img = 1.0f - curr_optimal_cost/max_optimal_cost;
    imshow("Cost Function", optimal_cost_img);
    waitKey(5);

    // Show the change of optimal cost.
    Mat optimal_cost_diff_img = optimal_cost_diff / cost_inf_norm;
    imshow("Cost Change", optimal_cost_diff_img);
    waitKey(5);

    // Policy improvement.
    cudaPolicyImprovment<<<blockPerGrid, threadPerBlock>>>(
        map_height, map_width, discount_factor, dev_trans_prob,
        dev_stage_cost, dev_optimal_cost1, dev_optimal_action);
    checkCudaErrors(cudaDeviceSynchronize());

    // Download the latest optimal action.
    checkCudaErrors(cudaMemcpy(curr_optimal_action.data, dev_optimal_action,
          sizeof(uint8_t)*map_height*map_width, cudaMemcpyDeviceToHost));

    // Compute the change of optimal action.
    Mat optimal_action_diff;
    cv::compare(prev_optimal_action,
        curr_optimal_action, optimal_action_diff, CMP_EQ);
    curr_optimal_action.copyTo(prev_optimal_action);
    printf("# of changed actions: %d\n", countNonZero(1-optimal_action_diff));

    // Show the change of actions.
    imshow("Action Change", (1-optimal_action_diff)*255);
    waitKey(5);

  } while (cost_inf_norm > max_optimal_cost*1e-3);

  destroyAllWindows();

  return;
}

void MdpPathPlanning2d::publishSolution() {

  // Publish the solution as markers.
  visualization_msgs::Marker optimal_cost_marker;
  optimal_cost_marker.header.stamp = ros::Time::now();
  optimal_cost_marker.header.frame_id = fixed_frame_id;
  optimal_cost_marker.ns = "MDP solution";
  optimal_cost_marker.id = 0;
  optimal_cost_marker.type = visualization_msgs::Marker::SPHERE_LIST;
  optimal_cost_marker.action = visualization_msgs::Marker::ADD;
  optimal_cost_marker.pose.position.x = 0.0;
  optimal_cost_marker.pose.position.y = 0.0;
  optimal_cost_marker.pose.position.z = 0.0;
  optimal_cost_marker.pose.orientation.x = 0.0;
  optimal_cost_marker.pose.orientation.y = 0.0;
  optimal_cost_marker.pose.orientation.z = 0.0;
  optimal_cost_marker.pose.orientation.w = 1.0;
  optimal_cost_marker.scale.x = map_resolution;
  optimal_cost_marker.scale.y = map_resolution;
  optimal_cost_marker.scale.z = map_resolution;

  double max_optimal_cost = 5.0/(1.0-discount_factor);
  optimal_cost_marker.points.resize(map_height*map_width);
  optimal_cost_marker.colors.resize(map_height*map_width);
  for (int y = 0, i = 0; y < map_height; ++y) {
    for (int x = 0; x < map_width; ++x, ++i) {
      optimal_cost_marker.points[i].x = map_resolution * (x+0.5);
      optimal_cost_marker.points[i].y = map_resolution * (y+0.5);
      optimal_cost_marker.points[i].z = 0.0;
      optimal_cost_marker.colors[i].r =
        1.0 - optimal_cost[i]/max_optimal_cost;
      optimal_cost_marker.colors[i].g =
        1.0 - optimal_cost[i]/max_optimal_cost;
      optimal_cost_marker.colors[i].b =
        1.0 - optimal_cost[i]/max_optimal_cost;
      optimal_cost_marker.colors[i].a = 1.0;
    }
  }

  visualization_msgs::Marker optimal_action_marker;
  optimal_action_marker.header.stamp = ros::Time::now();
  optimal_action_marker.header.frame_id = fixed_frame_id;
  optimal_action_marker.ns = "MDP solution";
  optimal_action_marker.id = 1;
  optimal_action_marker.type = visualization_msgs::Marker::SPHERE_LIST;
  optimal_action_marker.action = visualization_msgs::Marker::ADD;
  optimal_action_marker.pose.position.x = 0.0;
  optimal_action_marker.pose.position.y = 0.0;
  optimal_action_marker.pose.position.z = 0.0;
  optimal_action_marker.pose.orientation.x = 0.0;
  optimal_action_marker.pose.orientation.y = 0.0;
  optimal_action_marker.pose.orientation.z = 0.0;
  optimal_action_marker.pose.orientation.w = 1.0;
  optimal_action_marker.scale.x = map_resolution;
  optimal_action_marker.scale.y = map_resolution;
  optimal_action_marker.scale.z = map_resolution;

  optimal_action_marker.points.resize(map_height*map_width);
  optimal_action_marker.colors.resize(map_height*map_width);
  for (int y = 0, i = 0; y < map_height; ++y) {
    for (int x = 0; x < map_width; ++x, ++i) {
      optimal_action_marker.points[i].x = map_resolution*(x+0.5);
      optimal_action_marker.points[i].y = map_resolution*(y+0.5);
      optimal_action_marker.points[i].z = 0.0;

      switch (optimal_action[i]) {
        case 0:
          optimal_action_marker.colors[i].r = 0.0;
          optimal_action_marker.colors[i].g = 0.0;
          optimal_action_marker.colors[i].b = 0.0;
          optimal_action_marker.colors[i].a = 1.0;
          break;
        case 1:
          optimal_action_marker.colors[i].r = 0.0;
          optimal_action_marker.colors[i].g = 0.0;
          optimal_action_marker.colors[i].b = 1.0;
          optimal_action_marker.colors[i].a = 1.0;
          break;
        case 2:
          optimal_action_marker.colors[i].r = 0.0;
          optimal_action_marker.colors[i].g = 1.0;
          optimal_action_marker.colors[i].b = 0.0;
          optimal_action_marker.colors[i].a = 1.0;
          break;
        case 3:
          optimal_action_marker.colors[i].r = 0.0;
          optimal_action_marker.colors[i].g = 1.0;
          optimal_action_marker.colors[i].b = 1.0;
          optimal_action_marker.colors[i].a = 1.0;
          break;
        case 4:
          optimal_action_marker.colors[i].r = 1.0;
          optimal_action_marker.colors[i].g = 1.0;
          optimal_action_marker.colors[i].b = 1.0;
          optimal_action_marker.colors[i].a = 1.0;
          break;
        case 5:
          optimal_action_marker.colors[i].r = 1.0;
          optimal_action_marker.colors[i].g = 0.0;
          optimal_action_marker.colors[i].b = 0.0;
          optimal_action_marker.colors[i].a = 1.0;
          break;
        case 6:
          optimal_action_marker.colors[i].r = 1.0;
          optimal_action_marker.colors[i].g = 0.0;
          optimal_action_marker.colors[i].b = 1.0;
          optimal_action_marker.colors[i].a = 1.0;
          break;
        case 7:
          optimal_action_marker.colors[i].r = 1.0;
          optimal_action_marker.colors[i].g = 1.0;
          optimal_action_marker.colors[i].b = 0.0;
          optimal_action_marker.colors[i].a = 1.0;
          break;
        case 8:
          optimal_action_marker.colors[i].r = 1.0;
          optimal_action_marker.colors[i].g = 1.0;
          optimal_action_marker.colors[i].b = 1.0;
          optimal_action_marker.colors[i].a = 1.0;
          break;
      }
    }
  }

  optimal_cost_pub.publish(optimal_cost_marker);
  optimal_action_pub.publish(optimal_action_marker);

  return;
}

} // End namespace path_planning_2d.
