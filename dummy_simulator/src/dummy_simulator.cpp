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

#include <cstdlib>
#include <ctime>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <std_msgs/ByteMultiArray.h>
#include <nav_msgs/OccupancyGrid.h>
#include <std_srvs/Trigger.h>
#include <visualization_msgs/Marker.h>
#include <dummy_simulator/Belief.h>
#include <dummy_simulator/dummy_simulator.h>

using namespace cv;

namespace dummy_simulator {
DummySimulator::DummySimulator(ros::NodeHandle& n):
  nh(n) {
  return;
}

DummySimulator::~DummySimulator() {
  free(grid_map);
  free(belief);

  fclose(data_fid);
  fclose(experiment_fid);
  return;
}

bool DummySimulator::initialize() {
  if (!loadParameters()) {
    ROS_ERROR("Cannot load all required parameters...\n");
    return false;
  }

  if (!createRosIO()) {
    ROS_ERROR("Cannot create all ROS I/O...\n");
    return false;
  }

  // Random number generator.
  srand(time(NULL));

  // Load the map image.
  // Convert he image to occupancy grid map.
  Mat map_img = imread(map_path, IMREAD_GRAYSCALE);
  convertImgToMap(map_img);

  // Check if the start or the goal location is occupied.
  if (grid_map[robot_location[1]*
      map_img.cols+robot_location[0]] == 1) {
    ROS_ERROR("The start location is infeasible...");
    return false;
  }
  if (grid_map[goal_location[1]*
      map_img.cols+goal_location[0]] == 1) {
    ROS_ERROR("The goal location is infeasible...");
    return false;
  }

  // Initialize the path message.
  robot_path_msg.header.stamp = ros::Time::now();
  robot_path_msg.header.frame_id = fixed_frame_id;
  robot_path_msg.poses.push_back(geometry_msgs::PoseStamped());
  robot_path_msg.poses.back().header.stamp = ros::Time::now();
  robot_path_msg.poses.back().header.frame_id = fixed_frame_id;
  robot_path_msg.poses.back().pose.position.x = (robot_location[0]+0.5)*map_resolution;
  robot_path_msg.poses.back().pose.position.y = (robot_location[1]+0.5)*map_resolution;
  robot_path_msg.poses.back().pose.position.z = 0.0;
  robot_path_msg.poses.back().pose.orientation.x = 0.0;
  robot_path_msg.poses.back().pose.orientation.y = 0.0;
  robot_path_msg.poses.back().pose.orientation.z = 0.0;
  robot_path_msg.poses.back().pose.orientation.w = 1.0;

  belief = (float*)malloc(sizeof(float)*map_height*map_width);
  // Initialize the robot belief based on the robot location.
  //for (int i = 0; i < map_width*map_height; ++i) belief[i] = 0.0f;
  //belief[robot_location[1]*map_width+robot_location[0]] = 1.0f;

  // Initialize the robot belief uniformly.
  float belief_sum = 0.0f;
  for (int i = 0; i < map_width*map_height; ++i) {
    if (grid_map[i] < 1) {
      belief[i] = 1.0f;
      belief_sum += 1.0f;
    } else {
      belief[i] = 0.0f;
    }
  }
  for (int i = 0; i < map_width*map_height; ++i)
    belief[i] /= belief_sum;

  // Get the first measurement.
  getMeasurement(measurement);

  data_fid = fopen("simulation_data", "w");
  experiment_fid = fopen("experiment_data", "a+");

  return true;
}

bool DummySimulator::loadParameters() {
  if (!nh.getParam("map_path", map_path)) return false;
  if (!nh.getParam("map_resolution", map_resolution)) return false;
  if (!nh.getParam("discount_factor", discount_factor)) return false;
  if (!nh.getParam("start_x", robot_location[0])) return false;
  if (!nh.getParam("start_y", robot_location[1])) return false;
  if (!nh.getParam("goal_x", goal_location[0])) return false;
  if (!nh.getParam("goal_y", goal_location[1])) return false;

  nh.param<std::string>(
      "fixed_frame_id", fixed_frame_id, std::string("map"));
  nh.param<std::string>(
      "robot_frame_id", robot_frame_id, std::string("robot"));
  nh.param<double>("timer_freq", robot_timer_freq, 10.0);

  return true;
}

bool DummySimulator::createRosIO() {
  grid_map_pub = nh.advertise<nav_msgs::OccupancyGrid>("grid_map", 1, true);
  belief_pub = nh.advertise<Belief>("belief", 1);
  path_pub = nh.advertise<nav_msgs::Path>("robot_path", 1);
  belief_marker_pub = nh.advertise<
    visualization_msgs::Marker>("belief_marker", 1);
  loc_marker_pub = nh.advertise<
    visualization_msgs::Marker>("location_marker", 1);

  robot_timer = nh.createTimer(
      ros::Duration(1.0/robot_timer_freq),
      &DummySimulator::robotTimerCallback, this);

  // This only works with the pomdp planner.
  //reset_client = nh.serviceClient<
  //  std_srvs::Trigger>("reset_search_tree");
  //reset_client.waitForExistence();

  control_sub = nh.subscribe(
      "control_input", 1, &DummySimulator::controlCallback, this);

  return true;
}

void DummySimulator::controlCallback(
    const std_msgs::ByteConstPtr& u) {

  //// Save data to a file.
  //saveSimulationData(data_fid,
  //    robot_location, belief, u->data, measurement);

  action = u->data;
  printf("received action: %u\n", action);
  // Move the robot according to the input.
  moveRobot(action);

  // Update the prior belief.
  updateBelief(action);

  // Get the measurement after the robot moves.
  getMeasurement(measurement);

  // Update the posterior belief.
  updateBelief(measurement);

  printf("reward_sum: %f\n", reward_sum);
  printf("collision_num: %u\n", collision_num);
  printf("total_steps: %u\n\n", total_steps);

  //if (total_steps >= 400) {
  //  saveExperimentData(experiment_fid);
  //} else if (robot_location[0]==goal_location[0] &&
  //    robot_location[1]==goal_location[1] && action==4) {
  //  saveExperimentData(experiment_fid);
  //}

  return;
}

void DummySimulator::robotTimerCallback(
    const ros::TimerEvent& e) {

  //printf("action: %u\n", action);
  //printf("measurement: %u %u %u %u\n",
  //    measurement[0], measurement[1], measurement[2], measurement[3]);

  // Publish the robot's current location and the belief.
  BeliefPtr belief_msg(new Belief);
  belief_msg->header.stamp = ros::Time::now();
  belief_msg->header.frame_id = fixed_frame_id;

  belief_msg->action = action;

  belief_msg->measurement[0] = measurement[0];
  belief_msg->measurement[1] = measurement[1];
  belief_msg->measurement[2] = measurement[2];
  belief_msg->measurement[3] = measurement[3];

  belief_msg->location[0] = robot_location[0];
  belief_msg->location[1] = robot_location[1];

  belief_msg->belief.resize(map_height*map_width);
  for (int i = 0; i < belief_msg->belief.size(); ++i)
    belief_msg->belief[i] = belief[i];

  belief_pub.publish(belief_msg);

  // Publish the belief of the robot as a marker.
  visualization_msgs::Marker belief_marker;
  belief_marker.header.stamp = ros::Time::now();
  belief_marker.header.frame_id = fixed_frame_id;
  belief_marker.ns = "belief";
  belief_marker.id = 0;
  belief_marker.type = visualization_msgs::Marker::SPHERE_LIST;
  belief_marker.action = visualization_msgs::Marker::ADD;

  belief_marker.pose.position.x = 0.0;
  belief_marker.pose.position.y = 0.0;
  belief_marker.pose.position.z = 0.0;
  belief_marker.pose.orientation.x = 0.0;
  belief_marker.pose.orientation.y = 0.0;
  belief_marker.pose.orientation.z = 0.0;
  belief_marker.pose.orientation.w = 1.0;

  belief_marker.scale.x = map_resolution;
  belief_marker.scale.y = map_resolution;
  belief_marker.scale.z = map_resolution;

  belief_marker.points.resize(map_width*map_height);
  belief_marker.colors.resize(map_width*map_height);
  for (int y = 0, i = 0; y < map_height; ++y) {
    for (int x = 0; x < map_width; ++x, ++i) {
      belief_marker.points[i].x = map_resolution * (x+0.5);
      belief_marker.points[i].y = map_resolution * (y+0.5);
      belief_marker.points[i].z = 0.0f;

      belief_marker.colors[i].r = 1.0;
      belief_marker.colors[i].g = 0.0;
      belief_marker.colors[i].b = 0.0;
      belief_marker.colors[i].a = belief[i];
    }
  }

  belief_marker_pub.publish(belief_marker);

  // Publish the current and goal locations as a marker.
  visualization_msgs::Marker loc_marker;
  loc_marker.header.stamp = ros::Time::now();
  loc_marker.header.frame_id = fixed_frame_id;
  loc_marker.ns = "locations";
  loc_marker.id = 0;
  loc_marker.type = visualization_msgs::Marker::SPHERE_LIST;
  loc_marker.action = visualization_msgs::Marker::ADD;

  loc_marker.pose.position.x = 0.0;
  loc_marker.pose.position.y = 0.0;
  loc_marker.pose.position.z = 0.0;
  loc_marker.pose.orientation.x = 0.0;
  loc_marker.pose.orientation.y = 0.0;
  loc_marker.pose.orientation.z = 0.0;
  loc_marker.pose.orientation.w = 1.0;

  loc_marker.scale.x = map_resolution;
  loc_marker.scale.y = map_resolution;
  loc_marker.scale.z = map_resolution;

  loc_marker.points.resize(2);
  loc_marker.points[0].x = (robot_location[0]+0.5)*map_resolution;
  loc_marker.points[0].y = (robot_location[1]+0.5)*map_resolution;
  loc_marker.points[0].z = 0.0;
  loc_marker.points[1].x = (goal_location[0]+0.5)*map_resolution;
  loc_marker.points[1].y = (goal_location[1]+0.5)*map_resolution;
  loc_marker.points[1].z = 0.0;

  loc_marker.colors.resize(2);
  loc_marker.colors[0].r = 0.0;
  loc_marker.colors[0].g = 0.0;
  loc_marker.colors[0].b = 1.0;
  loc_marker.colors[0].a = 1.0;
  loc_marker.colors[1].r = 0.0;
  loc_marker.colors[1].g = 1.0;
  loc_marker.colors[1].b = 0.0;
  loc_marker.colors[1].a = 1.0;

  loc_marker_pub.publish(loc_marker);

  // Publish the path of the robot.
  robot_path_msg.header.stamp = ros::Time::now();
  //robot_path_msg.header.frame_id = fixed_frame_id;
  robot_path_msg.poses.push_back(geometry_msgs::PoseStamped());
  robot_path_msg.poses.back().header.stamp = ros::Time::now();
  robot_path_msg.poses.back().header.frame_id = fixed_frame_id;
  robot_path_msg.poses.back().pose.position.x = (robot_location[0]+0.5)*map_resolution;
  robot_path_msg.poses.back().pose.position.y = (robot_location[1]+0.5)*map_resolution;
  robot_path_msg.poses.back().pose.position.z = 0.0;
  robot_path_msg.poses.back().pose.orientation.x = 0.0;
  robot_path_msg.poses.back().pose.orientation.y = 0.0;
  robot_path_msg.poses.back().pose.orientation.z = 0.0;
  robot_path_msg.poses.back().pose.orientation.w = 1.0;

  path_pub.publish(robot_path_msg);

  return;
}

void DummySimulator::saveSimulationData(
    FILE* const fid, const int32_t* const location,
    const float* const belief, const uint8_t& action,
    const std::vector<uint8_t>& measurement) {

  for (size_t i = 0; i < map_height*map_width; ++i) {
    fprintf(fid, "%15.8f", belief[i]);
  }

  fprintf(fid, "%15u", location[0]);
  fprintf(fid, "%15u", location[1]);
  fprintf(fid, " %15u", action);

  const uint8_t observation =
    (measurement[3]<<3) + (measurement[2]<<2) +
    (measurement[1]<<1) + measurement[0];
  fprintf(fid, " %15u\n", observation);

  return;
}

void DummySimulator::saveExperimentData(FILE* const fid) {

  srand(time(NULL));

  // Disable the I/O.
  robot_timer.stop();
  control_sub.shutdown();

  // Reset the simulator state.
  // reset the path message.
  robot_path_msg.header.stamp = ros::Time::now();
  robot_path_msg.header.frame_id = fixed_frame_id;
  robot_path_msg.poses.erase(
      robot_path_msg.poses.begin()+1, robot_path_msg.poses.end());

  // Reset the robot position.
  nh.getParam("start_x", robot_location[0]);
  nh.getParam("start_y", robot_location[1]);

  // Reset the robot belief.
  float belief_sum = 0.0f;
  for (int i = 0; i < map_width*map_height; ++i) {
    if (grid_map[i] < 1) {
      belief[i] = 1.0f;
      belief_sum += 1.0f;
    } else {
      belief[i] = 0.0f;
    }
  }
  for (int i = 0; i < map_width*map_height; ++i)
    belief[i] /= belief_sum;

  // Get the first measurement.
  getMeasurement(measurement);

  // Save the experiment data.
  fprintf(fid, "%15.8f%15u%15u\n",
      reward_sum, collision_num, total_steps);
  reward_sum = 0.0f;
  collision_num = 0;
  total_steps = 0;
  step_discount = 1.0f;

  //// Reset the QVTS through service call.
  //std_srvs::Trigger trigger_msg;
  //reset_client.call(trigger_msg);

  // Reenable the I/O.
  robot_timer.start();
  control_sub = nh.subscribe(
      "control_input", 1, &DummySimulator::controlCallback, this);

  return;
}

void DummySimulator::convertImgToMap(const Mat& map_img) {

  // Convert the input image into occupancy grid map.
  Mat binary_img;
  cv::threshold(map_img, binary_img, 250, 1, THRESH_BINARY_INV);

  map_width = map_img.cols;
  map_height = map_img.rows;
  grid_map = (uint8_t*)malloc(
      sizeof(float)*map_height*map_width);
  memcpy(grid_map, binary_img.data,
      sizeof(uint8_t)*map_height*map_width);

  // Publish the map.
  nav_msgs::OccupancyGrid grid_map_msg;
  grid_map_msg.header.seq = 0;
  grid_map_msg.header.stamp = ros::Time::now();
  grid_map_msg.header.frame_id = fixed_frame_id;

  grid_map_msg.info.map_load_time = ros::Time::now();
  grid_map_msg.info.resolution = map_resolution;
  grid_map_msg.info.width = map_width;
  grid_map_msg.info.height = map_height;
  grid_map_msg.info.origin.position.x = 0.0;
  grid_map_msg.info.origin.position.y = 0.0;
  grid_map_msg.info.origin.position.z = 0.0;
  grid_map_msg.info.origin.orientation.x = 0.0;
  grid_map_msg.info.origin.orientation.y = 0.0;
  grid_map_msg.info.origin.orientation.z = 0.0;
  grid_map_msg.info.origin.orientation.w = 1.0;

  grid_map_msg.data.resize(map_height*map_width);
  for (int i = 0; i < grid_map_msg.data.size(); ++i)
    grid_map_msg.data[i] = grid_map[i]<1 ? 0 : 100;

  // Publish the map once.
  grid_map_pub.publish(grid_map_msg);

  return;
}

void DummySimulator::transitionProbability(
    const int32_t& x, const int32_t& y, const uint8_t& u,
    std::vector<float>& trans_prob_naive,
    std::vector<float>& trans_prob) {
  // Initialize the transition probability to all 0's.
  trans_prob.resize(9);
  trans_prob.assign(trans_prob.size(), 0.0f);

  trans_prob_naive.resize(9);
  trans_prob_naive.assign(trans_prob.size(), 0.0f);

  // Fill in the transition probability according to
  // the actions.
  // The 9 actions are:
  // 0 | 1 | 2
  // - - - - -
  // 3 | 4 | 5
  // - - - - -
  // 6 | 7 | 8
  switch (u) {
    case 0 :
      trans_prob[0] = 0.7f; trans_prob[1] = 0.1f;
      trans_prob[3] = 0.1f; trans_prob[4] = 0.1f;
      break;
    case 1 :
      trans_prob[0] = 0.1f; trans_prob[1] = 0.7f;
      trans_prob[2] = 0.1f; trans_prob[4] = 0.1f;
      break;
    case 2 :
      trans_prob[1] = 0.1f; trans_prob[2] = 0.7f;
      trans_prob[4] = 0.1f; trans_prob[5] = 0.1f;
      break;
    case 3 :
      trans_prob[0] = 0.1f; trans_prob[3] = 0.7f;
      trans_prob[4] = 0.1f; trans_prob[6] = 0.1f;
      break;
    case 4 :
      trans_prob[4] = 1.0f;
      break;
    case 5 :
      trans_prob[2] = 0.1f; trans_prob[4] = 0.1f;
      trans_prob[5] = 0.7f; trans_prob[8] = 0.1f;
      break;
    case 6 :
      trans_prob[3] = 0.1f; trans_prob[4] = 0.1f;
      trans_prob[6] = 0.7f; trans_prob[7] = 0.1f;
      break;
    case 7 :
      trans_prob[4] = 0.1f; trans_prob[6] = 0.1f;
      trans_prob[7] = 0.7f; trans_prob[8] = 0.1f;
      break;
    case 8 :
      trans_prob[4] = 0.1f; trans_prob[5] = 0.1f;
      trans_prob[7] = 0.1f; trans_prob[8] = 0.7f;
      break;
  }

  // Copy to the naive transition probability.
  trans_prob_naive = trans_prob;

  // In the case a cell is occupied, the probability
  // of transiting to the cell is shifted to the current
  // the state.
  for (int oy = -1, i = 0; oy < 2; ++oy) {
    for (int ox = -1; ox < 2; ++ox, ++i) {
      int32_t px = x + ox;
      int32_t py = y + oy;
      if (px < 0 || px >= map_width ||
          py < 0 || py >= map_height) {
        trans_prob[4] += trans_prob[i];
        trans_prob[i] = 0.0f;
        continue;
      }
      if (grid_map[py*map_width+px] > 0 && i != 4) {
        trans_prob[4] += trans_prob[i];
        trans_prob[i] = 0.0f;
        continue;
      }
    }
  }

  return;
}

void DummySimulator::moveRobot(const uint8_t& u) {

  // Get the transition probability.
  std::vector<float> trans_prob_naive;
  std::vector<float> trans_prob;
  transitionProbability(
      robot_location[0], robot_location[1],
      u, trans_prob_naive, trans_prob);

  std::vector<float> acc_trans_prob_naive = trans_prob_naive;
  for (int i = 1; i < acc_trans_prob_naive.size(); ++i)
    acc_trans_prob_naive[i] += acc_trans_prob_naive[i-1];

  std::vector<float> acc_trans_prob = trans_prob;
  for (int i = 1; i < acc_trans_prob.size(); ++i)
    acc_trans_prob[i] += acc_trans_prob[i-1];

  // Get a random number within [0, 1),
  // which is used to determine where the robot
  // is after executing the command.
  float val = static_cast<float>(std::rand()) /
    static_cast<float>(RAND_MAX);

  // Check if the robot collides to an obstacle.
  bool check_finished = false;
  for (int oy = -1, i = 0; oy < 2; ++oy) {
    for (int ox = -1; ox < 2; ++ox, ++i) {
      if (val < acc_trans_prob_naive[i]) {
        int32_t px = robot_location[0] + ox;
        int32_t py = robot_location[1] + oy;

        int32_t location_idx = py*map_width + px;
        int32_t goal_idx = goal_location[1]*map_width + goal_location[0];

        if (u==4 && location_idx!=goal_idx ) {
          reward_sum += step_discount*WRONG_STOP;
          total_steps += 1;
        } else if (u==4 && location_idx==goal_idx) {
          reward_sum += step_discount*GOAL;
        } else if (grid_map[location_idx] > 0) {
          reward_sum += step_discount*COLLISION;
          collision_num += 1;
          total_steps += 1;
        } else {
          reward_sum += step_discount*FREE_MOVE;
          total_steps += 1;
        }

        step_discount *= discount_factor;
        check_finished = true;
        break;
      }
    }
    if (check_finished) break;
  }

  // Move the robot according to the model.
  bool execution_finished = false;
  for (int oy = -1, i = 0; oy < 2; ++oy) {
    for (int ox = -1; ox < 2; ++ox, ++i) {
      if (val < acc_trans_prob[i]) {
        robot_location[0] += ox;
        robot_location[1] += oy;
        execution_finished = true;
        break;
      }
    }
    if (execution_finished) break;
  }

  return;
}

void DummySimulator::measurementLikelihood(
    std::vector<float>& meas_prob) {
  // Initialize the probability of all measurements to 0.
  meas_prob.assign(16, 1.0f);

  // Get the true status of the cells to be measured.
  // The four cells to be measured are,
  // x | 0 | x
  // - - - - -
  // 1 | x | 2
  // - - - - -
  // x | 3 | x
  int8_t ox[4] = {0, -1, 1, 0};
  int8_t oy[4] = {-1, 0, 0, 1};
  uint8_t m[4] = {0, 0, 0, 0};
  for (int i = 0; i < 4; ++i) {
    int32_t mx = robot_location[0] + ox[i];
    int32_t my = robot_location[1] + oy[i];
    if (mx < 0 || mx >= map_width ||
        my < 0 || my >= map_height) {
      m[i] = 1;
      continue;
    }
    m[i] = grid_map[my*map_width+mx];
  }

  //printf("true map: %d %d %d %d\n",
  //    m[0], m[1], m[2], m[3]);

  // Get the probability for each measurement.
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 4; ++j) {
      uint8_t z = (i>>j) & 0x01;
      meas_prob[i] *= (m[j]==z) ? 0.98f : 0.02f;
    }
  }

  return;
}

void DummySimulator::getMeasurement(
    std::vector<uint8_t>& meas) {
  // Get the measurement likelihood.
  std::vector<float> meas_prob;
  measurementLikelihood(meas_prob);

  std::vector<float> acc_meas_prob = meas_prob;
  for (int i = 1; i < acc_meas_prob.size(); ++i)
    acc_meas_prob[i] += acc_meas_prob[i-1];

  // Get a random number within [0, 1),
  // which is used to determine which measurement to take.
  float val = static_cast<float>(std::rand()) /
    static_cast<float>(RAND_MAX);
  //printf("random value: %f\n", val);

  // Get the actual measurement.
  uint8_t meas_idx = 0;
  for (int i = 0; i < 16; ++i) {
    if (val < acc_meas_prob[i]) {
      meas_idx = i;
      break;
    }
  }

  meas.assign(4, 0);
  meas[0] = (meas_idx>>0) & 0x01;
  meas[1] = (meas_idx>>1) & 0x01;
  meas[2] = (meas_idx>>2) & 0x01;
  meas[3] = (meas_idx>>3) & 0x01;

  return;
}

void DummySimulator::updateBelief(
    const uint8_t& u) {

  float* new_belief = (float*)malloc(
      sizeof(float)*map_height*map_width);
  for (int i = 0; i < map_height*map_width; ++i)
    new_belief[i] = 0.0f;

  // Loop through all the grids in the map.
  int8_t ox[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
  int8_t oy[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};

  for (int y = 0, idx = 0; y < map_height; ++y) {
    for (int x = 0; x < map_width; ++x, ++idx) {
      // Continue if there is no probability the robot is
      // at the current cell.
      if (belief[idx] == 0.0f) continue;

      // Get the transition probability.
      std::vector<float> trans_prob;
      std::vector<float> trans_prob_naive;
      transitionProbability(x, y, u, trans_prob_naive, trans_prob);
      for (int i = 0; i < 9; ++i) {
        int px = x + ox[i];
        int py = y + oy[i];
        if (px < 0 || px >= map_width ||
            py < 0 || py >= map_height) {
          continue;
        }
        int pidx = py*map_width + px;
        new_belief[pidx] += belief[idx] * trans_prob[i];
      }
    }
  }

  // Normalize the belief for numerical reasons.
  float new_belief_sum = 0.0f;
  for (int i = 0; i < map_height*map_width; ++i)
    new_belief_sum += new_belief[i];

  for (int i = 0; i < map_height*map_width; ++i)
    new_belief[i] /= new_belief_sum;

  // Update the belief.
  memcpy(belief, new_belief, sizeof(float)*map_height*map_width);
  free(new_belief);
  return;
}

void DummySimulator::updateBelief(
    const std::vector<uint8_t>& meas) {

  float* new_belief = (float*)malloc(
      sizeof(float)*map_height*map_width);
  for (int i = 0; i < map_height*map_width; ++i)
    new_belief[i] = 0.0f;
  float new_belief_sum = 0.0f;

  // Loop through all the grids in the map.
  int8_t ox[4] = {0, -1, 1, 0};
  int8_t oy[4] = {-1, 0, 0, 1};

  for (int y = 0, idx = 0; y < map_height; ++y) {
    for (int x = 0; x < map_width; ++x, ++idx) {
      // Continue if there is no probability the robot is
      // at the current cell.
      if (belief[idx] == 0.0f) {
        new_belief[idx] = 0.0f;
        continue;
      }

      // Compute the probability of having the measurement
      // at this cell of the grid.
      float l = 1.0f;
      uint8_t m = 0;
      for (int i = 0; i < 4; ++i) {
        int32_t mx = x + ox[i];
        int32_t my = y + oy[i];
        if (mx < 0 || mx >= map_width ||
            my < 0 || my >= map_height) {
          m = 1;
        } else {
          m = grid_map[my*map_width+mx];
        }
        l *= (m==meas[i]) ? 0.98f : 0.02f;
      }

      new_belief[idx] = l * belief[idx];
      new_belief_sum += new_belief[idx];
    }
  }

  // Normalize the new belief.
  for (int i = 0; i < map_height*map_width; ++i)
    new_belief[i] /= new_belief_sum;

  // Update the belief.
  memcpy(belief, new_belief,
      sizeof(float)*map_height*map_width);
  free(new_belief);

  return;
}

} // End namespace dummy_simulator.
