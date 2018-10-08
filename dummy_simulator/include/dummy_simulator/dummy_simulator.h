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

#ifndef DUMMY_SIMULATOR_H
#define DUMMY_SIMULATOR_H

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>

#include <ros/ros.h>
#include <std_msgs/Byte.h>
#include <nav_msgs/Path.h>



namespace dummy_simulator {
class DummySimulator {
public:
  typedef boost::shared_ptr<DummySimulator> Ptr;
  typedef boost::shared_ptr<const DummySimulator> ConstPtr;

  // Constructors.
  DummySimulator(ros::NodeHandle& n);
  DummySimulator(const DummySimulator&) = delete;
  DummySimulator operator=(const DummySimulator&) = delete;

  // Destructors.
  ~DummySimulator();

  // Initialize the class.
  bool initialize();

private:

  // Load parameters from the parameter server.
  bool loadParameters();

  // Create ROS interface for the class.
  bool createRosIO();

  // Callback function for the control input.
  void controlCallback(const std_msgs::ByteConstPtr& u);

  // Callback function for the timer.
  void robotTimerCallback(const ros::TimerEvent& e);

  // Convert an image to 2-D occupancy grid map.
  void convertImgToMap(const cv::Mat& map_img);

  // Generate the transition probability given an action.
  void transitionProbability(
      const int32_t& x, const int32_t& y, const uint8_t& u,
      std::vector<float>& trans_prob_naive,
      std::vector<float>& trans_prob);
  void moveRobot(const uint8_t& u);

  // Generate the measurement likelihood.
  void measurementLikelihood(
      std::vector<float>& meas_prob);
  void getMeasurement(std::vector<uint8_t>& meas);

  // Update the belief state of the robot location.
  void updateBelief(const uint8_t& u);
  void updateBelief(const std::vector<uint8_t>& meas);

  // Save the stepwise simulation data.
  void saveSimulationData(
      FILE* const fid, const int32_t* const location,
      const float* const belief, const uint8_t& action,
      const std::vector<uint8_t>& measurement);
  void saveExperimentData(FILE* const fid);


  // ROS node handle.
  ros::NodeHandle nh;

  // Frame ids.
  std::string fixed_frame_id;
  std::string robot_frame_id;

  // Map related variables.
  std::string map_path;
  double map_resolution;
  int32_t map_width, map_height;
  uint8_t* grid_map;

  // Start and goal location.
  int32_t robot_location[2];
  int32_t goal_location[2];
  uint8_t action = 4;
  std::vector<uint8_t> measurement = {0, 0, 0, 0};
  float* belief;

  // A timer to trigger odom publication.
  double robot_timer_freq;
  ros::Timer robot_timer;

  // History path of the robot.
  nav_msgs::Path robot_path_msg;

  // ROS publishers and subscribers.
  ros::Publisher grid_map_pub;
  ros::Publisher belief_pub;
  ros::Publisher path_pub;
  ros::Publisher belief_marker_pub;
  ros::Publisher loc_marker_pub;
  ros::Subscriber control_sub;

  // File handle saving stepwise data.
  FILE* data_fid = nullptr;
  FILE* experiment_fid = nullptr;

  // Planning statistics.
  enum Reward {
    COLLISION = -2, WRONG_STOP = -2, FREE_MOVE = -1, GOAL = 0};
  float reward_sum = 0.0;
  uint32_t collision_num = 0;
  uint32_t total_steps = 0;
  float discount_factor = 0.95f;
  float step_discount = 1.0f;
  ros::ServiceClient reset_client;
};

typedef DummySimulator::Ptr DummySimulatorPtr;
typedef DummySimulator::ConstPtr DummySimulatorConstPtr;
} // End namespace dummy_simulator.

#endif

