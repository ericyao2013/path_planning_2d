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

#include <stdio.h>
#include <stdint.h>
#include <float.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <path_planning_2d/helper_cuda/helper_cuda.h>

// Map of the environment.
uint8_t* dev_map;
// Transition probability P(xi, u, xj).
float* dev_trans_prob;
// Stage cost g(xi, u, xj);
float* dev_stage_cost;
// Optimal cost(cost-to-go) at each state J(x).
// The two copies are for the ease of value iteration,
// used as cost for previous step and current step respectively.
float* dev_optimal_cost1;
float* dev_optimal_cost2;
// Optimal action to take at each state u(x).
uint8_t* dev_optimal_action;

void allocateDeviceMemory(
    const uint32_t height, const uint32_t width) {
  checkCudaErrors(cudaMalloc(
        (void**)(&dev_map), sizeof(uint8_t)*height*width));
  checkCudaErrors(cudaMalloc(
        (void**)(&dev_trans_prob), sizeof(float)*height*width*9*9));
  checkCudaErrors(cudaMalloc(
        (void**)(&dev_stage_cost), sizeof(float)*height*width*9));
  checkCudaErrors(cudaMalloc(
        (void**)(&dev_optimal_cost1), sizeof(float)*height*width));
  checkCudaErrors(cudaMalloc(
        (void**)(&dev_optimal_cost2), sizeof(float)*height*width));
  checkCudaErrors(cudaMalloc(
        (void**)(&dev_optimal_action), sizeof(uint8_t)*height*width));

  // FIXME: It is wrong to initialize float array with memset.
  checkCudaErrors(cudaMemset(
        dev_optimal_cost1, 0.0, sizeof(float)*height*width));
  checkCudaErrors(cudaMemset(
        dev_optimal_cost2, 0.0, sizeof(float)*height*width));
  checkCudaErrors(cudaMemset(
        dev_optimal_action, 0, sizeof(uint8_t)*height*width));

  return;
}

void freeDeviceMemory() {
  checkCudaErrors(cudaFree(dev_map));
  checkCudaErrors(cudaFree(dev_trans_prob));
  checkCudaErrors(cudaFree(dev_stage_cost));
  checkCudaErrors(cudaFree(dev_optimal_cost1));
  checkCudaErrors(cudaFree(dev_optimal_cost2));
  checkCudaErrors(cudaFree(dev_optimal_action));
  return;
}

__device__
void  cudaTransitionProbability(
    uint8_t u, uint8_t* map,
    float* trans_prob, float* trans_prob_naive) {
  // It is assumed that the elements in `trans_prob`
  // are set to 0.0 by default.

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

  // In the case the current location is occupied,
  // the robot is trapped location for sure. This is
  // not important for executing the control since
  // such case never happens.
  if (map[4] == 1) {
    for (uint8_t i = 0; i < 9; ++i) trans_prob[i] = 0.0f;
    trans_prob[4] = 1.0f;
  }

  // Copy the results to the navie transition probability.
  memcpy(trans_prob_naive, trans_prob, sizeof(float)*9);

  // In the case a cell is occupied, the probability
  // of transiting to the cell is shifted to the current
  // the state.
  for (uint8_t i = 0; i < 9; ++i) {
    if (map[i] == 1 && i != 4) {
      trans_prob[4] += trans_prob[i];
      trans_prob[i] = 0.0f;
    }
  }

  return;
}

__device__
void cudaStageCost(uint32_t x, uint32_t y,
    uint32_t gx, uint32_t gy, uint8_t* map,
    float* trans_prob_naive, float* stage_cost) {

  // Generate the reward matrix based on the local map.
  float map_cost[9] = {0.0f};
  for (uint8_t i = 0; i < 9; ++i) {
    if (map[i] == 1) map_cost[i] = 2.0f;
    else map_cost[i] = 1.0f;
  }

  // Compute the reward based on the transition probability
  // of each action.
  for (uint8_t u = 0; u < 9; ++u) {
    for (uint8_t i = 0; i < 9; ++i)
      stage_cost[u] += map_cost[i]*trans_prob_naive[9*u+i];
  }

  stage_cost[4] = (x!=gx || y!=gy) ? 2.0f : 0.0f;
}

__global__
void cudaGenerateModelData(
    uint32_t height, uint32_t width, uint32_t gx, uint32_t gy,
    uint8_t* map, float* trans_prob, float* stage_cost) {

  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  int idx = y*width + x;
  // Check if this point is outside the range.
  if (x < 0 || x >= width || y < 0 || y >= height) return;

  // Crop the map around the cell of interest.
  // If a local map is outside the range, it will be padded
  // with occupied grids to represent the boundary of the map.
  uint8_t local_map[9];
  for (int8_t oy = -1, i = 0; oy < 2; ++oy) {
    for (int8_t ox = -1; ox < 2; ++ox, ++i) {
      int nx = x + ox; int ny = y + oy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height)
        local_map[i] = 1.0;
      else local_map[i] = map[ny*width+nx];
    }
  }

  // Generate the transition probability.
  float local_trans_prob[81] = {0.0f};
  float local_trans_prob_naive[81] = {0.0f};
  for (uint8_t u = 0; u < 9; ++u)
    cudaTransitionProbability(
        u, local_map, local_trans_prob+u*9, local_trans_prob_naive+u*9);
  memcpy(trans_prob+idx*81, local_trans_prob, sizeof(float)*81);

  // Generate the stage costs.
  float local_stage_cost[9] = {0.0f};
  cudaStageCost(x, y, gx, gy, local_map,
      local_trans_prob_naive, local_stage_cost);
  memcpy(stage_cost+idx*9, local_stage_cost, sizeof(float)*9);

  return;
}

__global__
void cudaOneStepValueIteration(
    uint32_t height, uint32_t width, float gamma,
    float* trans_prob, float* stage_cost,
    float* prev_optimal_cost, float* curr_optimal_cost,
    uint8_t* optimal_action) {

  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  int idx = y*width + x;
  // Check if this point is outside the range.
  if (x < 0 || x >= width || y < 0 || y >= height) return;

  // Transfer the data from global memory to local.
  float local_trans_prob[81] = {0.0f};
  float local_stage_cost[9] = {0.0f};
  float local_cost_to_go[9] = {0.0f};

  memcpy(local_trans_prob, trans_prob+idx*81, sizeof(float)*81);
  memcpy(local_stage_cost, stage_cost+idx*9, sizeof(float)*9);
  for (int8_t oy = -1, i = 0; oy < 2; ++oy) {
    for (int8_t ox = -1; ox < 2; ++ox, ++i) {
      int nx = x + ox; int ny = y + oy;
      if (nx >= 0 && nx < width && ny >= 0 && ny < height)
        local_cost_to_go[i] = prev_optimal_cost[ny*width+nx];
    }
  }

  // Update the optimal cost.
  float opt_cost = FLT_MAX;
  uint8_t opt_action = 0;

  for (uint8_t u = 0; u < 9; ++u) {

    float* tp = local_trans_prob + u*9;
    float cost = local_stage_cost[u];
    for (uint8_t i = 0; i < 9; ++i)
      cost += gamma*tp[i]*local_cost_to_go[i];

    if (cost < opt_cost) {
      opt_cost = cost;
      opt_action = u;
    }
  }

  curr_optimal_cost[idx] = opt_cost;
  optimal_action[idx] = opt_action;

  return;
}

__global__
void cudaOneStepPolicyEvaluation(
    uint32_t height, uint32_t width, float gamma,
    float* trans_prob, float* stage_cost,
    float* prev_optimal_cost, float* curr_optimal_cost,
    uint8_t* optimal_action) {

  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  int idx = y*width + x;
  // Check if this point is outside the range.
  if (x < 0 || x >= width || y < 0 || y >= height) return;

  // Transfer the data from global memory to local.
  uint8_t u = optimal_action[idx];
  float local_trans_prob[9] = {0.0f};
  //float local_stage_cost[9] = {0.0f};
  float local_stage_cost = 0.0f;
  float local_cost_to_go[9] = {0.0f};

  memcpy(local_trans_prob, trans_prob+idx*81+u*9, sizeof(float)*9);
  //memcpy(local_stage_cost, stage_cost+idx*81+u*9, sizeof(float)*9);
  local_stage_cost = stage_cost[9*idx+u];
  for (int8_t oy = -1, i = 0; oy < 2; ++oy) {
    for (int8_t ox = -1; ox < 2; ++ox, ++i) {
      int nx = x + ox; int ny = y + oy;
      if (nx >= 0 && nx < width && ny >= 0 && ny < height)
        local_cost_to_go[i] = prev_optimal_cost[ny*width+nx];
    }
  }

  // Policy evaluation.
  float cost = local_stage_cost;
  for (uint8_t i = 0; i < 9; ++i)
    cost += gamma*local_cost_to_go[i]*local_trans_prob[i];

  // Update the cost.
  curr_optimal_cost[idx] = cost;

  return;
}

__global__
void cudaPolicyImprovment(
    uint32_t height, uint32_t width, float gamma,
    float* trans_prob, float* stage_cost,
    float* optimal_cost, uint8_t* optimal_action) {

  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  int idx = y*width + x;
  // Check if this point is outside the range.
  if (x < 0 || x >= width || y < 0 || y >= height) return;

  // Transfer the data from global memory to local.
  float local_trans_prob[81] = {0.0f};
  float local_stage_cost[9] = {0.0f};
  float local_cost_to_go[9] = {0.0f};

  memcpy(local_trans_prob, trans_prob+idx*81, sizeof(float)*81);
  memcpy(local_stage_cost, stage_cost+idx*9, sizeof(float)*9);
  for (int8_t oy = -1, i = 0; oy < 2; ++oy) {
    for (int8_t ox = -1; ox < 2; ++ox, ++i) {
      int nx = x + ox; int ny = y + oy;
      if (nx >= 0 && nx < width && ny >= 0 && ny < height)
        local_cost_to_go[i] = optimal_cost[ny*width+nx];
    }
  }

  // Update the optimal cost.
  float opt_cost = FLT_MAX;
  uint8_t opt_action = 0;

  for (uint8_t u = 0; u < 9; ++u) {

    float* tp = local_trans_prob + u*9;
    float cost = local_stage_cost[u];
    for (uint8_t i = 0; i < 9; ++i)
      cost += gamma*local_cost_to_go[i]*tp[i];

    if (cost < opt_cost) {
      opt_cost = cost;
      opt_action = u;
    }
  }

  optimal_action[idx] = opt_action;

  return;
}



