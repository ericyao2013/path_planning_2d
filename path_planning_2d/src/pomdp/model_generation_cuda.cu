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
#include <path_planning_2d/pomdp_path_planning_2d.h>

// Map of the environment.
uint8_t* dev_map;
// Transition probability P(xi, u, xj).
float* dev_trans_prob;
// Measurement Likelihood L(xi, z).
float* dev_meas_prob;
// Stage reward R(xi, u);
float* dev_stage_reward;

float* host_trans_prob;
float* host_meas_prob;
float* host_stage_reward;


void allocateDeviceMemoryOfModel(
    const uint32_t height, const uint32_t width) {

  checkCudaErrors(cudaMalloc(
        (void**)(&dev_map), sizeof(uint8_t)*height*width));
  checkCudaErrors(cudaMalloc(
        (void**)(&dev_trans_prob), sizeof(float)*height*width*9*9));
  checkCudaErrors(cudaMalloc(
        (void**)(&dev_meas_prob), sizeof(float)*height*width*16));
  checkCudaErrors(cudaMalloc(
        (void**)(&dev_stage_reward), sizeof(float)*height*width*9));

  host_trans_prob = (float*)malloc(sizeof(float)*height*width*9*9);
  host_meas_prob = (float*)malloc(sizeof(float)*height*width*16);
  host_stage_reward = (float*)malloc(sizeof(float)*height*width*9);

  return;
}

void freeDeviceMemoryOfModel() {

  checkCudaErrors(cudaFree(dev_map));
  checkCudaErrors(cudaFree(dev_trans_prob));
  checkCudaErrors(cudaFree(dev_meas_prob));
  checkCudaErrors(cudaFree(dev_stage_reward));

  free(host_trans_prob);
  free(host_meas_prob);
  free(host_stage_reward);

  return;
}

void saveModelDataToFile(
    const uint32_t height, const uint32_t width) {

  // Save transition probability.
  FILE* trans_prob_fid = fopen("model_data_trans_prob", "w");
  for (uint32_t s = 0; s < height*width; ++s) {
    for (uint32_t u = 0; u < 9; ++u) {
      for (uint32_t i = 0; i < 9; ++i)
        fprintf(trans_prob_fid, "%15.8f", host_trans_prob[81*s+9*u+i]);
      fprintf(trans_prob_fid, "\n");
    }
  }
  fclose(trans_prob_fid);

  // Save measurement likelihood.
  FILE* meas_prob_fid = fopen("model_data_meas_prob", "w");
  for (uint32_t s = 0; s < height*width; ++s) {
    for (uint32_t z = 0; z < 16; ++z)
      fprintf(meas_prob_fid, "%15.8f", host_meas_prob[16*s+z]);
    fprintf(meas_prob_fid, "\n");
  }
  fclose(meas_prob_fid);

  // Save stage reward.
  FILE* stage_reward_fid = fopen("model_data_stage_reward", "w");
  for (uint32_t s = 0; s < height*width; ++s) {
    for (uint32_t u = 0; u < 9; ++u)
      fprintf(stage_reward_fid, "%15.8f", host_stage_reward[9*s+u]);
    fprintf(stage_reward_fid, "\n");
  }
  fclose(stage_reward_fid);

  return;
}

bool loadModelDataFromFile(
    const uint32_t height, const uint32_t width) {

  // Load transition probability.
  FILE* trans_prob_fid = fopen("model_data_trans_prob", "r");
  for (uint32_t s = 0; s < height*width; ++s) {
    for (uint32_t u = 0; u < 9; ++u) {
      for (uint32_t i = 0; i < 9; ++i) {
        if (fscanf(trans_prob_fid, "%f ", host_trans_prob+81*s+9*u+i) == EOF) {
          printf(RED "Data dimension is not set properly\n" RESET);
          return false;
        }
      }
    }
  }
  fclose(trans_prob_fid);

  // Save measurement likelihood.
  FILE* meas_prob_fid = fopen("model_data_meas_prob", "r");
  for (uint32_t s = 0; s < height*width; ++s) {
    for (uint32_t z = 0; z < 16; ++z) {
      if (fscanf(meas_prob_fid, "%f ", host_meas_prob+16*s+z) == EOF) {
        printf(RED "Data dimension is not set properly\n" RESET);
        return false;
      }
    }
  }
  fclose(meas_prob_fid);

  // Save stage reward.
  FILE* stage_reward_fid = fopen("model_data_stage_reward", "r");
  for (uint32_t s = 0; s < height*width; ++s) {
    for (uint32_t u = 0; u < 9; ++u) {
      if (fscanf(stage_reward_fid, "%f ", host_stage_reward+9*s+u) == EOF) {
        printf(RED "Data dimension is not set properly\n" RESET);
        return false;
      }
    }
  }
  fclose(stage_reward_fid);

  // Upload the data to the device.
  checkCudaErrors(cudaMemcpy(dev_trans_prob, host_trans_prob,
        sizeof(float)*height*width*81, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_meas_prob, host_meas_prob,
        sizeof(float)*height*width*16, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_stage_reward, host_stage_reward,
        sizeof(float)*height*width*9, cudaMemcpyHostToDevice));

  return true;
}

__device__
void  cudaTransitionProbability(const uint8_t u,
    const uint8_t* const __restrict__ map,
    float* const __restrict__ trans_prob,
    float* const __restrict__ trans_prob_naive) {
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

  // In the case the current location is occupied,
  // the robot is trapped location for sure. This is
  // not important for executing the control since
  // such case never happens.
  if (map[4] == 1) {
    for (uint8_t i = 0; i < 9; ++i) trans_prob[i] = 0.0f;
    trans_prob[4] = 1.0f;
  }

  return;
}

__device__
void cudaMeasurementLikelihood(
    const uint8_t* const __restrict__ map,
    float* const __restrict__ meas_prob) {

  // The measured cells around the robot are,
  // x | 0 | x
  // - - - - -
  // 1 | x | 2
  // - - - - -
  // x | 3 | x
  // In total, there are 16 different measurements.

  // Extract the true measruement of the
  // corresponding cells.
  uint8_t m[4] = {map[1], map[3], map[5], map[7]};

  for (uint8_t i = 0; i < 16; ++i) {
    float l0 = ((i>>0)&0x01) == m[0] ? 0.98 : 0.02;
    float l1 = ((i>>1)&0x01) == m[1] ? 0.98 : 0.02;
    float l2 = ((i>>2)&0x01) == m[2] ? 0.98 : 0.02;
    float l3 = ((i>>3)&0x01) == m[3] ? 0.98 : 0.02;
    meas_prob[i] = l0 * l1 * l2 * l3;
  }

  return;
}

__device__
void cudaStageReward(
    const uint32_t x, const uint32_t y,
    const uint32_t gx, const uint32_t gy,
    const uint8_t* const __restrict__ map,
    const float* const __restrict__ trans_prob_naive,
    float* const __restrict__ stage_reward) {

  // Generate the reward matrix based on the local map.
  float map_reward[9] = {0.0f};
  for (uint8_t i = 0; i < 9; ++i) {
    if (map[i] == 1) map_reward[i] = -2.0f;
    else map_reward[i] = -1.0f;
  }

  // It requires some special care if the robot remains
  // at the same location. Unless the location is the goal,
  // the reward is as low as colliding into an obstacle.
  //map_reward[4] = (x!=gx || y!=gy) ? -2.0f : 0.0f;

  // Compute the reward based on the transition probability
  // of each action.
  for (uint8_t u = 0; u < 9; ++u) {
    for (uint8_t i = 0; i < 9; ++i)
      stage_reward[u] += map_reward[i]*trans_prob_naive[9*u+i];
  }

  stage_reward[4] = (x!=gx || y!=gy) ? -2.0f : 0.0f;

  return;
}

__global__
void cudaGenerateModelData(
    const uint32_t height, const uint32_t width,
    const int32_t gx, const int32_t gy,
    const uint8_t* const __restrict__ map,
    float* const __restrict__ trans_prob,
    float* const __restrict__ meas_prob,
    float* const __restrict__ stage_reward) {

  const int32_t x = blockDim.x*blockIdx.x + threadIdx.x;
  const int32_t y = blockDim.y*blockIdx.y + threadIdx.y;
  const int32_t idx = y*width + x;
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
  //memcpy(trans_prob_naive+idx*81, local_trans_prob_naive, sizeof(float)*81);

  // Generate the measurement likelihood.
  float local_meas_prob[16] = {0.0f};
  cudaMeasurementLikelihood(local_map, local_meas_prob);
  memcpy(meas_prob+idx*16, local_meas_prob, sizeof(float)*16);

  // Generate the stage rewards.
  float local_stage_reward[9] = {0.0f};
  cudaStageReward(x, y, gx, gy,
      local_map, local_trans_prob_naive, local_stage_reward);
  memcpy(stage_reward+idx*9, local_stage_reward, sizeof(float)*9);

  return;
}

void generateModelData(
    const uint32_t height, const uint32_t width,
    const uint8_t* const map, const int32_t* const goal) {

  dim3 blockPerGrid(
      static_cast<int>(std::ceil(static_cast<float>(width)/8.0f)),
      static_cast<int>(std::ceil(static_cast<float>(height)/8.0f)));
  dim3 threadPerBlock(8, 8);

  // Upload the map to device.
  printf(CYAN "Upload map to the device...\n" RESET);
  checkCudaErrors(cudaMemcpy(dev_map, map,
        sizeof(uint8_t)*height*width, cudaMemcpyHostToDevice));

  // Generate model data.
  printf(CYAN "Generate model tensors...\n" RESET);
  cudaGenerateModelData<<<blockPerGrid, threadPerBlock>>>(
      height, width, goal[0], goal[1], dev_map,
      dev_trans_prob, dev_meas_prob, dev_stage_reward);
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(host_trans_prob, dev_trans_prob,
        sizeof(float)*height*width*9*9, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(host_meas_prob, dev_meas_prob,
        sizeof(float)*height*width*16, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(host_stage_reward, dev_stage_reward,
        sizeof(float)*height*width*9, cudaMemcpyDeviceToHost));

  //float* trans_prob = (float*)malloc(sizeof(float)*height*width*81);
  //float* meas_prob = (float*)malloc(sizeof(float)*height*width*16);
  //float* stage_reward = (float*)malloc(sizeof(float)*height*width*9);

  //checkCudaErrors(cudaMemcpy(trans_prob, dev_trans_prob,
  //      sizeof(float)*height*width*81, cudaMemcpyDeviceToHost));
  //checkCudaErrors(cudaMemcpy(meas_prob, dev_meas_prob,
  //      sizeof(float)*height*width*16, cudaMemcpyDeviceToHost));
  //checkCudaErrors(cudaMemcpy(stage_reward, dev_stage_reward,
  //      sizeof(float)*height*width*9, cudaMemcpyDeviceToHost));

  //uint32_t x_target = 20;
  //uint32_t y_target = 30;
  //uint32_t idx_target = y_target*width + x_target;
  //float* T_target = trans_prob + 81*idx_target;
  //float* L_target = meas_prob + 16*idx_target;
  //float* R_target = stage_reward + 9*idx_target;

  //printf(GREEN "Transition Probability at [%d, %d]: \n" RESET, x_target, y_target);
  //for (uint8_t a = 0; a < 9; ++a) {
  //  printf(BLUE "a=%d: " RESET, a);
  //  for (uint8_t s = 0; s < 9; ++s) {
  //    printf("%f ", T_target[9*a+s]);
  //  }
  //  printf("\n");
  //}

  //printf(GREEN "Measurement Likelihood at [%d, %d]: \n" RESET, x_target, y_target);
  //for (uint8_t o = 0; o < 16; ++o) {
  //  printf("%f ", L_target[o]);
  //}
  //printf("\n");

  //printf(GREEN "Stage Reward at [%d, %d]: \n" RESET, x_target, y_target);
  //for (uint8_t a = 0; a < 9; ++a) {
  //  printf("%f ", R_target[a]);
  //}
  //printf("\n");

  //free(trans_prob);
  //free(meas_prob);
  //free(stage_reward);

  return;
}

