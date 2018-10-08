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
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <boost/multi_array.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <path_planning_2d/helper_cuda/helper_cuda.h>

#include <ros/ros.h>
#include <path_planning_2d/pomdp_path_planning_2d.h>

using namespace std;
namespace bst = boost;

typedef bst::multi_array_types::index_range IndexRange;

// Declare the model data.
extern float* dev_trans_prob;
extern float* dev_meas_prob;
extern float* dev_stage_reward;

// Alpha vectors for the Fast Informed Bound algorithm.
float* dev_fib_alphas1;
float* dev_fib_alphas2;
float* dev_fib_alphas;
// The actions corresponding to the alpha vectors.
uint8_t* dev_fib_actions;

float* host_fib_alphas;
uint8_t* host_fib_actions;


void allocateDeviceMemoryOfFIB(
    const uint32_t height, const uint32_t width) {

  checkCudaErrors(cudaMalloc(
        (void**)(&dev_fib_alphas1), sizeof(float)*height*width*9));
  checkCudaErrors(cudaMalloc(
        (void**)(&dev_fib_alphas2), sizeof(float)*height*width*9));
  checkCudaErrors(cudaMalloc(
        (void**)(&dev_fib_actions), sizeof(uint8_t)*9));
  dev_fib_alphas = dev_fib_alphas1;

  float* fib_alphas = (float*)malloc(sizeof(float)*height*width*9);
  for (size_t i = 0; i < height*width*9; ++i) fib_alphas[i] = 0.0f;

  checkCudaErrors(cudaMemcpy(dev_fib_alphas1, fib_alphas,
        sizeof(float)*height*width*9, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_fib_alphas2, fib_alphas,
        sizeof(float)*height*width*9, cudaMemcpyHostToDevice));
  free(fib_alphas);

  uint8_t fib_actions[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  checkCudaErrors(cudaMemcpy(dev_fib_actions, fib_actions,
        sizeof(uint8_t)*9, cudaMemcpyHostToDevice));

  host_fib_alphas = (float*)malloc(sizeof(float)*height*width*9);
  host_fib_actions = (uint8_t*)malloc(sizeof(uint8_t)*9);

  return;
}

void freeDeviceMemoryOfFIB() {

  checkCudaErrors(cudaFree(dev_fib_alphas1));
  checkCudaErrors(cudaFree(dev_fib_alphas2));
  checkCudaErrors(cudaFree(dev_fib_actions));

  free(host_fib_alphas);
  free(host_fib_actions);

  return;
}


__global__
void cudaFIBValueIteration(
    const uint32_t height, const uint32_t width, const float gamma,
    const float* const __restrict__ trans_prob,
    const float* const __restrict__ meas_prob,
    const float* const __restrict__ stage_reward,
    const float* const __restrict__ prev_alphas,
    float* const __restrict__ curr_alphas) {

  const int32_t x = blockDim.x*blockIdx.x + threadIdx.x;
  const int32_t y = blockDim.y*blockIdx.y + threadIdx.y;
  const int32_t idx = y*width + x;
  // Check if this point is outside the range.
  if (x < 0 || x >= width || y < 0 || y >= height) return;

  // Copy the measurement likelihood and alpha vectors from the
  // the previous iteration into the local memory.
  float local_meas_prob[144] = {0.0f};
  float local_prev_alphas[81] = {0.0f};

  for (int8_t oy = -1, i = 0; oy < 2; ++oy) {
    for (int8_t ox = -1; ox < 2; ++ox, ++i) {
      int32_t nx = x + ox;
      int32_t ny = y + oy;
      int32_t nidx = ny*width + nx;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

      memcpy(local_meas_prob+16*i, meas_prob+16*nidx, sizeof(float)*16);
      memcpy(local_prev_alphas+9*i, prev_alphas+9*nidx, sizeof(float)*9);
    }
  }

  // Copy the stage reward for different actions at this state.
  float local_stage_reward[9] = {0.0f};
  memcpy(local_stage_reward, stage_reward+idx*9, sizeof(float)*9);

  //if (x==20 && y==30) {
  //  for (size_t s = 0; s < 9; ++s) {
  //    printf(GREEN "At state [%d]:\n" RESET, s);
  //    printf(BLUE "Local measurement likelihood: \n" RESET);
  //    for (size_t o = 0; o < 16; ++o) printf("%f ", local_meas_prob[16*s+o]);
  //    printf(BLUE "\nLocal alpha vectors: \n" RESET);
  //    for (size_t a = 0; a < 9; ++a) printf("%f ", local_prev_alphas[9*s+a]);
  //    printf("\n");
  //  }
  //}

  // Compute the elements for each alpha vector that
  // corresponds to this state.
  float local_curr_alphas[9] = {0.0f};
  for (int8_t a = 0; a < 9; ++a) {

    float local_trans_prob[9] = {0.0f};
    memcpy(local_trans_prob, trans_prob+81*idx+9*a, sizeof(float)*9);

    //if (x==20 && y==30) {
    //  printf(GREEN "a=%d Local Transition Probability:\n" RESET, a);
    //  for (size_t s = 0; s < 9; ++s) printf("%f ", local_trans_prob[s]);
    //  printf("\n");
    //}

    // Compute the stage reward.
    float reward = local_stage_reward[a];
    //if (x==20 && y==30) {
    //  printf(BLUE "a=%d reward: %f\n" RESET, a, reward);
    //}

    // Compute the reward-to-go.
    float reward_to_go = 0.0f;
    // Summation over all possible observations.
    for (int8_t o = 0; o < 16; ++o) {
      // This vector is useful for every action in the next step.
      float local_trans_meas[9] = {0.0f};
      for (int8_t sp = 0; sp < 9; ++sp)
        local_trans_meas[sp] = local_trans_prob[sp] * local_meas_prob[sp*16+o];

      float reward_to_go_o = -FLT_MAX;
      // Maximization over all possible actions for the next step.
      for (int8_t ap = 0; ap < 9; ++ap) {
        float reward_to_go_oa = 0.0f;
        // Summuation over all possible next states.
        for (int8_t sp = 0; sp < 9; ++sp)
          reward_to_go_oa += local_trans_meas[sp]*local_prev_alphas[sp*9+ap];
        if (reward_to_go_o < reward_to_go_oa) reward_to_go_o = reward_to_go_oa;
      }
      // Update the reward-to-go.
      reward_to_go += reward_to_go_o;
    }

    //if (x==20 && y==30) {
    //  printf(BLUE "a=%d reward-to-go: %f\n" RESET, a, reward_to_go);
    //}

    // Summing the stage reward and the discounted reward-to-go.
    local_curr_alphas[a] = reward + gamma*reward_to_go;
  }

  //if (x==20 && y==30) {
  //  printf(GREEN "local alphas:\n" RESET);
  //  for (size_t a = 0; a < 9; ++a) printf("%f ", local_curr_alphas[a]);
  //  printf("\n");
  //}

  // Update the alpha vectors.
  memcpy(curr_alphas+9*idx, local_curr_alphas, sizeof(float)*9);

  return;
}

void fastInformedBound(const uint32_t height,
    const uint32_t width, const float gamma) {

  dim3 blockPerGrid(
      static_cast<int>(ceil(static_cast<float>(width)/8.0f)),
      static_cast<int>(ceil(static_cast<float>(height)/8.0f)));
  dim3 threadPerBlock(8, 8);

  float* prev_fib_alphas = (float*)malloc(sizeof(float)*width*height*9);
  float* curr_fib_alphas = (float*)malloc(sizeof(float)*width*height*9);
  for (size_t i = 0; i < height*width*9; ++i) {
    prev_fib_alphas[i] = 0.0f;
    curr_fib_alphas[i] = 0.0f;
  }

  uint32_t total_iterations = 0;
  float alphas_inf_norm = 0;

  do {
    // Since the mapping for the Fast Informed Bound algorithm
    // is contractive. The alphas vectors converges with value
    // iterations.
    for (size_t i = 0; i < 5; ++i) {
      cudaFIBValueIteration<<<blockPerGrid, threadPerBlock>>>(
          height, width, gamma, dev_trans_prob, dev_meas_prob,
          dev_stage_reward, dev_fib_alphas1, dev_fib_alphas2);
      checkCudaErrors(cudaDeviceSynchronize());

      //getchar();

      cudaFIBValueIteration<<<blockPerGrid, threadPerBlock>>>(
          height, width, gamma, dev_trans_prob, dev_meas_prob,
          dev_stage_reward, dev_fib_alphas2, dev_fib_alphas1);
      checkCudaErrors(cudaDeviceSynchronize());

      //getchar();
    }
    total_iterations += 10;

    // Download the latest alpha vectors.
    checkCudaErrors(cudaMemcpy(curr_fib_alphas, dev_fib_alphas1,
          sizeof(float)*height*width*9, cudaMemcpyDeviceToHost));

    // Compute the inf norm of the changes of the alpha vectors.
    alphas_inf_norm = 0.0f;
    for (size_t i = 0; i < height*width*9; ++i) {
      float alpha_diff = fabs(prev_fib_alphas[i]-curr_fib_alphas[i]);
      if (alpha_diff > alphas_inf_norm) alphas_inf_norm = alpha_diff;
    }

    // Update the alphas vectors.
    memcpy(prev_fib_alphas, curr_fib_alphas,
        sizeof(float)*height*width*9);

    printf(GREEN "iterations[%d-%d] inf-norm: " RESET,
        total_iterations-10, total_iterations);
    printf("%f\n", alphas_inf_norm);

  } while (alphas_inf_norm>0.01f && ros::ok());

  free(prev_fib_alphas);
  free(curr_fib_alphas);

  // Copy the results to the host memory.
  checkCudaErrors(cudaMemcpy(host_fib_alphas, dev_fib_alphas,
        sizeof(float)*height*width*9, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(host_fib_actions, dev_fib_actions,
        sizeof(uint8_t)*9, cudaMemcpyDeviceToHost));

  return;
}

void evaluateFibCpu(
    const uint32_t height, const uint32_t width,
    const float* const belief, float& value, uint8_t& action) {

  bst::const_multi_array_ref<float, 2> fib_alphas(
      host_fib_alphas, bst::extents[height*width][9]);

  float fib_values[9] = {0.0f};
  for (uint8_t i = 0; i < 9; ++i) {
    bst::const_multi_array_ref<float, 2>::const_array_view<1>::type
      fib_alpha = fib_alphas[bst::indices[IndexRange()][i]];
    fib_values[i] = inner_product(belief, belief+height*width, fib_alpha.begin(), 0.0f);
  }

  size_t max_value_idx = max_element(fib_values, fib_values+9) - fib_values;
  value = fib_values[max_value_idx];
  action = host_fib_actions[max_value_idx];

  return;
}

void evaluateFibGpu(
    const uint32_t height, const uint32_t width,
    const float* const belief, float& value, uint8_t& action) {
  // Upload the belief vector to the device.
  float* dev_belief;
  checkCudaErrors(cudaMalloc(
        (void**)(&dev_belief), sizeof(float)*height*width));
  checkCudaErrors(cudaMemcpy(dev_belief, belief,
        sizeof(float)*height*width, cudaMemcpyHostToDevice));

  // Evaluate the value using the alpha vectors.
  float* fib_values = (float*)malloc(sizeof(float)*9);
  float* dev_fib_values;
  checkCudaErrors(cudaMalloc((void**)(&dev_fib_values), sizeof(float)*9));

  float dummy_alpha = 1.0f;
  float dummy_beta = 0.0f;

  cublasHandle_t cublas_handle;
  checkCudaErrors(cublasCreate(&cublas_handle));

  checkCudaErrors(cublasSgemv(cublas_handle, CUBLAS_OP_N,
        9, height*width, &dummy_alpha,
        dev_fib_alphas, 9,
        dev_belief, 1, &dummy_beta,
        dev_fib_values, 1));
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(fib_values, dev_fib_values,
        sizeof(float)*9, cudaMemcpyDeviceToHost));

  // Find the corresponding value and action.
  size_t max_value_idx = max_element(fib_values, fib_values+9) - fib_values;
  value = fib_values[max_value_idx];
  action = host_fib_actions[max_value_idx];

  // Free allocated memory.
  free(fib_values);
  checkCudaErrors(cudaFree(dev_belief));
  checkCudaErrors(cudaFree(dev_fib_values));
  checkCudaErrors(cublasDestroy(cublas_handle));

  return;
}

void saveFibDataToFile(const uint32_t height, const uint32_t width) {

  // Save the alpha vectors.
  FILE* fib_alphas_fid = fopen("fib_alphas", "w");
  for (uint32_t s = 0; s < height*width; ++s) {
    for (uint32_t u = 0; u < 9; ++u)
      fprintf(fib_alphas_fid, "%15.8f", host_fib_alphas[s*9+u]);
    fprintf(fib_alphas_fid, "\n");
  }
  fclose(fib_alphas_fid);

  // Save the actions.
  FILE* fib_actions_fid = fopen("fib_actions", "w");
  for (uint32_t u = 0; u < 9; ++u)
    fprintf(fib_actions_fid, "%10u\n", host_fib_actions[u]);
  fclose(fib_actions_fid);

  return;
}

bool loadFibDataFromFile(const uint32_t height, const uint32_t width) {

  // Load alpha vectors.
  FILE* fib_alphas_fid = fopen("fib_alphas", "r");
  for (uint32_t s = 0; s < height*width; ++s) {
    for (uint32_t u = 0; u < 9; ++u) {
      if (fscanf(fib_alphas_fid, "%f", host_fib_alphas+s*9+u) == EOF) {
        printf(RED "Data dimension is not set properly\n" RESET);
        return false;
      }
    }
  }
  fclose(fib_alphas_fid);

  // Load the actions.
  FILE* fib_actions_fid = fopen("fib_actions", "r");
  for (uint32_t u = 0; u < 9; ++u) {
    if (fscanf(fib_actions_fid, "%u", host_fib_actions+u) == EOF) {
      printf(RED "Data dimension is not set properly\n" RESET);
      return false;
    }
  }
  fclose(fib_actions_fid);

  // Upload the data to the device.
  checkCudaErrors(cudaMemcpy(dev_fib_alphas, host_fib_alphas,
        sizeof(float)*height*width*9, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_fib_actions, host_fib_actions,
        sizeof(uint8_t)*9, cudaMemcpyHostToDevice));

  return true;
}
