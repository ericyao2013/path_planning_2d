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
#include <cassert>

#include <iostream>
#include <vector>
#include <numeric>
#include <functional>
#include <boost/multi_array.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <path_planning_2d/helper_cuda/helper_cuda.h>
#include <path_planning_2d/pomdp_path_planning_2d.h>
#include <ros/ros.h>

using namespace std;
namespace bst = boost;

typedef bst::multi_array_types::index_range IndexRange;

// Declare the model data.
extern float* dev_trans_prob;
extern float* dev_meas_prob;
extern float* dev_stage_reward;

extern float* host_trans_prob;
extern float* host_meas_prob;
extern float* host_stage_reward;

uint32_t belief_set_size;

float* dev_pbvi_alphas;
uint8_t* dev_pbvi_actions;

float* host_pbvi_alphas;
uint8_t* host_pbvi_actions;


void allocateDeviceMemoryOfPBVI(
    const uint32_t height, const uint32_t width,
    const uint32_t set_size) {

  checkCudaErrors(cudaMalloc((void**)(&dev_pbvi_alphas),
        sizeof(float)*set_size*height*width));
  checkCudaErrors(cudaMalloc((void**)(&dev_pbvi_actions),
        sizeof(uint8_t)*set_size));

  host_pbvi_alphas = (float*)malloc(sizeof(float)*height*width*set_size);
  host_pbvi_actions = (uint8_t*)malloc(sizeof(uint8_t)*set_size);

  belief_set_size = set_size;

  return;
}

void freeDeviceMemoryOfPBVI() {

  checkCudaErrors(cudaFree(dev_pbvi_alphas));
  checkCudaErrors(cudaFree(dev_pbvi_actions));

  free(host_pbvi_alphas);
  free(host_pbvi_actions);

  return;
}

__global__
void cudaBayesBeliefUpdate(
    const uint32_t height, const uint32_t width,
    const float* const __restrict__ trans_prob,
    const float* const __restrict__ meas_prob,
    const float* const __restrict__ belief_in,
    const uint8_t u, const uint8_t z,
    float* const __restrict__ belief_out) {

  const int32_t x = blockDim.x*blockIdx.x + threadIdx.x;
  const int32_t y = blockDim.y*blockIdx.y + threadIdx.y;
  const int32_t idx = y*width + x;
  if (x<0 || x>=width || y<0 || y>=height) return;

  // Get data from the global memory.
  // TODO: Optimize the global memory access.
  float local_trans_prob[9] = {0.0f};
  float local_belief_in[9] = {0.0f};

  for (int8_t oy = -1, s = 0; oy < 2; ++oy) {
    for (int8_t ox = -1; ox < 2; ++ox, ++s) {
      const int32_t sx = x + ox;
      const int32_t sy = y + oy;
      const int32_t sidx = sy*width + sx;
      if (sx<0 || sx>=width || sy<0 || sy>=height) continue;

      local_trans_prob[s] = trans_prob[81*sidx+9*u+(8-s)];
      local_belief_in[s] = belief_in[sidx];
    }
  }

  // Compute the prior probability.
  float p = 0.0f;
  for (uint8_t s = 0; s < 9; ++s)
    p += local_trans_prob[s] * local_belief_in[s];

  // Compute the posterior probability.
  float local_meas_prob = meas_prob[16*idx+z];
  p *= local_meas_prob;

  // Save the result to the output.
  // Note the output belief is un-normalized.
  belief_out[idx] = p;

  return;
}

void normalizeProbDensity(
    bst::multi_array_ref<float, 1>& density) {
  // Find the sum of the density vector.
  float sum = accumulate(density.begin(), density.end(), 0.0f);

  // Normalize the density.
  for_each(density.begin(), density.end(),
      [&sum](float& x)->float{return x/=sum;});

  return;
}

uint32_t sampleFromProbDensity(
    const bst::const_multi_array_ref<float, 1>& density) {
  // Create the corresponding distribution.
  bst::multi_array<float, 1> distribution(bst::extents[density.size()]);
  partial_sum(density.begin(),
      density.end(), distribution.begin(), plus<float>());

  // Generate a random sample based on the distribution.
  float sample_rand = (float)rand() / ((float)RAND_MAX+1.0f);
  const auto sample_iter = find_if(
      distribution.begin(), distribution.end(),
      [&sample_rand](const float& x)->bool{return x >= sample_rand;});

  return sample_iter - distribution.begin();
}



void generateBeliefSet(
    const uint32_t height, const uint32_t width,
    const uint32_t max_size,
    const float* const __restrict__ b0,
    float* const __restrict__ b_set_out) {

  dim3 blockPerGrid(
      static_cast<int>(std::ceil(static_cast<float>(width)/8.0f)),
      static_cast<int>(std::ceil(static_cast<float>(height)/8.0f)));
  dim3 threadPerBlock(8, 8);

  // Get the model data.
  const float* const trans_prob = host_trans_prob;
  const float* const meas_prob = host_meas_prob;

  // Initialize the belief set.
  uint32_t set_size = 1;
  bst::multi_array_ref<float, 2> b_set(b_set_out, bst::extents[max_size][height*width]);
  memcpy(b_set.origin(), b0, sizeof(float)*height*width);

  // These pointers will be useful in the inner loops later.
  float *dev_bi, *dev_new_bi;
  checkCudaErrors(cudaMalloc(
        (void**)(&dev_bi), sizeof(float)*height*width));
  checkCudaErrors(cudaMalloc(
        (void**)(&dev_new_bi), sizeof(float)*height*width));

  // Expand the belief set.
  while (set_size < max_size && ros::ok()) {
    // Template to hold the new beliefs.
    // A new belief is generated based on each existing belief.
    bst::multi_array<float, 2> new_bs(bst::extents[set_size][height*width]);
    vector<float> new_bs_l1(set_size, FLT_MAX);

    for (size_t i = 0; i < set_size; ++i) {
      // Get the belief of interest.
      bst::multi_array<float, 2>::const_subarray<1>::type bi = b_set[i];
      checkCudaErrors(cudaMemcpy(dev_bi, bi.origin(),
            sizeof(float)*height*width, cudaMemcpyHostToDevice));

      // Allocate memory to hold the new beliefs for each action.
      bst::multi_array<float, 2> new_bis(bst::extents[9][height*width]);
      vector<float> new_bis_l1(9, FLT_MAX);

      // Generate one new belief for each different action.
      for (uint8_t a = 0; a < 9; ++a) {
        bst::multi_array<float, 2>::subarray<1>::type new_bi = new_bis[a];
        // Procedure to sample a measurement:
        // 1. Sample a current state based on bi.
        // 2. Sample a next state based on the current state and the action.
        // 3. Sample a measurement based on the next state.
        uint32_t s = sampleFromProbDensity(
            bst::const_multi_array_ref<float, 1>(bi.origin(), bst::extents[height*width]));
        uint32_t ns_local = sampleFromProbDensity(
            bst::const_multi_array_ref<float, 1>(trans_prob+81*s+9*a, bst::extents[9]));
        uint32_t ns = (s/width+ns_local/3-1)*width + (s%width+ns_local%3-1);
        uint8_t z = sampleFromProbDensity(
            bst::const_multi_array_ref<float, 1>(meas_prob+16*ns, bst::extents[16]));
        //printf("action/s/ns_local/ns/z: %d, %d, %d, %d, %d\n", a, s, ns_local, ns, z);

        // Given the current belief, action, and the measurement,
        // compute the unnormaized posterior belief.
        cudaBayesBeliefUpdate<<<blockPerGrid, threadPerBlock>>>(
            height, width, dev_trans_prob,
            dev_meas_prob, dev_bi, a, z, dev_new_bi);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaMemcpy(new_bi.origin(), dev_new_bi,
              sizeof(float)*height*width, cudaMemcpyDeviceToHost));

        // Normalize the new belief.
        bst::multi_array_ref<float, 1> new_bi_ref(
            new_bi.origin(), bst::extents[height*width]);
        normalizeProbDensity(new_bi_ref);

        // Compute L1 distance between the new belief
        // and the existing belief set.
        for (size_t j = 0; j < set_size; ++j) {
          float l1 = 0.0f;
          for (size_t k = 0; k < height*width; ++k)
            l1 += abs(new_bi[k]-b_set[j][k]);

          if (l1 < new_bis_l1[a]) new_bis_l1[a] = l1;
        }
      }

      // Find the best new belief based on the L1 distance from
      // the new belief and the existing belief set.
      uint8_t best_new_bi_idx = max_element(
          new_bis_l1.begin(), new_bis_l1.end()) - new_bis_l1.begin();

      // Add the best belief to the belief set.
      new_bs[i] = new_bis[best_new_bi_idx];
      new_bs_l1[i] = new_bis_l1[best_new_bi_idx];
    }

    // Add the newly generated beliefs to the belief set.
    if (set_size < 100) {
      // Copy all the new beliefs into the belief set if
      // the size of the belief set is still small.
      for (size_t i = 0; i < new_bs.shape()[0]; ++i) {
        b_set[set_size++] = new_bs[i];
        if (set_size >= max_size) break;
      }
    } else {
      // Partially sort the new beliefs based on the L1 distance.
      vector<size_t> sorted_idx(new_bs_l1.size());
      iota(sorted_idx.begin(), sorted_idx.end(), 0);
      partial_sort(
          sorted_idx.begin(), sorted_idx.end(), sorted_idx.begin()+100,
          [&new_bs_l1](const size_t& i, const size_t& j){
            return new_bs_l1[i] > new_bs_l1[j];
          });

      // Copy the best 100 new beliefs.
      for (size_t i = 0; i < 100; ++i) {
        b_set[set_size++] = new_bs[sorted_idx[i]];
        if (set_size >= max_size) break;
      }
    }
    printf(GREEN "set size: " RESET);
    printf("%d\n", set_size);
  }

  // Free the allocated variables.
  checkCudaErrors(cudaFree(dev_bi));
  checkCudaErrors(cudaFree(dev_new_bi));

  return;
}

__global__
void cudaComputeGammaOA(
    const uint32_t height, const uint32_t width,
    const uint32_t set_size, const float gamma,
    const float* const __restrict__ trans_prob_a,
    const float* const __restrict__ meas_prob_o,
    const float* const __restrict__ alphas,
    float* const __restrict__ alphas_ao) {

  const int32_t x = blockDim.x*blockIdx.x + threadIdx.x;
  const int32_t y = blockDim.y*blockIdx.y + threadIdx.y;
  const int32_t idx = y*width + x;
  if (x<0 || x>=width || y<0 || y>=height) return;

  float trans_meas_prob_ao[9] = {0.0f};
  memcpy(trans_meas_prob_ao, trans_prob_a+idx*9, sizeof(float)*9);

  for (int8_t oy = -1, s = 0; oy < 2; ++oy) {
    for (int8_t ox = -1; ox < 2; ++ox, ++s) {
      const int32_t sx = x + ox;
      const int32_t sy = y + oy;
      const int32_t sidx = sy*width + sx;
      if (sx<0 || sx>=width || sy<0 || sy>=height) continue;

      trans_meas_prob_ao[s] *= meas_prob_o[sidx];
    }
  }

  // TODO: Optimize the memory access if possible.
  for (size_t i = 0; i < set_size; ++i) {
    float alphas_ao_i_s = 0.0f;

    for (int8_t oy = -1, s = 0; oy < 2; ++oy) {
      for (int8_t ox = -1; ox < 2; ++ox, ++s) {
        const int32_t sx = x + ox;
        const int32_t sy = y + oy;
        const int32_t sidx = sy*width + sx;
        if (sx<0 || sx>=width || sy<0 || sy>=height) continue;

        alphas_ao_i_s += trans_meas_prob_ao[s] *
          alphas[i*height*width+sidx];
      }
    }

    alphas_ao[i*height*width+idx] = gamma * alphas_ao_i_s;
  }

  return;
}

void backupAlphaVectors(
    const uint32_t height, const uint32_t width,
    const float gamma, const uint32_t set_size,
    const float* const __restrict__ b_set_in,
    float* const __restrict__ alphas_out,
    uint8_t* const __restrict__ actions_out) {

  dim3 blockPerGrid(
      static_cast<int>(std::ceil(static_cast<float>(width)/8.0f)),
      static_cast<int>(std::ceil(static_cast<float>(height)/8.0f)));
  dim3 threadPerBlock(8, 8);

  // Initialize cublas.
  cublasHandle_t cublas_handle;
  checkCudaErrors(cublasCreate(&cublas_handle));

  // Get the model data.
  bst::const_multi_array_ref<float, 3> trans_prob(
      host_trans_prob, bst::extents[height*width][9][9]);
  bst::const_multi_array_ref<float, 2> meas_prob(
      host_meas_prob, bst::extents[height*width][16]);
  bst::const_multi_array_ref<float, 2> stage_reward(
      host_stage_reward, bst::extents[height*width][9]);

  // Set multi_array headers for the belief set and alpha vectors.
  bst::const_multi_array_ref<float, 2> b_set(
      b_set_in, bst::extents[set_size][height*width]);
  bst::multi_array_ref<float, 2> alphas(
      alphas_out, bst::extents[set_size][height*width]);
  bst::multi_array_ref<uint8_t, 1> actions(
      actions_out, bst::extents[set_size]);

  // Allocate device memory for the problem data.
  float *dev_alphas, *dev_b_set;
  checkCudaErrors(cudaMalloc((void**)(&dev_b_set),
        sizeof(float)*height*width*set_size));
  checkCudaErrors(cudaMemcpy(dev_b_set, b_set.origin(),
        sizeof(float)*height*width*set_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void**)(&dev_alphas),
        sizeof(float)*height*width*set_size));
  checkCudaErrors(cudaMemcpy(dev_alphas, alphas.origin(),
        sizeof(float)*height*width*set_size, cudaMemcpyHostToDevice));
  //uint8_t* dev_actions;
  //checkCudaErrors(cudaMalloc((void**)(&dev_actions),
  //      sizeof(uint8_t)*set_size));


  // Intermediate variables for the backup operation.
  bst::multi_array<float, 2> trans_prob_a(
      bst::extents[height*width][9]);
  bst::multi_array<float, 1> meas_prob_o(
      bst::extents[height*width]);

  float *dev_trans_prob_a, *dev_meas_prob_o;
  checkCudaErrors(cudaMalloc((void**)(&dev_trans_prob_a),
        sizeof(float)*height*width*9));
  checkCudaErrors(cudaMalloc((void**)(&dev_meas_prob_o),
        sizeof(float)*height*width));

  bst::multi_array<float, 3> Gamma_a(
      bst::extents[9][set_size][height*width]);
  bst::multi_array<float, 4> Gamma_ao(
      bst::extents[9][16][set_size][height*width]);

  float *dev_alphas_ao, *dev_alphas_a;
  checkCudaErrors(cudaMalloc((void**)(&dev_alphas_a),
        sizeof(float)*height*width*set_size));
  checkCudaErrors(cudaMalloc((void**)(&dev_alphas_ao),
        sizeof(float)*height*width*set_size));

  bst::multi_array<float, 2> alphas_ao_reward(
      bst::extents[set_size][set_size]);
  bst::multi_array<float, 2> alphas_ao_max(
      bst::extents[set_size][height*width]);

  float *dev_alphas_ao_reward, *dev_alphas_ao_max;
  checkCudaErrors(cudaMalloc((void**)(&dev_alphas_ao_reward),
        sizeof(float)*set_size*set_size));
  checkCudaErrors(cudaMalloc((void**)(&dev_alphas_ao_max),
        sizeof(float)*set_size*height*width));

  // How much iterations to backup?
  // The hardcoded numbers are:
  // The precision requirement is 1e-3.
  // The inf norm of stage reward is 5.
  uint32_t max_backup_iterations =
    static_cast<uint32_t>(ceil(log(1.0e-3f/5.0f)/log(gamma)));

  // Perform the backup operation on the alpha vectors.
  ros::Time start_time;
  for (size_t backup_iter = 0;
      backup_iter<max_backup_iterations && ros::ok(); ++backup_iter) {

    start_time = ros::Time::now();

    // Compute Gamma_ao.
    //printf(YELLOW "Compute Gamma_ao..." RESET "\n");
    for (uint8_t a = 0; a < 9; ++a) {
      trans_prob_a = trans_prob[bst::indices[IndexRange()][a][IndexRange()]];
      checkCudaErrors(cudaMemcpy(dev_trans_prob_a, trans_prob_a.origin(),
            sizeof(float)*height*width*9, cudaMemcpyHostToDevice));

      //printf("trans_prob_a[a=%d]: \n", a);
      //printMatrix<float>(trans_prob_a.origin(), height*width, 9);

      for (uint8_t o = 0; o < 16; ++o) {
        meas_prob_o = meas_prob[bst::indices[IndexRange()][o]];
        checkCudaErrors(cudaMemcpy(dev_meas_prob_o, meas_prob_o.origin(),
              sizeof(float)*height*width, cudaMemcpyHostToDevice));

        //printf("meas_prob_o[o=%d]: \n", o);
        //printMatrix<float>(meas_prob_o.origin(), height*width, 1);

        cudaComputeGammaOA<<<blockPerGrid, threadPerBlock>>>(
            height, width, set_size, gamma,
            dev_trans_prob_a, dev_meas_prob_o,
            dev_alphas, dev_alphas_ao);
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaMemcpy(Gamma_ao[a][o].origin(), dev_alphas_ao,
              sizeof(float)*height*width*set_size, cudaMemcpyDeviceToHost));
        //for (size_t i = 0; i < set_size; ++i) {
        //  printf("Gamma_ao[a:%d, o:%d, i:%d]:\n", a, o, i);
        //  printMatrix<float>(Gamma_ao[a][o][i].origin(), height, width);
        //}
      }
    }


    // Set Gamma_a based on the stage reward.
    //printf(YELLOW "Initialize Gamma_a..." RESET "\n");
    for (uint8_t a = 0; a < 9; ++a) {
      bst::multi_array<float, 2>::const_array_view<1>::type
        stage_reward_a = stage_reward[bst::indices[IndexRange()][a]];
      for (size_t i = 0; i < set_size; ++i) {
        Gamma_a[a][i] = stage_reward_a;
      }
    }

    //for (uint8_t a = 0; a < 9; ++a) {
    //  for (size_t i = 0; i < set_size; ++i) {
    //    printf("Gamma_a[a:%d, i:%d]:\n", a, i);
    //    printMatrix<float>(Gamma_a[a][i].origin(), 1, height*width);
    //  }
    //}

    // Update Gamma_a based on the Gamma_ao.
    //printf(YELLOW "Update Gamma_a..." RESET "\n");
    for (uint8_t a = 0; a < 9; ++a) {

      checkCudaErrors(cudaMemcpy(dev_alphas_a, Gamma_a[a].origin(),
            sizeof(float)*set_size*height*width, cudaMemcpyHostToDevice));

      for (uint8_t o = 0; o < 16; ++o) {

        checkCudaErrors(cudaMemcpy(dev_alphas_ao, Gamma_ao[a][o].origin(),
              sizeof(float)*height*width*set_size, cudaMemcpyHostToDevice));

        // TODO: Comment this well.
        float dummy_a = 1.0f, dummy_b = 0.0f;
        checkCudaErrors(cublasSgemm(cublas_handle,
              CUBLAS_OP_T,
              CUBLAS_OP_N,
              set_size, set_size, height*width,
              &dummy_a,
              dev_alphas_ao, height*width,
              dev_b_set, height*width,
              &dummy_b,
              dev_alphas_ao_reward, set_size));
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaMemcpy(
              alphas_ao_reward.origin(), dev_alphas_ao_reward,
              sizeof(float)*set_size*set_size, cudaMemcpyDeviceToHost));

        //for (size_t i = 0; i < set_size; ++i) {
        //  printf("belief[i:%lu]: \n", i);
        //  printMatrix<float>(b_set[i].origin(), 1, height*width);
        //}

        //for (size_t i = 0; i < set_size; ++i) {
        //  printf("Gamma_ao[a=%d][o=%d][i:%lu]: \n", a, o, i);
        //  printMatrix<float>(Gamma_ao[a][o][i].origin(), 1, height*width);
        //}

        //printf("a=%d o=%d alphas_ao_reward: \n", a, o);
        //printMatrix<float>(alphas_ao_reward.origin(), set_size, set_size);

        for (size_t i = 0; i < set_size; ++i) {
          bst::multi_array<float, 2>::subarray<1>::type
            b_reward = alphas_ao_reward[i];
          size_t max_o_idx = max_element(
              b_reward.begin(), b_reward.end()) - b_reward.begin();
          alphas_ao_max[i] = Gamma_ao[a][o][max_o_idx];
        }


        //for (size_t i = 0; i < set_size; ++i) {
        //  printf("alphas_ao_max[i:%lu]: \n", i);
        //  printMatrix<float>(alphas_ao_max[i].origin(), 1, height*width);
        //}


        checkCudaErrors(cudaMemcpy(dev_alphas_ao_max, alphas_ao_max.origin(),
              sizeof(float)*set_size*height*width, cudaMemcpyHostToDevice));
        // TODO: comment this well.
        float dummy_a2 = 1.0f, dummy_b2 = 1.0f;
        checkCudaErrors(cublasSgeam(cublas_handle,
              CUBLAS_OP_N, CUBLAS_OP_N,
              height*width, set_size,
              &dummy_a2,
              dev_alphas_ao_max, height*width,
              &dummy_b2,
              dev_alphas_a, height*width,
              dev_alphas_a, height*width));

        //for (size_t i = 0; i < set_size; ++i) {
        //  printf("Before update Gamma_a[a:%d, i:%d]:\n", a, i);
        //  printMatrix<float>(Gamma_a[a][i].origin(), 1, height*width);
        //}

        //checkCudaErrors(cudaMemcpy(Gamma_a[a].origin(), dev_alphas_a,
        //      sizeof(float)*set_size*height*width, cudaMemcpyDeviceToHost));

        //for (size_t i = 0; i < set_size; ++i) {
        //  printf("After update Gamma_a[a:%d, i:%d]:\n", a, i);
        //  printMatrix<float>(Gamma_a[a][i].origin(), 1, height*width);
        //}

      }

      checkCudaErrors(cudaMemcpy(Gamma_a[a].origin(), dev_alphas_a,
            sizeof(float)*set_size*height*width, cudaMemcpyDeviceToHost));
    }

    // Update the alpha vectors.
    // TODO: Optimize this section.
    //printf(YELLOW "Update alpha vectors..." RESET "\n");
    //for (uint8_t a = 0; a < 9; ++a) {
    //  for (size_t i = 0; i < set_size; ++i) {
    //    printf("Gamma_a[a:%d, i:%d]:\n", a, i);
    //    printMatrix<float>(Gamma_a[a][i].origin(), 1, height*width);
    //  }
    //}

    //for (size_t i = 0; i < set_size; ++i) {
    //  printf("belief[i:%lu]: \n", i);
    //  printMatrix<float>(b_set[i].origin(), 1, height*width);
    //}

    for (size_t i = 0; i < set_size; ++i) {
      float opt_reward = -FLT_MAX;
      uint8_t opt_action = 0;

      for (uint8_t a = 0; a < 9; ++a) {
        float action_reward = inner_product(
            b_set[i].begin(), b_set[i].end(), Gamma_a[a][i].begin(), 0.0f);
        if (action_reward > opt_reward) {
          opt_reward = action_reward;
          opt_action = a;
        }
      }

      alphas[i] = Gamma_a[opt_action][i];
      actions[i] = opt_action;
    }

    //for (size_t i = 0; i < set_size; ++i) {
    //  printf("alphas[i:%lu] action[i:%u]: %d\n", i, i, actions[i]);
    //  printMatrix<float>(alphas[i].origin(), 1, height*width);
    //}

    // Upload the updated alpha vectors to the device.
    checkCudaErrors(cudaMemcpy(dev_alphas, alphas.origin(),
          sizeof(float)*set_size*height*width, cudaMemcpyHostToDevice));

    printf(GREEN "backup iter[%lu] time: " RESET "%f\n",
        backup_iter, (ros::Time::now()-start_time).toSec());
  } // End backup for loop.


  // Free allocated memory.
  checkCudaErrors(cudaFree(dev_b_set));
  checkCudaErrors(cudaFree(dev_alphas));
  //checkCudaErrors(cudaFree(dev_actions));

  checkCudaErrors(cudaFree(dev_trans_prob_a));
  checkCudaErrors(cudaFree(dev_meas_prob_o));

  checkCudaErrors(cudaFree(dev_alphas_a));
  checkCudaErrors(cudaFree(dev_alphas_ao));
  checkCudaErrors(cudaFree(dev_alphas_ao_reward));
  checkCudaErrors(cudaFree(dev_alphas_ao_max));

  checkCudaErrors(cublasDestroy(cublas_handle));

  return;
}

void pointBasedValueIteration(
    const uint32_t height, const uint32_t width,
    const vector<float>& initial_belief, const float gamma) {

  // Generate the belief set.
  printf(CYAN "Generate the belief set..." RESET "\n");
  float* belief_set = (float*)malloc(
      sizeof(float)*height*width*belief_set_size);
  generateBeliefSet(height, width,
      belief_set_size, initial_belief.data(), belief_set);

  //// Check the output.
  //for (size_t i = 0; i < belief_set_size; ++i)
  //  printf("belief_set[%lu] norm: %f\n", i,
  //      accumulate(belief_set+i*height*width, belief_set+(i+1)*height*width, 0.0f));

  // Backup the alpha vectors.
  printf(CYAN "Backup alpha vectors..." RESET "\n");
  // Initialize the alpha vectors to 0.
  for (size_t i = 0; i < belief_set_size*height*width; ++i)
    host_pbvi_alphas[i] = 0.0f;

  backupAlphaVectors(height, width, gamma, belief_set_size,
      belief_set, host_pbvi_alphas, host_pbvi_actions);
  checkCudaErrors(cudaMemcpy(dev_pbvi_alphas, host_pbvi_alphas,
        sizeof(float)*belief_set_size*height*width, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_pbvi_actions, host_pbvi_actions,
        sizeof(uint8_t)*belief_set_size, cudaMemcpyHostToDevice));

  // Free allocated memory.
  free(belief_set);

  return;
}

void evaluatePbviCpu(
    const uint32_t height, const uint32_t width,
    const float* const belief, float& value, uint8_t& action) {

  bst::const_multi_array_ref<float, 2> pbvi_alphas(
      host_pbvi_alphas, bst::extents[belief_set_size][height*width]);

  vector<float> pbvi_values(belief_set_size);
  for (uint32_t i = 0; i < belief_set_size; ++i) {
    bst::const_multi_array_ref<float, 2>::const_subarray<1>::type
      pbvi_alpha = pbvi_alphas[i];
    pbvi_values[i] = inner_product(
        belief, belief+height*width, pbvi_alpha.begin(), 0.0f);
  }

  size_t max_value_idx = max_element(
      pbvi_values.begin(), pbvi_values.end()) - pbvi_values.begin();
  value = pbvi_values[max_value_idx];
  action = host_pbvi_actions[max_value_idx];

  return;
}

void evaluatePbviGpu(
    const uint32_t height, const uint32_t width,
    const float* const belief, float& value, uint8_t& action) {

  // Upload the belief vector to the device.
  float* dev_belief;
  checkCudaErrors(cudaMalloc(
        (void**)(&dev_belief), sizeof(float)*height*width));
  checkCudaErrors(cudaMemcpy(dev_belief, belief,
        sizeof(float)*height*width, cudaMemcpyHostToDevice));

  // Evaluate the value using the alpha vectors.
  float* pbvi_values = (float*)malloc(sizeof(float)*belief_set_size);
  float* dev_pbvi_values;
  checkCudaErrors(cudaMalloc((void**)(&dev_pbvi_values), sizeof(float)*belief_set_size));

  float dummy_alpha = 1.0f;
  float dummy_beta = 0.0f;

  cublasHandle_t cublas_handle;
  checkCudaErrors(cublasCreate(&cublas_handle));

  checkCudaErrors(cublasSgemv(cublas_handle, CUBLAS_OP_T,
        height*width, belief_set_size, &dummy_alpha,
        dev_pbvi_alphas, height*width,
        dev_belief, 1, &dummy_beta,
        dev_pbvi_values, 1));
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(pbvi_values, dev_pbvi_values,
        sizeof(float)*belief_set_size, cudaMemcpyDeviceToHost));

  // Find the corresponding value and action.
  size_t max_value_idx = max_element(pbvi_values,
      pbvi_values+belief_set_size) - pbvi_values;
  value = pbvi_values[max_value_idx];
  action = host_pbvi_actions[max_value_idx];

  // Free allocated memory.
  free(pbvi_values);
  checkCudaErrors(cudaFree(dev_belief));
  checkCudaErrors(cudaFree(dev_pbvi_values));
  checkCudaErrors(cublasDestroy(cublas_handle));

  return;
}

void savePbviDataToFile(const uint32_t height, const uint32_t width) {

  // Save the alpha vectors.
  FILE* pbvi_alphas_fid = fopen("pbvi_alphas", "w");
  for (uint32_t b = 0; b < belief_set_size; ++b) {
    for (uint32_t s = 0; s < height*width; ++s)
      fprintf(pbvi_alphas_fid, "%15.8f", host_pbvi_alphas[b*height*width+s]);
    fprintf(pbvi_alphas_fid, "\n");
  }
  fclose(pbvi_alphas_fid);

  // Save the actions.
  FILE* pbvi_actions_fid = fopen("pbvi_actions", "w");
  for (uint32_t b = 0; b < belief_set_size; ++b)
    fprintf(pbvi_actions_fid, "%10u\n", host_pbvi_actions[b]);
  fclose(pbvi_actions_fid);

  return;
}

bool loadPbviDataFromFile(const uint32_t height, const uint32_t width) {

  // Load the alpha vectors.
  FILE* pbvi_alphas_fid = fopen("pbvi_alphas", "r");
  for (uint32_t b = 0; b < belief_set_size; ++b) {
    for (uint32_t s = 0; s < height*width; ++s)
      if (fscanf(pbvi_alphas_fid, "%f", host_pbvi_alphas+b*height*width+s) == EOF) {
        printf(RED "Data dimension is not set properly...\n" RESET);
        return false;
      }
  }
  fclose(pbvi_alphas_fid);

  // Load the actions.
  FILE* pbvi_actions_fid = fopen("pbvi_actions", "r");
  for (uint32_t b = 0; b < belief_set_size; ++b) {
    if (fscanf(pbvi_actions_fid, "%u", host_pbvi_actions+b) == EOF) {
      printf(RED "Data dimension is not set properly...\n" RESET);
      return false;
    }
  }
  fclose(pbvi_actions_fid);

  // Upload the data to the device.
  checkCudaErrors(cudaMemcpy(dev_pbvi_alphas, host_pbvi_alphas,
        sizeof(float)*belief_set_size*height*width, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_pbvi_actions, host_pbvi_actions,
        sizeof(uint8_t)*belief_set_size, cudaMemcpyHostToDevice));

  return true;
}
