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

#include <float.h>
#include <math.h>

#include <set>
#include <boost/optional.hpp>
#include <boost/multi_array.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <path_planning_2d/helper_cuda/helper_cuda.h>

#include <ros/ros.h>
#include <path_planning_2d/pomdp_path_planning_2d.h>
#include <path_planning_2d/search_tree.h>

using namespace std;
namespace bst = boost;

typedef bst::multi_array_types::index_range IndexRange;

// Update the belief with Bayes filter.
// Externed from point_based_value_iteration.cu.
__global__
void cudaBayesBeliefUpdate(
    const uint32_t height, const uint32_t width,
    const float* const __restrict__ trans_prob,
    const float* const __restrict__ meas_prob,
    const float* const __restrict__ belief_in,
    const uint8_t u, const uint8_t z,
    float* const __restrict__ belief_out);

// Evalute the FIB and PBVI values respectively.
void  evaluateFibCpu(const uint32_t,
    const uint32_t, const float* const, float&, uint8_t&);
void  evaluateFibGpu(const uint32_t,
    const uint32_t, const float* const, float&, uint8_t&);
void  evaluatePbviCpu(const uint32_t,
    const uint32_t, const float* const, float&, uint8_t&);
void  evaluatePbviGpu(const uint32_t,
    const uint32_t, const float* const, float&, uint8_t&);

// Model data.
extern float* dev_trans_prob;
extern float* dev_meas_prob;
extern float* dev_stage_reward;

extern float* host_trans_prob;
extern float* host_meas_prob;
extern float* host_stage_reward;

template<typename T>
void printMatrix(const T* const ptr,
    const uint32_t height, const uint32_t width) {

  for (size_t i = 0; i < height; ++i) {
    printf("Row[%lu] ", i);
    for (size_t j = 0; j < width; ++j) {
      printf("%8.6f ", *(ptr+i*width+j));
    }
    cout << endl;
  }

  return;
}

__global__
void cudaInitializeCurandStates(
    const uint32_t n, curandState* const states) {
  const uint32_t idx = blockDim.x*blockIdx.x + threadIdx.x;
  if (idx >= n) return;

  curand_init(1234, idx, 0, states+idx);
  return;
}

__global__
void cudaForwardSampling(const uint32_t sample_num,
    const uint32_t height, const uint32_t width,
    curandState* const __restrict__ curand_states,
    const float* const __restrict__ trans_prob,
    const float* const __restrict__ meas_prob,
    const uint32_t* const __restrict__ samples,
    const uint8_t action,
    uint8_t* const __restrict__ observations) {

  const uint32_t idx = blockDim.x*blockIdx.x + threadIdx.x;
  if (idx >= sample_num) return;

  curandState local_curand_state = curand_states[idx];
  uint32_t sample1 = samples[idx];

  // Get the sample of the next state based on the
  // sample of the current state and the action.
  float local_trans_dist[9] = {0.0f};
  memcpy(local_trans_dist, trans_prob+sample1*81+action*9, sizeof(float)*9);
  for (uint8_t i = 1; i < 9; ++i)
    local_trans_dist[i] += local_trans_dist[i-1];

  float sample2_rand = curand_uniform(&local_curand_state);
  uint32_t sample2 = 0;
  for (uint8_t i = 0; i < 9; ++i) {
    if (sample2_rand <= local_trans_dist[i]) {
      sample2 = i;
      break;
    }
  }
  sample2 = sample1 + (sample2/3-1)*width + (sample2%3-1);

  // Get the sample of the observation based on the sample
  // of the next state.
  float local_meas_dist[16] = {0.0f};
  memcpy(local_meas_dist, meas_prob+16*sample2, sizeof(float)*16);
  for (uint8_t i = 1; i < 16; ++i)
    local_meas_dist[i] += local_meas_dist[i-1];

  float observation_rand = curand_uniform(&local_curand_state);
  uint8_t observation = 0;
  for (uint8_t i = 0; i < 16; ++i) {
    if (observation_rand <= local_meas_dist[i]) {
      observation = i;
      break;
    }
  }

  observations[idx] = observation;
  curand_states[idx] = local_curand_state;

  return;
}

namespace path_planning_2d {

// QNode static member variables.
uint32_t QNode::height = 0;
uint32_t QNode::width = 0;
float QNode::gamma = 0.0f;

// VNode static member variables.
uint32_t VNode::height = 0;
uint32_t VNode::width = 0;
float VNode::gamma = 0.0f;

QNode::QNode(const float* const b, const uint8_t a, VNode* const p):
  belief(new float[QNode::height*QNode::width]),
  action(a), parent(p){

  memcpy(belief, b, sizeof(float)*QNode::height*QNode::width);

  // Compute the stage reward for this action.
  bst::const_multi_array_ref<float, 2> stage_reward(
      host_stage_reward, bst::extents[QNode::height*QNode::width][9]);
  bst::const_multi_array_ref<float, 2>::const_array_view<1>::type
    action_reward = stage_reward[bst::indices[IndexRange()][action]];
  reward = inner_product(belief, belief+QNode::height*QNode::width,
      action_reward.begin(), 0.0f);

  // Sample some observations.
  const uint32_t sample_num = 50;
  uint8_t* const observations = new uint8_t[sample_num];
  forwardSampling(sample_num, observations);

  // Get the unique observations and the number for each observation.
  set<uint8_t> unique_observation_set;
  for (uint32_t i = 0; i < sample_num; ++i)
    unique_observation_set.insert(observations[i]);

  vector<uint8_t> unique_observations(unique_observation_set.size());
  vector<float> unique_observation_freq(unique_observation_set.size());

  auto unique_observation_set_iter = unique_observation_set.cbegin();
  for (size_t i = 0; i < unique_observation_set.size();
      ++i, ++unique_observation_set_iter) {
    unique_observations[i] = *unique_observation_set_iter;
    unique_observation_freq[i] = static_cast<float>(
        count(observations, observations+sample_num, *unique_observation_set_iter)) /
      static_cast<float>(sample_num);
  }
  delete[] observations;

  // Expand the child V nodes based on the sampled observations.
  dim3 blockPerGrid((QNode::width+7)/8, (QNode::height+7)/8);
  dim3 threadPerBlock(8, 8);

  children.assign(unique_observations.size(), nullptr);

  float *dev_belief_in, *dev_belief_out;
  float* belief_out = new float[QNode::height*QNode::width];
  checkCudaErrors(cudaMalloc(&dev_belief_in,
        sizeof(float)*QNode::height*QNode::width));
  checkCudaErrors(cudaMalloc(&dev_belief_out,
        sizeof(float)*QNode::height*QNode::width));
  checkCudaErrors(cudaMemcpy(dev_belief_in, belief,
        sizeof(float)*QNode::height*QNode::width, cudaMemcpyHostToDevice));

  for (size_t i = 0; i < unique_observations.size(); ++i) {
    const uint8_t z = unique_observations[i];
    const float weight = unique_observation_freq[i];

    cudaBayesBeliefUpdate<<<blockPerGrid, threadPerBlock>>>(
        QNode::height, QNode::width, dev_trans_prob, dev_meas_prob,
        dev_belief_in, a, z, dev_belief_out);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(belief_out, dev_belief_out,
          sizeof(float)*QNode::height*QNode::width, cudaMemcpyDeviceToHost));

    // Normalize the update belief.
    float belief_out_sum = accumulate(belief_out,
        belief_out+QNode::height*QNode::width, 0.0f);
    for_each(belief_out, belief_out+QNode::height*QNode::width,
        [&belief_out_sum](float& x)->void{ x/=belief_out_sum;});

    children[i] = new VNode(belief_out, z, weight, this);
  }

  delete[] belief_out;
  checkCudaErrors(cudaFree(dev_belief_in));
  checkCudaErrors(cudaFree(dev_belief_out));

  // Update the stats for this node.
  update();

  return;
}

QNode::~QNode() {
  delete[] belief;
  //for_each(children.begin(), children.end(),
  //    [](VNode* const vnode)->void{delete vnode;});
  return;
}

void QNode::update() {

  // Update the upper and lower bound.
  float upper_reward_to_go = 0.0f;
  float lower_reward_to_go = 0.0f;
  for (auto& vnode : children) {
    upper_reward_to_go += vnode->upper_bound * vnode->weight;
    lower_reward_to_go += vnode->lower_bound * vnode->weight;
  }

  upper_bound = reward + QNode::gamma*upper_reward_to_go;
  lower_bound = reward + QNode::gamma*lower_reward_to_go;

  // Update the heuristic of this node and the best node
  // to expand within the subtree of this node.
  heuristic = 0.0f;
  for (auto& vnode : children) {
    float vnode_heuristic =
      QNode::gamma * vnode->weight * vnode->heuristic;
    if (vnode_heuristic > heuristic) {
      heuristic = vnode_heuristic;
      vnode_to_expand = vnode->vnode_to_expand;
    }
  }

  // Update the depth of this node.
  uint32_t child_depth = 0;
  for (auto& vnode : children) {
    if (vnode->depth > child_depth) {
      child_depth = vnode->depth;
      depth = child_depth + 1;
    }
  }

  return;
}

void QNode::print() {
  printf(BLUE "Q node @[%p]\n" RESET, this);

  printMatrix<float>(belief, QNode::height, QNode::width);
  printf("action: %u\n", action);
  printf("upper bound: %f\n", upper_bound);
  printf("lower bound: %f\n", lower_bound);
  printf("heuristic: %f\n", heuristic);
  printf("reward: %f\n", reward);
  printf("depth: %u\n", depth);

  printf("V node to expand: %p\n", vnode_to_expand);
  printf("parent: %p\n", parent);
  printf("children[%lu]: ", children.size());
  for(auto& item : children) printf("%p ", item);
  printf("\n");

  // Recursively print the child nodes.
  for(auto& item : children) item->print();

  return;
}

void QNode::forwardSampling(
    const uint32_t sample_num, uint8_t* const observations) {

  dim3 blockPerGrid((sample_num+31)/32);
  dim3 threadPerBlock(32);

  // Set up the curand states to be used in the kernel function.
  curandState* dev_curand_states;
  checkCudaErrors(cudaMalloc(
        &dev_curand_states, sizeof(curandState)*sample_num));
  cudaInitializeCurandStates<<<blockPerGrid, threadPerBlock>>>(
      sample_num, dev_curand_states);
  checkCudaErrors(cudaDeviceSynchronize());

  // Get samples of the current state based on the belief.
  vector<float> distribution(QNode::height*QNode::width);
  partial_sum(belief, belief+QNode::height*QNode::width,
      distribution.data(), plus<float>());

  vector<uint32_t> samples(sample_num, 0);
  for (auto& sample : samples) {
    float sample_rand = (float)rand() / ((float)RAND_MAX+1.0f);
    const auto sample_iter = find_if(
        distribution.begin(), distribution.end(),
        [&sample_rand](const float& x)->bool{return x >= sample_rand;});
    sample = sample_iter - distribution.begin();
  }

  // Get a measurement sample corresponding to each sample
  // of the current state.
  uint32_t* dev_samples;
  checkCudaErrors(cudaMalloc(
        &dev_samples, sizeof(uint32_t)*sample_num));
  checkCudaErrors(cudaMemcpy(dev_samples, samples.data(),
        sizeof(uint32_t)*sample_num, cudaMemcpyHostToDevice));

  uint8_t* dev_observations;
  checkCudaErrors(cudaMalloc(
        &dev_observations, sizeof(uint8_t)*sample_num));

  cudaForwardSampling<<<blockPerGrid, threadPerBlock>>>(
      sample_num, QNode::height, QNode::width,
      dev_curand_states, dev_trans_prob, dev_meas_prob,
      dev_samples, action, dev_observations);
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(observations, dev_observations,
        sizeof(uint8_t)*sample_num, cudaMemcpyDeviceToHost));

  // Free the allocated memory.
  checkCudaErrors(cudaFree(dev_curand_states));
  checkCudaErrors(cudaFree(dev_samples));
  checkCudaErrors(cudaFree(dev_observations));

  return;
}

VNode::VNode(
    const float* const b, const uint8_t z,
    const float w, QNode* const p):
  belief(new float[VNode::height*VNode::width]),
  observation(z), weight(w), parent(p){

  memcpy(belief, b, sizeof(float)*VNode::height*VNode::width);

  // Compute the states of this node.
  uint8_t dummy_action = 0;
  evaluateFibCpu(VNode::height, VNode::width, belief, upper_bound, dummy_action);
  evaluatePbviCpu(VNode::height, VNode::width, belief, lower_bound, dummy_action);
  //printf("upper bound: %f\n", upper_bound);
  //printf("lower bound: %f\n", lower_bound);
  //upper_bound = 0.0f;
  //lower_bound = -5.0f / (1.0f-VNode::gamma);
  heuristic = upper_bound - lower_bound;
  vnode_to_expand = this;

  return;
}

VNode::~VNode() {
  delete[] belief;
  //for_each(children.begin(), children.end(),
  //    [](QNode* const qnode)->void{delete qnode;});
  return;
}

void VNode::update() {

  vector<float> children_upper_bound(children.size());
  vector<float> children_lower_bound(children.size());

  for (size_t i = 0; i < children.size(); ++i) {
    children_upper_bound[i] = children[i]->upper_bound;
    children_lower_bound[i] = children[i]->lower_bound;
  }

  size_t upper_max_idx = max_element(children_upper_bound.begin(),
      children_upper_bound.end()) - children_upper_bound.begin();
  size_t lower_max_idx = max_element(children_lower_bound.begin(),
      children_lower_bound.end()) - children_lower_bound.begin();

  // Update the stats of this node.
  upper_bound = children[upper_max_idx]->upper_bound;
  lower_bound = children[lower_max_idx]->lower_bound;

  heuristic = -FLT_MAX;
  for (const QNode* const child : children) {
    if (child->upper_bound <= lower_bound) continue;
    if (child->heuristic > heuristic) {
      heuristic = child->heuristic;
      vnode_to_expand = child->vnode_to_expand;
    }
  }

  // Update the depth of this node.
  uint32_t child_depth = 0;
  for (auto& qnode : children) {
    if (qnode->depth > child_depth) {
      child_depth = qnode->depth;
      depth = child_depth + 1;
    }
  }

  return;
}

void VNode::expand() {

  // Initialize one child Q node for each action.
  children.assign(9, nullptr);

  for (uint8_t a = 0; a < 9; ++a) {
    children[a] = new QNode(belief, a, this);
  }

  // Update the stats of this V node.
  update();

  return;
}

void VNode::print() {
  printf(GREEN "V node @[%p]\n" RESET, this);

  printMatrix<float>(belief, QNode::height, QNode::width);
  printf("observation: %u\n", observation);
  printf("upper bound: %f\n", upper_bound);
  printf("lower bound: %f\n", lower_bound);
  printf("heuristic: %f\n", heuristic);
  printf("weight: %f\n", weight);
  printf("depth: %u\n", depth);

  printf("V node to expand: %p\n", vnode_to_expand);
  printf("parent: %p\n", parent);
  printf("children[%lu]: ", children.size());
  for(auto& item : children) printf("%p ", item);
  printf("\n");

  // Recursively print the child nodes.
  for(auto& item : children) item->print();

  return;
}

// SearchTree static member variables.
uint32_t SearchTree::height = 0;
uint32_t SearchTree::width = 0;

SearchTree::SearchTree(const float* const b):
  root(new VNode(b, 0, 0.0f, nullptr)) {
  return;
}

SearchTree::~SearchTree() {

  if (root != nullptr) deleteSubTree(root);
  return;
}

void SearchTree::expand() {

  // Expand the most promising leaf V node.
  VNode* const vnode_to_expand = root->vnode_to_expand;
  vnode_to_expand->expand();

  // Update the search tree.
  VNode* vnode = vnode_to_expand;
  while (vnode->parent != nullptr) {
    QNode* const parent_qnode = vnode->parent;
    parent_qnode->update();
    VNode* const parent_vnode = parent_qnode->parent;
    parent_vnode->update();

    vnode = parent_vnode;
  }

  return;
}

void SearchTree::getOptimalAction(uint8_t& a, float& r) {

  a = 0;
  r = -FLT_MAX;

  // Find the action with the largest upper bound.
  for (const QNode* const qnode : root->children) {
    if (qnode->upper_bound > r) {
      r = qnode->upper_bound;
      a = qnode->action;
    }
  }

  return;
}

void SearchTree::deleteSubTree(QNode* const qnode) {

  for (VNode* const vnode : qnode->children) {
    if (vnode != nullptr) deleteSubTree(vnode);
  }

  delete qnode;

  return;
}

void SearchTree::deleteSubTree(VNode* const vnode) {

  for (QNode* const qnode : vnode->children) {
    if (qnode != nullptr) deleteSubTree(qnode);
  }

  delete vnode;

  return;
}

void SearchTree::update(const uint8_t a, const uint8_t z) {

  // Find the root Q node based on the action.
  // The rest of the Q nodes together with their sub-tree will be removed.
  QNode* root_qnode = nullptr;
  for (QNode* const qnode : root->children) {
    if (qnode->action == a) {
      root_qnode = qnode;
    } else {
      deleteSubTree(qnode);
    }
  }

  // Find the root V node based on the observation.
  // The rest of the V nodes together with their sub-tree will be removed.
  VNode* root_vnode = nullptr;
  for (VNode* const vnode : root_qnode->children) {
    if (vnode->observation == z) {
      root_vnode = vnode;
    } else {
      deleteSubTree(vnode);
    }
  }

  if (root_vnode != nullptr) {
    // If the v node corresponding to the new observation
    // already exists, simply update the root to the v node.
    delete root_qnode;
    delete root;

    root_vnode->parent = nullptr;
    root = root_vnode;

  } else {
    // If the v node with the corresponding new observation does
    // not exist. The whole tree from the root will be removed.
    // A new root will be initialized with the corresponding new belief.

    dim3 blockPerGrid((SearchTree::width+7)/8, (SearchTree::height+7)/8);
    dim3 threadPerBlock(8, 8);

    const float* const prev_belief = root->belief;
    float* const curr_belief = (float*)malloc(
        sizeof(float)*SearchTree::height*SearchTree::width);

    float *dev_prev_belief, *dev_curr_belief;
    checkCudaErrors(cudaMalloc((void**)(&dev_prev_belief),
          sizeof(float)*SearchTree::height*SearchTree::width));
    checkCudaErrors(cudaMalloc((void**)(&dev_curr_belief),
          sizeof(float)*SearchTree::height*SearchTree::width));
    checkCudaErrors(cudaMemcpy(dev_prev_belief, prev_belief,
          sizeof(float)*SearchTree::height*SearchTree::width, cudaMemcpyHostToDevice));

    cudaBayesBeliefUpdate<<<blockPerGrid, threadPerBlock>>>(
        SearchTree::height, SearchTree::width, dev_trans_prob,
        dev_meas_prob, dev_prev_belief, a, z, dev_curr_belief);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(curr_belief, dev_curr_belief,
          sizeof(float)*SearchTree::height*SearchTree::width, cudaMemcpyDeviceToHost));

    // Normalize the current belief.
    float curr_belief_sum = accumulate(curr_belief,
        curr_belief+QNode::height*QNode::width, 0.0f);
    for_each(curr_belief, curr_belief+QNode::height*QNode::width,
        [&curr_belief_sum](float& x)->void{ x/=curr_belief_sum;});

    root_vnode = new VNode(curr_belief, 0, 0.0f, nullptr);

    free(curr_belief);
    checkCudaErrors(cudaFree(dev_prev_belief));
    checkCudaErrors(cudaFree(dev_curr_belief));

    delete root_qnode;
    delete root;
    root = root_vnode;
  }

  return;
}

void SearchTree::print() {
  printf(YELLOW "Search tree @[%p]\n", this);
  printf("root node: %p\n", root);
  root->print();
  return;
}

} // End namespace path_planning_2d.
