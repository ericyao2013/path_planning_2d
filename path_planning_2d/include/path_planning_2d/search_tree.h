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
#include <vector>

#ifndef SSP_SEARCH_TREE_H
#define SSP_SEARCH_TREE_H

namespace path_planning_2d {

class QNode;
class VNode;

class QNode {
public:

  // Constructors.
  QNode(const float* const b, const uint8_t a, VNode* const p);
  QNode(const QNode&) = delete;
  QNode operator=(const QNode&) = delete;

  // Destructor.
  ~QNode();

  // Update this Q node based on its child nodes.
  void update();

  // Print the stats of this node.
  void print();

  // Size of the map.
  static uint32_t height;
  static uint32_t width;
  static float gamma;

public:

  // Belief and action associated with this Q node.
  float* const belief = nullptr;
  const uint8_t action = 0;

  // Parent and children of this Q node.
  VNode* parent = nullptr;
  std::vector<VNode*> children;

  // States of this node which should be updated if any
  // child node gets updated.
  float upper_bound = FLT_MAX;
  float lower_bound = -FLT_MAX;
  float heuristic = FLT_MIN;
  float reward = 0.0f;
  VNode* vnode_to_expand = nullptr;

  // 0-based depth of this Q node.
  uint32_t depth = 1;

private:

  // Sample a given number of observations o_{k+1} given
  // the belief b_k and the action a_k.
  void forwardSampling(
      const uint32_t sample_num, uint8_t* const observations);
};

class VNode {
public:

  // Constructors.
  VNode(const float* const b, const uint8_t z,
      const float w, QNode* const p);
  VNode(const VNode&) = delete;
  VNode operator=(const VNode&) = delete;

  // Destructor.
  ~VNode();

  // Update this V node based on its child nodes.
  void update();

  // Expand the V node.
  void expand();

  // Print the stats of this node.
  void print();

  // Size of the map.
  static uint32_t height;
  static uint32_t width;
  static float gamma;

public:

  // Belief and action associated with this V node.
  float* const belief = nullptr;
  const uint8_t observation = 0;

  // Parent and children of this V node.
  QNode* parent = nullptr;
  std::vector<QNode*> children;

  // States of this node which should be updated if any
  // child node gets updated.
  float upper_bound = FLT_MAX;
  float lower_bound = -FLT_MAX;
  float heuristic = FLT_MIN;
  VNode* vnode_to_expand = nullptr;
  const float weight = 0.0f;

  // 0-based depth of this V node.
  uint32_t depth = 0;
};

class SearchTree {
public:
  // Constructor.
  SearchTree(const float* const b);
  SearchTree(const SearchTree&) = delete;
  SearchTree operator=(const SearchTree&) = delete;

  // Destructor.
  ~SearchTree();

  // Expand the search tree.
  void expand();

  // Get the optimal action and the corresponding reward.
  void getOptimalAction(uint8_t& a, float& r);

  // Update the search tree with the new action
  // and the observation.
  void update(const uint8_t a, const uint8_t z);

  // Get the depth of the tree.
  uint32_t getDepth() { return root->depth; }

  // Print the nodes of the search tree.
  void print();

  static uint32_t height;
  static uint32_t width;

private:

  void deleteSubTree(QNode* const qnode);
  void deleteSubTree(VNode* const vnode);

  VNode* root = nullptr;
};

} // End namespace path_planning_2d.

#endif
