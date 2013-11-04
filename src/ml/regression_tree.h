#pragma once
/**
 * @file
 * @author  David Nemeskey
 * @version 0.1
 *
 * @section LICENSE
 *
 * Copyright [2013] [MTA SZTAKI]
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * In-memory regression tree model.
 */

//#include "ml_model.h"

#include <Eigen/Dense>
#include <string>
#include <sstream>

#include "ml/ml_model.h"

class DataContainer;
class LearningRate;

using Eigen::ArrayXXd;
using Eigen::ArrayXd;
using Eigen::ArrayXXi;
using Eigen::ArrayXi;

// TODO: Create in-memory model ancestor class
class RegressionTree : public MlModel {
protected:
  /** A node in the tree; ancestor class. */
  struct Node {
    /** The node id. */
    int id;
    /** The index of the feature we test in this node. */
    size_t feature_no;
    /**
     * The output: the average of the outputs of the data points that fall under
     * the nodes.
     */
    double output;
    /** The LMSE error of the node. */
    double error;
    /** The sibling nodes. */
    struct Node* left;
    struct Node* right;

    Node(int id);
    /** Deletes the whole tree under the current node. */
    ~Node();
    /*
     * A node is a leaf if both @c left and @c right is @c NULL (== one of them,
     * as there are no intermediate nodes with only one child).
     */
    bool is_leaf() const;
  };

  /**
   * A node in the tree that represents a decision point based on real-valued
   * feature.
   */
  struct RealNode : public Node {
    /** The feature value we test against. */
    double feature_val;

    RealNode(int id);
  };

  /** Comparator for build_tree. */
  struct Comp {
    const ArrayXXd& data;
    ArrayXXd::Index column;

    Comp(const ArrayXXd& data, ArrayXXd::Index column=0);
    /**
     * Checks if <tt>data[i][column] < data[j][column]</tt>. Basically we are
     * creating an index of @c data's @c column'th column.
     */
    bool operator() (int i, int j) const;
  };

  /**
   * Shallow copies the tree.
   * @todo Real cloning.
   */
  RegressionTree(RegressionTree& orig);
  
public:
  RegressionTree();
  ~RegressionTree();

  RegressionTree* clone();

  /**
   * Builds the tree.
   * @param[in] data the data we are learning from.
   * @param[in] delta if the error does not decrease by at least @p delta, stop.
   * @param[in] q if one of the children would have at most q nodes, stop.
   */
  void build_tree(const DataContainer& data, double delta, size_t q);

  // TODO: does this model need gradients at all? I don't think so.

  double score(double* const& features) const;

  /** Prints the tree. */
  std::string str() const;

private:
  /**
   * Used by str(): recursively traverses the tree, and collects the string
   * representation in @param ss.
   */
  void str_inner(std::stringstream& ss, RealNode* node, size_t level) const;

  /** Creates the @c sorted array. */
  void create_sorted(const DataContainer& data);

  /**
   * Recursively splits the nodes in the tree.
   *
   * @param[in,out] valid an index of the valid rows -- each needs its won valid
   *                      vector.
   * @param[in,out] max_id the maximum node id thus far.
   * @param[in] num_docs the number of documents under the current node.
   */
  void split_node(Node* node, const DataContainer& data,
                  ArrayXi& valid, int& max_id,
                  ArrayXi::Index num_docs, double delta, size_t q);

  /** The learning rate function. */
  LearningRate* learning_rate;

  /**
   * An index of data. The value of the <tt>n</tt>th cell in each column in
   * @c sorted is the row of the <tt>n</tt>th smallest number in the same column
   * in @c data.
   */
  ArrayXXi sorted;

  /** The tree. */
  RealNode* tree;
};

void test_regression_tree();
void test_regression_tree2();

