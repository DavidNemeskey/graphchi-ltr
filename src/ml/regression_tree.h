#ifndef DEF_REGRESSION_TREE_H
#define DEF_REGRESSION_TREE_H
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

#include "ml_model.h"

#include <Eigen/Dense>

class LearningRate;

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXXd;
using Eigen::ArrayXd;
using Eigen::ArrayXXi;

// TODO: Create in-memory model ancestor class
class RegressionTree {
protected:
  /** A node in the tree; ancestor class. */
  struct Node {
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
    struct Node* left, right;

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
  };

  /** Comparator for build_tree. */
  struct Comp {
    ArrayXXd::Index column;
    const ArrayXXd& data;

    Comp(const ArrayXXd& data, ArrayXXd::Index column=0);
    /**
     * Checks if <tt>data[i][column] < data[j][column]</tt>. Basically we are
     * creating an index of @c data's @c column'th column.
     */
    bool operator() (int i, int j) const;
  };

  RegressionTree(RegressionTree& orig);
  
public:
  RegressionTree(size_t dimensions, LearningRate* learning_rate=NULL);
  ~RegressionTree();

  /**
   * Reads a data point: the features and the output value, and stores them in
   * the data matrix.
   */
  void read_data_item(double* const& features, double& output);

  /** Finalizes the data; to be called after all data points have been read. */
  void finalize_data();

  // TODO: does this model need gradients at all? I don't think so.

  double score(double* const& features) const;

  /** Prints the tree. */
  std::string str() const;

private:
  /** Dimensions of the feature space. */
  size_t dimensions;
  /** The learning rate function. */
  LearningRate* learning_rate;

  /** The data, read fully into memory. */
  ArrayXXd data;
  /** The outputs associated with the data items in @c data. */
  ArrayXd outputs;
  /**
   * An index of data. The value of the <tt>n</tt>th cell in each column in
   * @c sorted is the row of the <tt>n</tt>th smallest number in the same column
   * in @c data.
   */
  ArrayXXi sorted;

  /** The number of columns in @c data and the number of elements in @c classes. */
  MatrixXd::Index cols;

  /** The tree. */
  RealNode* tree;
};

#endif
