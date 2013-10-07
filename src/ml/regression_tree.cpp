#include "ml/regression_tree.h"
#include <Eigen/Constants>

using Eigen::Map;
using Eigen::RowVectorXd;
using Eigen::ArrayXi;

RegressionTree::RegressionTree(size_t dimensions_, LearningRate* learning_rate_)
  : dimensions(dimensions_), learning_rate(learning_rate_),
    data(ArrayXXd(1000, dimensions_)), outputs(ArrayXd(1000)), cols(0),
    tree(NULL) {
  if (learning_rate == NULL) {
    learning_rate = new ConstantLearningRate(0.9);
  }
}

RegressionTree::~RegressionTree() {
  delete learning_rate;
  delete tree;
}

void RegressionTree::read_data_item(double* const& features, double& output) {
  /* Expand the data matrix. */
  if (data.cols() == cols) {
    data.conservativeResize(NoChange, 2 * data.cols());
    outputs.conservativeResize(2 * outputs.size());
  }

  data.row(cols) = Map<RowVectorXd>(features, dimensions);
  outputs(cols) = cls;
  cols++;
}

void RegressionTree::finalize_data() {
  data.conservativeResize(NoChange, cols);
  classes.conservativeResize(cols);
}

/** Creates the @c sorted array. */
void create_sorted() {
  /* First, create the sorted array, ... */
  sorted.resize(data.rows(), data.cols());
  ArrayXi tmp = ArrayXi::Constant(data.cols(), 0);
  for (size_t i = 0; i < sorted.rows(); i++) {
    sorted.row(i) = tmp;
    tmp += 1;
  }
  /* ... then sort! */
  auto begin = sorted.data();
  Comp comp(data);
  for (ArrayXXd::Index col = 0; col < data.cols(); col++) {
    comp.column = col;
    std::sort(begin, begin + sorted.rows(), comp);
    begin += sorted.rows();
  }
}

/**
 * @param[in] delta if the error does not decrease by at least @p delta, stop.
 * @param[in] q if one of the children would have at most q nodes, stop.
 */
void build_tree(double delta, size_t q) {
  create_sorted();
  tree = new RealNode();
  tree->output = outputs.sum() / outputs.size();
  tree->error  = (outputs - tree->output).pow(2);
  split_node(tree, delta, q);
}

/**
 * Recursively splits the nodes in the tree.
 */
void split_node(RealNode* node, double delta, size_t q) {
  double min_error   = node->error;
  size_t min_feature = dimensions;
  double min_value   = 0;

  ArrayXd sorted_outputs(outputs.size());

  /* Iterate through all features. */
  for (size_t f = 0; f < dimensions; f++) {
    // TODO: resort after each 
    /* The outputs sorted by the current feature. */
    for (size_t i = 0; i < sorted_outputs.size(); i++) {
      sorted_outputs(i) = outputs(sorted(i, f));
    }

    /* All possible split with that feature. */
    for (size_t split = q; split < sorted.rows() - q; split++) {
      size_t anti_split = sorted.rows() - split;
      auto head = sorted_outputs.head(split);
      auto tail = sorted_outputs.tail(sorted.rows() - split);
      double curr_error = (head - split).pow(2) / split +
                          (tail - anti_split).pow(2) / anti_split;
      if (curr_error < min_error) {
        min_error   = curr_error;
        min_feature = f;
        min_value   = data(sorted(split, f), f);
      }
    }
  }

  if (min_error + delta < node->error) {
    // TODO: split
  }
}

bool RegressionTree::Node::is_leaf() const {
  return left == NULL;
}

bool RegressionTree::Node::~Node() {
  if (left != NULL) {
    delete left;
    delete right;
  }
}

RegressionTree::Comp::Comp(ArrayXXd& data_, ArrayXXd::Index column_)
  : data(data_), column(column_) {}

bool RegressionTree::Comp::operator() (int i, int j) const {
  return data(i, column) < data(j, column);
}

