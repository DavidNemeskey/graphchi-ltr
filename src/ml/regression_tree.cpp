#include "ml/regression_tree.h"
#include "ml/learning_rate.h"
//#include <Eigen/util/Constants.h>

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
    data.conservativeResize(Eigen::NoChange, 2 * data.cols());
    outputs.conservativeResize(2 * outputs.size());
  }

  data.row(cols) = Map<RowVectorXd>(features, dimensions);
  outputs(cols) = output;
  cols++;
}

void RegressionTree::finalize_data() {
  data.conservativeResize(Eigen::NoChange, cols);
  outputs.conservativeResize(cols);
}

void RegressionTree::create_sorted() {
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
void RegressionTree::build_tree(double delta, size_t q) {
  create_sorted();
  tree = new RealNode(0);
  tree->output = outputs.sum() / outputs.size();
  tree->error  = (outputs - tree->output).pow(2).sum();
  ArrayXi valid = ArrayXi::Ones(sorted.rows());
  int max_id = 0;
  split_node(tree, valid, max_id, valid.size(), delta, q);
}

void RegressionTree::split_node(Node* node, ArrayXi& valid, int& max_id,
                                ArrayXi::Index num_docs, double delta, size_t q) {
  double min_error   = node->error;
  size_t min_feature = dimensions;
  double min_value   = 0;
  double min_left_error  = 0;
  double min_right_error = 0;
  ArrayXd::SegmentReturnType min_head = outputs.head(0);  // placeholder only
  ArrayXd::SegmentReturnType min_tail = outputs.tail(0);  // placeholder only
  size_t min_left_valids = 0;

  // TODO: to argument?
  /* Create a list of the outputs. */
  ArrayXd sorted_outputs(num_docs);

  /* Iterate through all features. */
  for (size_t f = 0; f < dimensions; f++) {
    /* The outputs sorted by the current feature. */
    for (size_t i = 0, j = 0; j < sorted_outputs.size(); i++) {
      if (valid(sorted(i, f)) == node->id) sorted_outputs(j++) = outputs(sorted(i, f));
    }

    /* Number of valids on the left side of the split. */
    size_t left_valids = 0;
    /* All possible split with that feature. */
    for (size_t split = 0; split < sorted.rows() - q; split++) {
      if (valid(sorted(split, f)) != node->id) continue;  // Document not under this node

      if (left_valids >= q) {
        auto head = sorted_outputs.head(left_valids);
        auto tail = sorted_outputs.tail(num_docs - left_valids);
        double left_error = (head - head.sum() / head.size()).pow(2).sum();
        double right_error = (tail - tail.sum() / tail.size()).pow(2).sum();
        double curr_error = left_error + right_error;
        if (curr_error < min_error) {
          min_error   = curr_error;
          min_feature = f;
          min_value   = data(sorted(split, f), f);
          min_left_error = left_error;
          min_right_error = right_error;
          min_head = head;
          min_tail = tail;
          min_left_valids = left_valids;
        }
      }

      left_valids++;
    }
  }

  if (min_error + delta < node->error) {
    node->feature_no  = min_feature;
    // TODO: handle enum node type
    ((RealNode*)node)->feature_val = min_value;

    int left_id  = ++max_id;
    int right_id = ++max_id;
    ArrayXi::Index left_docs  = 0;
    ArrayXi::Index right_docs = 0;
    // TODO: sparse / set / map
    for (ArrayXXi::Index i = 0; i < valid.size(); i++) {
      if (valid(i) == node->id) {
        if (data(i, min_feature) < min_value) {
          valid(i) = left_id;
          left_docs++;
        } else {
          valid(i) = right_id;
          right_docs++;
        }
      }
    }  // for

    node->left  = new RealNode(left_id);
    node->left->output = min_head.sum() / min_head.size();
    node->left->error = min_left_error;
    node->right = new RealNode(right_id);
    node->right->output = min_tail.sum() / min_tail.size();
    node->right->error = min_right_error;

    split_node(node->left, valid, max_id, left_docs, delta, q);
    split_node(node->right, valid, max_id, right_docs, delta, q);
  }
}

RegressionTree::Node::Node(int id_) : id(id_) {}

bool RegressionTree::Node::is_leaf() const {
  return left == NULL;
}

RegressionTree::Node::~Node() {
  if (left != NULL) {
    delete left;
    delete right;
  }
}

RegressionTree::RealNode::RealNode(int id_) : Node(id_) {}

RegressionTree::Comp::Comp(const ArrayXXd& data_, ArrayXXd::Index column_)
  : data(data_), column(column_) {}

bool RegressionTree::Comp::operator() (int i, int j) const {
  return data(i, column) < data(j, column);
}

