#include "ml/regression_tree.h"

#include <iostream>

#include "ml/learning_rate.h"
#include "ml/utils.h"

using Eigen::Map;
using Eigen::RowVectorXd;
using Eigen::ArrayXi;

RegressionTree::RegressionTree(size_t dimensions_, LearningRate* learning_rate_)
  : dimensions(dimensions_), learning_rate(learning_rate_),
    data(ArrayXXd(1000, dimensions_)), outputs(ArrayXd(1000)), rows_read(0),
    tree(NULL) {
  if (learning_rate == NULL) {
    learning_rate = new ConstantLearningRate(0.9);
  }
}

RegressionTree::~RegressionTree() {
  delete learning_rate;
  delete tree;
}

void RegressionTree::read_data_item(double* const& features, const double& output) {
  /* Expand the data matrix. */
  if (data.rows() == rows_read) {
    data.conservativeResize(Eigen::NoChange, 2 * data.rows());
    outputs.conservativeResize(2 * outputs.size());
  }

  data.row(rows_read) = Map<RowVectorXd>(features, dimensions);
  outputs(rows_read) = output;
  rows_read++;
}

// TODO: merge the two
void RegressionTree::read_data_item(Eigen::ArrayXd& features, const double& output) {
  /* Expand the data matrix. */
  if (data.rows() == rows_read) {
    data.conservativeResize(Eigen::NoChange, 2 * data.rows());
    outputs.conservativeResize(2 * outputs.size());
  }

  // TODO: size() check?
  data.row(rows_read) = features.head(dimensions);
  outputs(rows_read) = output;
  rows_read++;
}

void RegressionTree::finalize_data() {
  data.conservativeResize(rows_read, Eigen::NoChange);
  outputs.conservativeResize(rows_read);
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
  ArrayXi valid = ArrayXi::Zero(sorted.rows());
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
//  ArrayXd::SegmentReturnType min_head = outputs.head(0);  // placeholder only
//  ArrayXd::SegmentReturnType min_tail = outputs.tail(0);  // placeholder only
  ArrayXd min_head, min_tail; 
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
    /*
     * The last value of the feature. Recorded for two reasons:
     * 1. we cannot put a cutting point between two equal-valued item;
     * 2. the check value will be halfway between the last and the current items.
     */
    double last_value = DNaN;
    /* 
     * All possible split with that feature -- we cannot start at q, because we
     * have to keep track of split.
     */
    for (size_t split = 0; split <= sorted.rows() - q; split++) {
      if (valid(sorted(split, f)) != node->id) continue;  // Document not under this node

      double& curr_value = data(sorted(split, f));

      /* last_value/1: skip if feature value is the same as last item's. */
      if (double_equals(last_value, curr_value)) {
        left_valids++;  // still, it was a valid item
        continue;
      }

      size_t right_valids = num_docs - left_valids;
      if (left_valids >= q && right_valids >= q) {
        auto head = sorted_outputs.head(left_valids);
        auto tail = sorted_outputs.tail(right_valids);
        double left_error = (head - head.sum() / head.size()).pow(2).sum();
        double right_error = (tail - tail.sum() / tail.size()).pow(2).sum();
        double curr_error = left_error + right_error;
        if (curr_error < min_error) {
          min_error   = curr_error;
          min_feature = f;
          // last_value/2.
          min_value   = isnan(last_value)
                        ? curr_value - epsilon
                        : (last_value + curr_value) / 2;
          min_left_error = left_error;
          min_right_error = right_error;
          min_head = head;
          min_tail = tail;
          min_left_valids = left_valids;
        }
      }

      left_valids++;
      last_value = data(sorted(split, f), f);
    }  // for split
  }  // for features

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

    if (left_docs >= 2 * q) {
      split_node(node->left, valid, max_id, left_docs, delta, q);
    }
    if (right_docs >= 2 * q) {
      split_node(node->right, valid, max_id, right_docs, delta, q);
    }
  }
}

std::string RegressionTree::str() const {
  std::stringstream ss;
  str_inner(ss, tree, 0);
  return ss.str();
}

void RegressionTree::str_inner(std::stringstream& ss,
                               RealNode* node, size_t level) const {
  for (size_t i = 0; i < level; i++) ss << "  ";
  ss << node->id;
  if (node->left == NULL) {
    ss << ": " << node->output << " (err: " << node->error << ")" << std::endl;
  } else {
    ss << ": " << node->output << " (err: " << node->error << ") $"
       << node->feature_no << " < " << node->feature_val << " ? " << std::endl;
    str_inner(ss, (RealNode*)node->left, level + 1);
    str_inner(ss, (RealNode*)node->right, level + 1);
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

void test_regression_tree2() {
  RegressionTree r(1);
  Eigen::ArrayXd f(1);
  f << 1.5;
  r.read_data_item(f, 1);
  f << 1.5;
  r.read_data_item(f, 1);
  f << 3;
  r.read_data_item(f, 1);
  f << 3;
  r.read_data_item(f, 2);
  f << 3;
  r.read_data_item(f, 2);
  f << 3;
  r.read_data_item(f, 2);
  r.finalize_data();
  r.build_tree(0, 1);
  std::cout << r.str();
}

void test_regression_tree() {
  RegressionTree r(3);
  Eigen::ArrayXd f(3);
  f << 1, 1, 8;
  r.read_data_item(f, 1);
  f << 2, 3, 7;
  r.read_data_item(f, 2);
  f << 3, 4, 4;
  r.read_data_item(f, 2);
  f << 4, 2, 3;
  r.read_data_item(f, 1);
  f << 5, 6, 1;
  r.read_data_item(f, 3);
  f << 6, 8, 2;
  r.read_data_item(f, 4);
  f << 7, 5, 5;
  r.read_data_item(f, 3);
  f << 8, 7, 6;
  r.read_data_item(f, 4);
  r.finalize_data();
  r.build_tree(0, 2);
  std::cout << r.str();
}

