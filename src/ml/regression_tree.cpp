#include "ml/regression_tree.h"

#include <iostream>

#include "ml/data_container.h"
#include "ml/utils.h"

using Eigen::Map;
using Eigen::RowVectorXd;
using Eigen::ArrayXi;

RegressionTree::RegressionTree() : MlModel(), tree(NULL) {
}

RegressionTree::RegressionTree(RegressionTree& orig) {
  tree = orig.tree;
}

RegressionTree* RegressionTree::clone() {
  return new RegressionTree(*this);
}

RegressionTree::~RegressionTree() {
  delete tree;
}

void RegressionTree::create_sorted(const DataContainer& data) {
  /* First, create the sorted array, ... */
  sorted.resize(data.data().rows(), data.data().cols());
  ArrayXi tmp = ArrayXi::Constant(data.data().cols(), 0);
  for (size_t i = 0; i < sorted.rows(); i++) {
    sorted.row(i) = tmp;
    tmp += 1;
  }
  /* ... then sort! */
  auto begin = sorted.data();
  Comp comp(data.data());
  for (ArrayXXd::Index col = 0; col < data.data().cols(); col++) {
    comp.column = col;
    std::sort(begin, begin + sorted.rows(), comp);
    begin += sorted.rows();
  }
}

ArrayXi RegressionTree::build_tree(const DataContainer& data, double delta, size_t q) {
  create_sorted(data);

  tree = new RealNode(0);
  tree->output = data.relevance().sum() / data.relevance().size();
  tree->error  = (data.relevance() - tree->output).pow(2).sum();
  ArrayXi valid = ArrayXi::Zero(sorted.rows());
  int max_id = 0;
  split_node(tree, data, valid, max_id, valid.size(), delta, q);

  sorted.resize(0, 0);
  num_nodes_ = static_cast<size_t>(max_id) + 1;
  return valid;
}

void RegressionTree::split_node(Node* node, const DataContainer& data,
                                ArrayXi& valid, int& max_id,
                                ArrayXi::Index num_docs, double delta, size_t q) {
  double min_error   = node->error;
  size_t min_feature = data.dimensions;
  const ArrayXXd& data_    = data.data();
  const ArrayXXd& outputs_ = data.relevance();
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
  for (size_t f = 0; f < data.dimensions; f++) {
    /* The outputs sorted by the current feature for the samples under the current node. */
    for (size_t i = 0, j = 0; j < sorted_outputs.size(); i++) {
      if (valid(sorted(i, f)) == node->id)
        sorted_outputs(j++) = outputs_(sorted(i, f));
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

      const double& curr_value = data_(sorted(split, f), f);

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
      last_value = data_(sorted(split, f), f);
    }  // for split
  }  // for features

  if (min_error + delta < node->error) {
    node->feature_no  = min_feature;
    // TODO: handle enum node type
    ((RealNode*)node)->feature_val = min_value;

    int left_id  = ++max_id;
    int right_id = ++max_id;
    // TODO: sparse / set / map
    for (ArrayXXi::Index i = 0; i < valid.size(); i++) {
      if (valid(i) == node->id) {
        if (data_(i, min_feature) < min_value) {
          valid(i) = left_id;
        } else {
          valid(i) = right_id;
        }
      }
    }  // for

    node->left  = new RealNode(left_id);
    node->left->output = min_head.sum() / min_head.size();
    node->left->error = min_left_error;
    node->right = new RealNode(right_id);
    node->right->output = min_tail.sum() / min_tail.size();
    node->right->error = min_right_error;

    split_node(node->left, data, valid, max_id, min_left_valids, delta, q);
    split_node(node->right, data, valid, max_id, num_docs - min_left_valids, delta, q);
  }
}

void RegressionTree::set_output(Node* node, double output) {
  node->output = output;
}
void RegressionTree::set_output(int node_id, double output)
    throw (std::invalid_argument) {
  if (!set_output(tree, node_id, output)) {
    throw std::invalid_argument("no such node");
  }
}

bool RegressionTree::set_output(Node* node, int node_id, double output)
    throw (std::invalid_argument) {
  if (node->id == node_id) {
    node->output = output;
    return true;
  } else if (node->left != NULL) {
    return (set_output(node->left, node_id, output) == true)
           ? true
           : set_output(node->right, node_id, output);
  } else {
    return false;
  }
}

// TODO: IMPLEMENT!
double RegressionTree::score(double* const& features) const {
  return 0;
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

size_t RegressionTree::num_nodes() {
  return num_nodes_;
}

RegressionTree::Node::Node(int id_) : id(id_), left(NULL), right(NULL) {}

bool RegressionTree::Node::is_leaf() const {
  return left == NULL;
}

RegressionTree::Node::~Node() {
  if (left != NULL) {
    delete left;
    delete right;
    left = right = NULL;
  }
}

RegressionTree::RealNode::RealNode(int id_) : Node(id_) {}

RegressionTree::Comp::Comp(const ArrayXXd& data_, ArrayXXd::Index column_)
  : data(data_), column(column_) {}

bool RegressionTree::Comp::operator() (int i, int j) const {
  return data(i, column) < data(j, column);
}

void test_regression_tree2() {
  InputDataContainer d(1);
  Eigen::ArrayXd f(1);
  f << 1.5;
  d.read_data_item(0, f, 1);
  f << 1.5;
  d.read_data_item(0, f, 1);
  f << 3;
  d.read_data_item(0, f, 1);
  f << 3;
  d.read_data_item(0, f, 2);
  f << 3;
  d.read_data_item(0, f, 2);
  f << 3;
  d.read_data_item(0, f, 2);
  d.finalize_data();

  RegressionTree r;
  r.build_tree(d, 0, 1);
  std::cout << r.str();
}

void test_regression_tree() {
  InputDataContainer d(3);
  Eigen::ArrayXd f(3);
  f << 1, 1, 8;
  d.read_data_item(0, f, 1);
  f << 2, 3, 7;
  d.read_data_item(0, f, 2);
  f << 3, 4, 4;
  d.read_data_item(0, f, 2);
  f << 4, 2, 3;
  d.read_data_item(0, f, 1);
  f << 5, 6, 1;
  d.read_data_item(0, f, 3);
  f << 6, 8, 2;
  d.read_data_item(0, f, 4);
  f << 7, 5, 5;
  d.read_data_item(0, f, 3);
  f << 8, 7, 6;
  d.read_data_item(0, f, 4);
  d.finalize_data();

  // Core dump if it's just RegressionTree r; TODO: why?!
  RegressionTree* r = new RegressionTree();
  ArrayXi iii = r->build_tree(d, 0, 2);
  std::cout << r->str();
  std::cout << "num_nodes " << r->num_nodes() << std::endl;
  std::cout << "mapping " << std::endl << iii << std::endl;
}

