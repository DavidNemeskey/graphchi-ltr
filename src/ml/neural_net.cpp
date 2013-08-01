#include "ml/neural_net.h"
#include <iostream>
#include <iterator>
#include <functional>

using Eigen::Map;
using Eigen::RowVectorXd;

namespace {
struct SigmaFunctor {
  SigmaFunctor(double K) : K(K) {}
  double operator()(double x) const { return 1 / (1 + exp(-K * x)); }
private:
  double K;
};

void print_vec(const std::string& s, const VectorXd& v) {
  std::cout << s << " ";
  for (VectorXd::Index i = 0; i < v.size(); i++) {
    std::cout << v(i) << " ";
  }
  std::cout << std::endl;
}
};

NeuralNetwork::NeuralNetwork(size_t dimensions, size_t hidden_neurons,
                             double learning_rate, Activation* act_fn)
    : MlModel(dimensions, learning_rate), hidden_neurons(hidden_neurons) {
  initialize_weights(hidden_neurons);
  outputs = VectorXd::Zero(hidden_neurons);
  afn.reset(act_fn != NULL ? act_fn : new Sigma(1));
}

double NeuralNetwork::score(double* const& features) const {
  return score_inner(features, outputs);
}

double NeuralNetwork::score_inner(double* const& features,
                                  VectorXd& outputs1) const {
  outputs1 = Map<RowVectorXd>(features, dimensions) * w1.topRows(w1.rows() - 1)
             + w1.bottomRows(1);  // noise
//  for (VectorXd::Index i = 0; i < outputs1.size(); i++) {
//    std::cout << "outputs[" << i << "] == " << outputs1[i] << std::endl;
//  }
  outputs1 = outputs1.unaryExpr(afn->act());  // TODO: into the previous expression
//  for (VectorXd::Index i = 0; i < outputs1.size(); i++) {
//    std::cout << "sigma(outputs[" << i << "]) == " << outputs1[i] << std::endl;
//  }

  double y = 0;
  y = outputs1.transpose() * wy.head(hidden_neurons)  // TODO: one line
                           + wy(1);  // noise
  y = afn->act()(y);

  return y;
}

Gradient* NeuralNetwork::get_gradient_object() {
  return new NeuralNetworkGradient(*this);
}

void NeuralNetwork::initialize_weights(size_t hidden_neurons) {
  std::uniform_real_distribution<double> unif(0.1, 1.0);
  std::default_random_engine re;
  re.seed(1001);

  /* +1 for noise input: noise */
  w1.resize(dimensions + 1, hidden_neurons);
  for (WeightMatrix::Index i = 0; i < w1.rows(); i++) {
    for (WeightMatrix::Index j = 0; j < w1.cols(); j++) {
      w1(i, j) = unif(re);
    }
  }

  wy.resize(hidden_neurons + 1);
  for (size_t i = 0; i < hidden_neurons; wy(i++) = unif(re));
}

std::ostream& operator<<(std::ostream& os, const NeuralNetwork& nn) {
  os << "HIDDEN LAYER:" << std::endl;
  size_t hidden_neurons = nn.w1.cols();
  for (size_t i = 0; i < hidden_neurons; i++) {
    os << "NEURON: " << i << ":";
    for (size_t j = 0; j < nn.dimensions + 1; j++) {
      os << " " << nn.w1(j, i);
    }
    os << std::endl;
  }
  return os;
}

NeuralNetworkGradient::NeuralNetworkGradient(NeuralNetwork& parent)
    : parent(parent) {
  outputs.resize(parent.outputs.size());
  gradients1.resize(parent.w1.rows(), parent.w1.cols());
  gradientsy.resize(parent.wy.size());
  reset();
}

void NeuralNetworkGradient::reset() {
  gradients1 = MatrixXd::Zero(gradients1.rows(), gradients1.cols());
  gradientsy = VectorXd::Zero(gradientsy.size());
}

void NeuralNetworkGradient::update(double* const& features,
                                   double y, double mult) {
  /* Have to run score() again to fill up the outputs vector... */
  parent.score_inner(features, outputs);

//  std::cout << "updating " << features[0] << ", " << features[1] << ", " << features[2] << "..." << std::endl;
//  std::cout << "Outputs ";
//  for (VectorXd::Index i = 0; i < outputs.size(); i++) {
//    std::cout << outputs(i) << " ";
//  }
////  std::copy(outputs.begin(), outputs.end(),
////            std::ostream_iterator<double>(std::cout, " "));
//  std::cout << std::endl;

  /* First, let's update the output layer. */
  //double deltay = y * (1 - y);
  double deltay = parent.afn->deriv()(y);
//  std::cout << "deltay == " << deltay << std::endl;
  double plmdy = parent.learning_rate * mult * deltay;

  gradientsy.head(parent.hidden_neurons) += plmdy * outputs;
  gradientsy(parent.hidden_neurons) += plmdy;  // noise
//  std::cout << "Updating wy -= " << parent.learning_rate << " * " << mult
//            << " * " << deltay << " * outputs" << std::endl;
//  print_vec("Gradientsy:", gradientsy);

  //VectorXd deltah = (outputs.array() * (1 - outputs.array())).matrix();
  VectorXd deltah = outputs.unaryExpr(parent.afn->deriv());
//  print_vec("deltah:", deltah);
  /* y'(1) * w(2) -- shouldn't be matched like this, not readable */
  VectorXd deltah_wy = (parent.wy.head(parent.hidden_neurons).array() * deltah.array());
//  print_vec("parent.wy:", parent.wy);
//  print_vec("deltah_wy:", deltah_wy);
//  std::cout << "Updating neurons with lr " << parent.learning_rate
//            << " * deltay " << deltay << " * mult " << mult << std::endl;
  gradients1.topRows(gradients1.rows() - 1) += plmdy *
                Map<VectorXd>(features, parent.dimensions) * deltah_wy.transpose();
  gradients1.bottomRows(1) += plmdy * deltah_wy.transpose();  // noise
//  std::cout << "delta gradients1: " << std::endl << parent.learning_rate * mult * deltay * Map<VectorXd>(features, parent.dimensions) * deltah_wy.transpose() << std::endl;
//  std::cout << "gradients1: " << std::endl << gradients1 << std::endl;
}

void NeuralNetworkGradient::update_parent() {
  parent.w1 -= gradients1;
  parent.wy -= gradientsy;
}

