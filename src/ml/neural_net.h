#ifndef DEF_NEURAL_NET_H
#define DEF_NEURAL_NET_H
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
 * A simple neural network with a single hidden layer.
 */

#include <vector>

#include "ml_model.h"

#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/** Stores the weights between two processing layers. Size: input x output. */
typedef MatrixXd WeightMatrix;
/**
 * Stores the weights of the output layer.
 * @todo delete if not needed
 */
typedef VectorXd WeightVector;

/**
 * A model based on a neural network. It has the following restrictions:
 *
 * 1. a single hidden layer
 * 2. the same activation function in all layers
 *
 * @todo activation function to separate class
 * @todo regularization!
 * @todo use the eigen library to speed up computation
 */
class NeuralNetwork : public MlModel {
public:
  /** @param[in] hidden_neurons the number of neurons in the hidden layer. */
  NeuralNetwork(size_t dimensions, size_t hidden_neurons,
                double learning_rate=0.0001);

  inline double score(double* const& features) const;

  Gradient* get_gradient_object();

  /** The sigma function... */
  double sigma(double x) const;
  /** ... and its derivative. */
  double sigma_deriv(double x) const;
  /** The inverse of the sigma (logistic) function. */
  double logit(double x) const;

private:
  /**
   * Scores the document and puts the outputs of layer 1 to @p outputs1.
   * score() invokes this method with @c outputs as @p outputs1.
   */
  double score_inner(double* const& features,
                     VectorXd& outputs1) const;
//                     std::vector<double>& outputs1) const;
  /** Initializes the individual weights to random numbers between 0.1 and 1. */
  void initialize_weights(size_t hidden_neurons);

private:
  /** Parameter for @c sigma. */
  double K;
  /**
   * Weights of the first (and only) hidden layer. An
   * dimensions x hidden_neurons-sized matrix.
   */
  WeightMatrix w1;
  /** Weights of the output layer. Its length == hidden_neurons. */
  WeightVector wy;
  /** Outputs of the hidden layer. Filled by score(), needed by update(). */
  mutable VectorXd outputs;

  friend class NeuralNetworkGradient;
  friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork& nn);
};

std::ostream& operator<<(std::ostream& os, const NeuralNetwork& nn);

class NeuralNetworkGradient : public Gradient {
public:
  NeuralNetworkGradient(NeuralNetwork& parent);

  /** Resets the gradients to 0. */
  void reset();

  void update(double* const& features, double y, double mult=1);

  /** Updates the parent with the gradients. */
  void update_parent();

private:
  /** Reference to the parent. */
  NeuralNetwork& parent;
  /**
   * The outputs of the first layer. Needed as we need to run our parent's
   * score method to compute all gradients.
   */
  VectorXd outputs;
  /** Gradients for the first layer. */
  WeightMatrix gradients1;
  /** Gradients for the output layer. */
  WeightVector gradientsy;
};

#endif
