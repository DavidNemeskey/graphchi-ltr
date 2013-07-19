#ifndef DEF_NEURAL_NET_H
#define DEF_NEURAL_NET_H

#include <cmath>
#include <random>
#include <iostream>

#include "ml_algorithm.hpp"

/** Stores the weights between two processing layers. Size: input x output. */
typedef std::vector<std::vector<double> > WeightMatrix;
/**
 * Stores the weights of the output layer.
 * @todo delete if not needed
 */
typedef std::vector<double> WeightVector;

/**
 * A model based on a neural network. It has the following constraints:
 *
 * 1. a single hidden layer
 * 2. the same activation function in all layers
 *
 * @todo activation function to separate class
 * @todo use the eigen library to speed up computation
 */
class NeuralNetwork : public MlModel {
public:
  /** @param[in] hidden_neurons the number of neurons in the hidden layer. */
  NeuralNetwork(size_t dimensions, size_t hidden_neurons,
                double learning_rate=0.0001)
    : MlModel(dimensions, learning_rate), K(1)
  {
    initialize_weights(hidden_neurons);
    outputs.resize(hidden_neurons);
  }

  void score(FeatureEdge& features) const {
    outputs.fill(0);
    for (size_t x = 0; x < dimensions; x++) {
      for (size_t h = 0; h < outputs.size(); h++) {
        outputs[h] += features.features[x] * w1[x][h];
      }
    }
    for (size_t i = 0; i < outputs.size(); i++) {
      outputs[i] = sigma(outputs[i]);
    }

    double y = 0;
    for (size_t h = 0; h < outputs.size(); h++) {
      y += outputs[h] * wy[h];
    }
    y = sigma(y);

    return y;
  }

  void update(FeatureEdge features, double y, double mult=1) {
    /* Have to run score() again to fill up the outputs vector... */
    score(features);

    /* The updated weights. */
    WeightMatrix new_w1;
    WeightVector new_wy;
    new_w1.resize(w1.size());
    for (size_t i = 0; i < w1.size(); i++) {
      new_w1[i].resize(w1[i].size());
      for (size_t j = 0; j < w1[i].size(); j++) {
        new_w1[i][j] = w1[i][j];
      }
    }
    new_wy.resize(wy.size());
    for (size_t i = 0; i < wy.size(); i++) {
      new_wy[i] = wy[i];
    }

    // TODO: clone() to maintain a reference to the original model
    /* First, let's update the output layer. */
    double deltay = y * (1 - y);
    for (size_t j = 0; j < wj.size(); j++) {
      /* sgm'(s) * d(s) / d(w_j). */
      new_wy[j] -= learning_rate * mult * deltay * outputs[j];
    }

    /* That was the easy part; now the hidden layer... */
    for (size_t h = 0; h < wj.size(); h++) {
      double deltah = outputs[h] * (1 - outputs[h]);
      for (size_t i = 0; i < new_w1.size(); i++) {
        new_w1[i][h] -= learning_rate * mult *
                        deltay * wy[h] * deltah * features.features[i];
      }
    }

    /* Update the real weights. */
  }

  virtual MlModel* clone() { return this; }
  virtual MlModel& operator=(MlModel const& orig) { return *this; }
  virtual void reset() {}
  virtual MlModel& operator+=(MlModel const& other) { return *this; }
  virtual MlModel& operator-=(MlModel const& other) { return *this; }

private:
  /** The sigma function... */
  double sigma(double x) const {
    return 1 / (1 + exp(-K * x));
  }
  /** ... and its derivative. */
  double sigma_deriv(double x) const {
    double fx = sigma(x);
    return fx * (1 - fx);
  }
  /** The inverse of the sigma (logistic) function. */
  double logit(double x) const {
    return log(x) - log(1 - x);
  }

  /** Initializes the individual weights to random numbers between 0.1 and 1. */
  void initialize_weights(size_t hidden_neurons) {
    std::uniform_real_distribution<double> unif(0.1, 1.0);
    std::default_random_engine re;
    re.seed(1001);

    w1.resize(dimensions);  // TODO: +1 noise
    for (size_t i = 0; i < dimensions; i++) {
      w1[i].resize(hidden_neurons);
      for (size_t j = 0; j < hidden_neurons; j++) {
        w1[i][j] = unif(re);
      }
    }

    wy.resize(hidden_neurons);
    for (size_t i = 0; i < hidden_neurons; wy[i++] = unif(re));
  }

private:
  /** Parameter for @c sigma. */
  double K;
  /** Weights of the first (and only) hidden layer. */
  WeightMatrix w1;
  /** Weights of the output layer. */
  WeightVector wy;

  /** Outputs of the hidden layer. Filled by score(), needed by update(). */
  std::vector<double> outputs;
};

#endif
