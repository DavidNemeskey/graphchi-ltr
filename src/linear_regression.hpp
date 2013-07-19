#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "ml_algorithm.hpp"

#include <algorithm>

/**
 * A simple linear regression model.
 * @todo regularization!
 * @todo use Eigen vectors
 */
class LinearRegression : public MlModel {
public:
  LinearRegression(size_t dimensions, double learning_rate=0.001)
    : MlModel(dimensions, learning_rate) {
    weights.resize(dimensions + 1, 1);
  }

//  LinearRegression(LinearRegression const& orig) {
//    copy_content(orig);
//  }

//  MlModel& operator=(MlModel const& orig) {
//    LinearRegression const& o = dynamic_cast<LinearRegression const&>(orig);
//    if (this != &o) {
//      copy_content(o);
//    }
//    return *this;
//  }

  MlModel* get_gradient_object() {
    return new LinearRegressionGradient(this);
  }

//  void copy_content(LinearRegression const& orig) {
//    dimensions    = orig.dimensions;
//    learning_rate = orig.learning_rate;
//    weights       = orig.weights;
//  }

  double score(double* const& features) const {
    double score = 0;
    for (size_t i = 0; i < dimensions; i++) {
      //DYN score += weights[i] * features.get(i);
      score += weights[i] * features[i];
    }
    score += weights[dimensions];
    //DYN features.set(features.size() - 1, score);
    return score;
  }

//  void update(double* const& features, double output, double mult=1) {
////    std::cout << "LINREG_UPDATE BEFORE ";
////    std::copy(weights.begin(), weights.end(), std::ostream_iterator<double>(std::cout, " "));
////    std::cout << std::endl;
//    for (size_t i = 0; i < dimensions; i++) {
////      //DYN weights[i] -= learning_rate * error * features.get(i);
////      std::cout << "weight[" << i << "] -= " << learning_rate << " * " <<
////        error << " * " << features.features[i] << " = " <<
////        learning_rate * error * features.features[i] << std::endl;
//      weights[i] -= learning_rate * mult * features[i];
//    }
//    weights[dimensions] -= learning_rate * mult;
////    std::cout << "LINREG_UPDATE AFTER ";
////    std::copy(weights.begin(), weights.end(), std::ostream_iterator<double>(std::cout, " "));
////    std::cout << std::endl;
//  }
// DEBUG private:
  /** The weight vector. Size is dimensions + 1, the last item is the noise. */
  std::vector<double> weights;
};

class LinearRegressionGradient : public LinearRegression, public Gradient {
public:
  LinearRegressionGradient(LinearRegression* parent)
    : MlModel(parent->dimensions, parent->learning_rate) {
    weights.resize(dimensions + 1, 1);  // Initialize to 0
  }

  /** Resets the weights to 0. */
  void reset() {
    std::fill(weights.begin(), weights.end(), 0);
  }

  double score(double* const& features) const {
    return parent->score(features);
  }

  /** Computes the gradients. */
  void update(double* const& features, double output, double mult=1) {
    for (size_t i = 0; i < dimensions; i++) {
      weights[i] += learning_rate * mult * features[i];
    }
    weights[dimensions] += learning_rate * mult;
  }

  void update_parent() {
    std::copy(weights.begin(), weights.end(), parent->weights.begin());
  }

private:
  /** Reference to the parent. */
  LinearRegression* parent;
};

#endif
