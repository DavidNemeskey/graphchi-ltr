#ifndef DEF_ML_ALGORITHM_H
#define DEF_ML_ALGORITHM_H
/**
 * @file
 * @author  David Nemeskey
 * @version 0.1
 *
 * @section LICENSE
 *
 * Copyright [2013] [Carnegie Mellon University]
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
 * This file contains the machine learning models used by the LTR algorithms.
 * The implementations of the models may be split into separate files later.
 */

#include <algorithm>
#include <iterator>
#include <iostream>

class MlModel {
protected:
  /** Default constructor, do not use. */
  MlModel() {}

public:
  MlModel(size_t dimensions, double learning_rate=0.001)
    : dimensions(dimensions), learning_rate(learning_rate) {}
  // TODO: argument type to template? parameter
  /**
   * features[-2]: relevance, features[-1]: score -- these must not be used to
   * compute the score. The score field must be updated by this method.
   */
  virtual double score(FeatureEdge& features) const=0;

  /**
   * Updates the weights of the model.
   * @param[in] features the features of the current item.
   * @param[in] output the output for the current item, i.e. score(features).
   * @param[in] mult a multiplier for the gradients. The LTR algorithm, for
   *                 example, passes d C/d s_i here.
   */
  virtual void update(FeatureEdge features, double output, double mult=1)=0; 

  /**
   * Clones this object. Because we don't know the type of the model during
   * runtime, we cannot use a copy constructor for this.
   */
  virtual MlModel* clone()=0;
  /* We can do this the right way. */
  virtual MlModel& operator=(MlModel const& orig)=0;
  /** Resets the weights to 1. */
  virtual void reset()=0;

  /* For update: foreach (m in models) model += m; m /= len(models); */

  /** Adds the weights of @p other to this model's. */
  virtual MlModel& operator+=(MlModel const& other)=0;
  /** Subtracts the weights of @p other to this model's. */
  virtual MlModel& operator-=(MlModel const& other)=0;

protected:
  /** Dimensions of the feature vector. */
  size_t dimensions;
  /** The learning rate. */
  double learning_rate;
};

// TODO: learning_rate to class!
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

  LinearRegression(LinearRegression const& orig) {
    copy_content(orig);
  }

  MlModel& operator=(MlModel const& orig) {
    LinearRegression const& o = dynamic_cast<LinearRegression const&>(orig);
    if (this != &o) {
      copy_content(o);
    }
    return *this;
  }

  MlModel* clone() {
    return new LinearRegression(*dynamic_cast<LinearRegression*>(this));
  }

  void reset() {
    std::fill(weights.begin(), weights.end(), 1);
  }

  MlModel& operator+=(MlModel const& other) {
    LinearRegression const& o = dynamic_cast<LinearRegression const&>(other);
    for (size_t i = 0; i < o.weights.size(); i++) {
      weights[i] += o.weights[i];
    }
    return *this;
  }

  MlModel& operator-=(MlModel const& other) {
    LinearRegression const& o = dynamic_cast<LinearRegression const&>(other);
    for (size_t i = 0; i < o.weights.size(); i++) {
      weights[i] -= o.weights[i];
    }
    return *this;
  }

  void copy_content(LinearRegression const& orig) {
    dimensions    = orig.dimensions;
    learning_rate = orig.learning_rate;
    weights       = orig.weights;
  }

  double score(FeatureEdge& features) const {
    double score = 0;
    for (size_t i = 0; i < dimensions; i++) {
      //DYN score += weights[i] * features.get(i);
      score += weights[i] * features.features[i];
    }
    score += weights[dimensions];
    //DYN features.set(features.size() - 1, score);
    features.score = score;
    return score;
  }

  void update(FeatureEdge features, double output, double mult=1) {
//    std::cout << "LINREG_UPDATE BEFORE ";
//    std::copy(weights.begin(), weights.end(), std::ostream_iterator<double>(std::cout, " "));
//    std::cout << std::endl;
    for (size_t i = 0; i < dimensions; i++) {
//      //DYN weights[i] -= learning_rate * error * features.get(i);
//      std::cout << "weight[" << i << "] -= " << learning_rate << " * " <<
//        error << " * " << features.features[i] << " = " <<
//        learning_rate * error * features.features[i] << std::endl;
      weights[i] -= learning_rate * mult * features.features[i];
    }
    weights[dimensions] -= learning_rate * mult;
//    std::cout << "LINREG_UPDATE AFTER ";
//    std::copy(weights.begin(), weights.end(), std::ostream_iterator<double>(std::cout, " "));
//    std::cout << std::endl;
  }
// DEBUG private:
  /** The weight vector. Size is dimensions + 1, the last item is the noise. */
  std::vector<double> weights;
};

#endif
