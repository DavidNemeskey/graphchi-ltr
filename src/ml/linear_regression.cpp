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
 * A simple linear regression model.
 */

#include "linear_regression.h"
#include <iostream>
#include "ml/learning_rate.h"
//#include <iterator>

using Eigen::Map;

LinearRegression::LinearRegression(
    size_t dimensions, LearningRate* learning_rate)
  : MlModel(dimensions, learning_rate) {
  weights = VectorXd::Constant(dimensions + 1, 1);
}

Gradient* LinearRegression::get_gradient_object() {
  return new LinearRegressionGradient(*this);
}

double LinearRegression::score(double* const& features) const {
  double score = Map<VectorXd>(features, dimensions).transpose() *
                 weights.head(dimensions);
  score += weights[dimensions];
  return score;
}

LinearRegressionGradient::LinearRegressionGradient(LinearRegression& parent)
    : parent(parent) {
  reset();
}

void LinearRegressionGradient::reset() {
  gradients = VectorXd::Constant(parent.dimensions + 1, 0);
}

void LinearRegressionGradient::update(double* const& features, double output, double mult) {
  gradients.head(parent.dimensions) +=
      Map<VectorXd>(features, parent.dimensions) * mult * parent.learning_rate->get();
  gradients[parent.dimensions] += parent.learning_rate->get() * mult;
}

void LinearRegressionGradient::update_parent() {
  std::cout << "LINREG_UPDATE_PARENT";
  for (VectorXd::Index i = 0; i < gradients.size(); i++) {
    std::cout << gradients[i] << " ";
  }
//  std::copy(gradients.begin(), gradients.end(),
//            std::ostream_iterator<double>(std::cout, " "));
  std::cout << std::endl;
  parent.weights -= gradients;
}

