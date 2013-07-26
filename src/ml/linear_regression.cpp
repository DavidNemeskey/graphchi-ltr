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
//#include <iostream>
//#include <iterator>

LinearRegression::LinearRegression(
    size_t dimensions, double learning_rate)
  : MlModel(dimensions, learning_rate) {
  weights.resize(dimensions + 1, 1);
}

Gradient* LinearRegression::get_gradient_object() {
  return new LinearRegressionGradient(*this);
}

double LinearRegression::score(double* const& features) const {
  double score = 0;
  for (size_t i = 0; i < dimensions; i++) {
    score += weights[i] * features[i];
  }
  score += weights[dimensions];
  return score;
}

LinearRegressionGradient::LinearRegressionGradient(LinearRegression& parent)
    : parent(parent) {
  gradients.resize(parent.dimensions + 1, 0);  // Initialize to 0
}

void LinearRegressionGradient::reset() {
  std::fill(gradients.begin(), gradients.end(), 0);
}

void LinearRegressionGradient::update(double* const& features, double output, double mult) {
  for (size_t i = 0; i < parent.dimensions; i++) {
    gradients[i] += parent.learning_rate * mult * features[i];
  }
  gradients[parent.dimensions] += parent.learning_rate * mult;
}

void LinearRegressionGradient::update_parent() {
//  std::cout << "LINREG_UPDATE_PARENT";
//  std::copy(gradients.begin(), gradients.end(),
//            std::ostream_iterator<double>(std::cout, " "));
//  std::cout << std::endl;
  for (size_t i = 0; i < gradients.size(); i++) {
    parent.weights[i] -= gradients[i];
  }
}

