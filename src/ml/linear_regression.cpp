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

#include <sstream>
#include <iostream>

#include "ml/learning_rate.h"
//#include <iterator>

using Eigen::Map;

LinearRegression::LinearRegression(
    size_t dimensions, LearningRate* learning_rate)
  : MlModel(dimensions, learning_rate) {
  weights = VectorXd::Constant(dimensions + 1, 1);
}

LinearRegression::LinearRegression(LinearRegression& orig) : MlModel(orig) {
  weights = orig.weights;
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

LinearRegression* LinearRegression::clone() {
  return new LinearRegression(*this);
}

std::string LinearRegression::str() const {
  std::ostringstream ss;
  ss << "LinearRegression (dim: " << dimensions << "):";
  for (VectorXd::Index i = 0; i < weights.size(); i++) {
    ss << " " << weights[i];
  }
  return ss.str();
}

LinearRegressionGradient::LinearRegressionGradient(LinearRegression& parent)
    : Gradient(parent) {
  reset();
}

void LinearRegressionGradient::reset() {
  gradients = VectorXd::Constant(
      static_cast<LinearRegression&>(parent).dimensions + 1, 0);
}

void LinearRegressionGradient::update(
    double* const& features, double output, double mult) {
  LinearRegression& p = static_cast<LinearRegression&>(parent);
  gradients.head(p.dimensions) +=
      Map<VectorXd>(features, p.dimensions) * mult * p.learning_rate->get();
  gradients[p.dimensions] += p.learning_rate->get() * mult;
}

void LinearRegressionGradient::__update_parent(size_t num_items) {
  std::cout << "LINREG_UPDATE_PARENT ";
  for (VectorXd::Index i = 0; i < gradients.size(); i++) {
    std::cout << gradients[i] / num_items << " ";
  }
//  std::copy(gradients.begin(), gradients.end(),
//            std::ostream_iterator<double>(std::cout, " "));
  std::cout << std::endl;
  static_cast<LinearRegression&>(parent).weights -= gradients / num_items;
}

std::string LinearRegressionGradient::str() const {
  std::ostringstream ss;
  ss << "LinearRegressionGradient (dim: "
     << static_cast<LinearRegression&>(parent).dimensions << "):";
  for (VectorXd::Index i = 0; i < gradients.size(); i++) {
    ss << " " << gradients[i];
  }
  return ss.str();
}

