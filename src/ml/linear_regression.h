#ifndef DEF_LINEAR_REGRESSION_H
#define DEF_LINEAR_REGRESSION_H
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

#include <vector>
#include <Eigen/Dense>

#include "ml_model.h"

using Eigen::VectorXd;

/**
 * A simple linear regression model.
 * @todo regularization!
 * @todo use Eigen vectors
 */
class LinearRegression : public DifferentiableModel {
protected:
  LinearRegression(LinearRegression& orig);

public:
  LinearRegression(size_t dimensions, LearningRate* learning_rate=NULL);

  LinearRegression* clone();

  Gradient* get_gradient_object();

  double score(double* const& features) const;

  /** Prints the weights. */
  std::string str() const;

//protected:
  /** The weight vector. Size is dimensions + 1, the last item is the noise. */
  VectorXd weights;

  friend class LinearRegressionGradient;
};

class LinearRegressionGradient : public Gradient {
public:
  LinearRegressionGradient(LinearRegression& parent);

  /** Resets the gradients to 0. */
  void reset();

  /** Computes the gradients. */
  void update(double* const& features, double output, double mult=1);

  std::string str() const;

protected:
  /** Updates the parent with the gradients. */
  void __update_parent(size_t num_items);

private:
  /** The gradients. */
  VectorXd gradients;
};

#endif
