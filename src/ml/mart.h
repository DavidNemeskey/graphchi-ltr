#pragma once
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
 * In-memory MART.
 */

#include <cstdlib>
#include <Eigen/Dense>

#include "ndcg_optimizer.h"
#include "ml/boosting.h"

using Eigen::ArrayXXd;
using Eigen::ArrayXd;
using Eigen::ArrayXi;

class DataContainer;
class LearningRate;
class RegressionTree;

class MART {
public:
  MART(LearningRate* learning_rate=NULL);
  ~MART();

  void learn(const DataContainer& data, size_t no_trees);

private:
  /**
   * Returns the indices of the data items that belong to a different query than
   * the one before them; called by learn().
   */
  std::vector<ArrayXi::Index> queries(const DataContainer& data) const;

  /** The derivative of C over s_i == lambda_ij. */
  double dC_per_ds_i(const double S_ij, const double s_i, const double s_j);

  /** The learning rate function. */
  LearningRate* learning_rate;

  /**
   * The evaluation metric used to optimize LambdaMART.
   * @todo Accept metrics other than nDCG.
   */
  std::vector<RealNdcgOptimizer> metric;
  // TODO: remove sigma
  /** The (ignored) sigma parameter. */
  double sigma;

  /** The boosting container -- could be a parent class too. */
  Boosting trees;
};

