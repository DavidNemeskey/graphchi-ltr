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
 * Information retrieval measure optimizers for the LambdaXXX models.
 * @todo This is the third Ndcg computer -- merge with the other two.
 */

#include <vector>
#include <Eigen/Dense>

using Eigen::ArrayXd;

/**
 * The order the functions must be called:
 * 1. compute_idcg
 * 2. rankings
 * 3. delta or compute_ndcg
 */
class RealNdcgOptimizer {
public:
  /** Computes the ideal DCG, required for nDCG, and stores it. */
  void compute_idcg(ArrayXd relevance);

  /** Sorts the documents according to their rankings. */
  void rankings(const ArrayXd& outputs);

  /** Initializes the object (calls compute_idcg()). */
  inline void initialize(const ArrayXd& relevance) {
    compute_idcg(relevance);
  }
  /** Initializes the object (calls compute_idcg() rankings()). */
  inline void initialize(const ArrayXd& relevance, const ArrayXd& outputs) {
    compute_idcg(relevance);
    rankings(outputs);
  }

  /**
   * Returns the delta in the nDCG score if document @p i and @p j change
   * places in the ranking.
   */
  double delta(const ArrayXd& relevance, ArrayXd::Index i, ArrayXd::Index j);

  /** Computes the nDCG. */
  double compute_ndcg(const ArrayXd& relevance);

//private:
  /**
   * Computes the contribution of document @p i in the DCG score at rank
   * @p rank.
   */
  inline double dcg_at_i(const ArrayXd& relevance,
                         ArrayXd::Index i, ArrayXd::Index rank);
  /** The value of log(2), pre-computed. */
  static const double log_2;
  /** The ideal DCG. */
  double idcg;
  /** The document indices in their ranking order. */
  /** A document index -> rank map. */
  std::vector<ArrayXd::Index> ranking_order;
};

