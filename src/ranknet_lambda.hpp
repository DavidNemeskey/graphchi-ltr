#ifndef DEF_RANKNET_LAMBDA_H
#define DEF_RANKNET_LAMBDA_H
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
 * The sped-up RankNet algorithm, as described in Christopher J.C. Burges.
 * From RankNet to LambdaRank to LambdaMART: An Overview. 2010. In the paper, it
 * is adviced that this implementation is used for mini-batch learning, though I
 * don't think the arguments are valid in our case, since we are already doing
 * batch learning.
 */

#include <cmath>

#include "ltr_algorithm.hpp"

class RankNetLambda : public LtrAlgorithm {
public:
  /** @param[in] sigma parameter of the sigmoid. */
  RankNetLambda(DifferentiableModel* model, EvaluationMeasure* eval,
                StoppingCondition stop, LtrRunningPhase phase=TRAINING,
                double sigma=1)
      : LtrAlgorithm(model, eval, stop, phase), sigma(sigma) {
  }

  /**************************** Mathematics stuff *****************************/


  /** The derivative of C over s_i. */
  double dC_per_ds_i(const double S_ij, const double s_i, const double s_j) {
    return sigma * ((0.5 - 0.5 * S_ij) - 1 / (1 + exp(sigma * (s_i - s_j))));
  }

  /****************************** GraphChi stuff ******************************/

  /** The actual RankNet implementation. */
  virtual void compute_gradients(
      graphchi_vertex<TypeVertex, FeatureEdge> &query, Gradient* umodel) {
    std::vector<double> lambdas(query.num_outedges());
    std::vector<double> s_is(query.num_outedges());

    /* First, we compute all the outputs. */
    for (int i = 0; i < query.num_outedges(); i++) {
      s_is[i] = get_score(query.outedge(i));
    }

    /* Now, we compute the errors (lambdas). */
    for (int i = 0; i < query.num_outedges() - 1; i++) {
      int rel_i = get_relevance(query.outedge(i));
      for (int j = i + 1; j < query.num_outedges(); j++) {
        int rel_j = get_relevance(query.outedge(j));
        if (rel_i != rel_j) {
          double S_ij = rel_i > rel_j ? 1 : -1;
          double lambda_ij = dC_per_ds_i(S_ij, s_is[i], s_is[j]);
          /* lambda_ij = -lambda_ji */
          lambdas[i] += lambda_ij;
          lambdas[j] -= lambda_ij;
        }
      }
    }

    /* Finally, the model update. */
    for (int i = 0; i < query.num_outedges(); i++) {
      umodel->update(query.outedge(i)->get_vector()->get_data(), s_is[i], lambdas[i]);
    }
  }

protected:
  /** The sigma parameter of the sigmoid. */
  double sigma;
};

#endif
