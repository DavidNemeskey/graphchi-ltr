#ifndef DEF_LAMBDARANK_H
#define DEF_LAMBDARANK_H
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
 * The LambdaRank algorithm. Since it is very closely related to RankNet, we
 * derive it from that class.
 */

#include <cmath>

#include "ranknet_lambda.hpp"
#include "lambdarank_optimize.hpp"

class LambdaRank : public RankNetLambda {
public:
  /** @param[in] sigma parameter of the sigmoid. */
  LambdaRank(MlModel* model, EvaluationMeasure* eval,
             LtrRunningPhase phase=TRAINING, double sigma=1)
      : RankNetLambda(model, eval, phase, sigma) {
  }

  /****************************** GraphChi stuff ******************************/

  /** The actual RankNet implementation. */
  virtual void compute_gradients(
      graphchi_vertex<TypeVertex, FeatureEdge> &query, Gradient* umodel) {
    std::vector<double> lambdas(query.num_outedges());
    std::vector<double> s_is(query.num_outedges());

    /* First, we compute all the outputs... */
    for (int i = 0; i < query.num_outedges(); i++) {
      s_is[i] = get_score(query.outedge(i));
//      std::cout << "s[" << i << "] == " << s_is[i] << std::endl;
    }
    /* ...and the retrieval measure scores. */
    opt.compute(query);


    /* Now, we compute the errors (lambdas). */
    for (int i = 0; i < query.num_outedges() - 1; i++) {
      int rel_i = get_relevance(query.outedge(i));
      for (int j = i + 1; j < query.num_outedges(); j++) {
        int rel_j = get_relevance(query.outedge(j));
        if (rel_i != rel_j) {
          double S_ij = rel_i > rel_j ? 1 : -1;
          double lambda_ij = dC_per_ds_i(S_ij, s_is[i], s_is[j]) *
                             opt.delta(query, i, j) * 5;
          /* lambda_ij = -lambda_ji */
          lambdas[i] += lambda_ij;
          lambdas[j] -= lambda_ij;
        }
      }
    }

    /* Finally, the model update. */
    for (int i = 0; i < query.num_outedges(); i++) {
      // -lambdas[i], as C is a utility function in this case
      umodel->update(query.outedge(i)->get_data().features, s_is[i], -lambdas[i]);
    }
  }

private:
  /**
   * The information retrieval measure we are optimizing for.
   * @todo add more measures.
   */
  NdcgOptimizer opt;
};

#endif
