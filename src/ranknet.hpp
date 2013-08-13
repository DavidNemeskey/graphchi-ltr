#ifndef DEF_RANKNET_H
#define DEF_RANKNET_H
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
 * The RankNet algorithm, as described in Christopher J.C. Burges. From RankNet
 * to LambdaRank to LambdaMART: An Overview. 2010.
 *
 * @deprecated The compute_gradients() method computes the document scores for
 *             each pair, which incurs a performance penalty. Use the new
 *             RankNetLamba class that is free of this problem. This class is
 *             here only for educational reasons.
 */

#include <cmath>

#include "ltr_algorithm.hpp"

class RankNet : public LtrAlgorithm {
public:
  /** @param[in] sigma parameter of the sigmoid. */
  RankNet(MlModel* model, EvaluationMeasure* eval,
          LtrRunningPhase phase=TRAINING, double sigma=1)
      : LtrAlgorithm(model, eval, phase), sigma(sigma) {
  }

  /**************************** Mathematics stuff *****************************/

  /**
   * The P_ij in the paper; a sigmoid function of s_i - s_j.
   * @param[in] s_i the score given to the first document.
   * @param[in] s_j the score given to the second document.
   * @return P_ij.
   */
  double prob_ij(double s_i, double s_j) {
    return 1 / (1 + exp(-sigma * (s_i - s_j)));
  }

  /**
   * The cross-entropy cost function in its raw form. This and the sigmoid
   * functions can actually be merged into a much simpler method.
   * @param[in] P_ij the estimated probability that U_i > U_j.
   * @param[in] T_ij the known probability that U_i > U_j.
   * @return C.
   */
  double cost(double P_ij, double T_ij) {
    return -T_ij * log(P_ij) - (1 - T_ij) * log(1 - P_ij);
  }

  /** The derivative of C over s_i. */
  double dC_per_ds_i(const double S_ij, const double s_i, const double s_j) {
    return sigma * ((0.5 - 0.5 * S_ij) - 1 / (1 + exp(sigma * (s_i - s_j))));
  }

  /****************************** GraphChi stuff ******************************/

  /** The actual RankNet implementation. */
  virtual void compute_gradients(
      graphchi_vertex<TypeVertex, FeatureEdge> &query, Gradient* umodel) {
      // TODO Make the other version where the documents have edges between them
    for (int i = 0; i < query.num_outedges() - 1; i++) {
      int rel_i = get_relevance(query.outedge(i));
      double s_i   = get_score(query.outedge(i));
      for (int j = i + 1; j < query.num_outedges(); j++) {
        int rel_j = get_relevance(query.outedge(j));
        if (rel_i != rel_j) {
          double s_j = get_score(query.outedge(j));
          double S_ij = rel_i > rel_j ? 1 : -1;
          double error = dC_per_ds_i(S_ij, s_i, s_j);
//          std::cout << "DOC " << query.outedge(i)->vertex_id() << "(" << rel_i <<
//            ") vs " << query.outedge(j)->vertex_id() << "(" <<
//            rel_j << "), S_ij: " << S_ij << " s_i: " << s_i << ", s_j: " <<
//            s_j << ", prob_ij: " << prob_ij(s_i, s_j) << ", cost: " <<
//            cost(prob_ij(s_i, s_j), 0.5*(1 + S_ij)) << ", error: " << error << std::endl;
          //DYN umodel->update(*(query.outedge(i)->get_vector()), s_i, error);
          //DYN umodel->update(*(query.outedge(j)->get_vector()), s_j, -error);
          umodel->update(query.outedge(i)->get_data().features, s_i, error);
          umodel->update(query.outedge(j)->get_data().features, s_j, -error);
          /* error(s_i) = -error(s_j) */
        }
      }
    }
  }

private:
  /** The sigma parameter of the sigmoid. */
  double sigma;
};

#endif
