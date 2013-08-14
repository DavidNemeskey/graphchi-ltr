#ifndef DEF_LAMBDARANK_OPTIMIZE_H
#define DEF_LAMBDARANK_OPTIMIZE_H
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
 * Information retrieval measure optimizers for LambdaRank. 
 * @todo Merge with evaluation_measures.hpp.
 */

#include <map>

#include "ltr_common.hpp"

/**
 * Computes the nDCG, and provides methods that return the delta when two items
 * are switched.
 */
class NdcgOptimizer {
public:
  /**
   * Computes the score for the query. This version computes the idcg, as well
   * as the document->rank map, as we need these to compute differences. It does
   * NOT compute the nDCG itself.
   */
  void compute(graphchi_vertex<TypeVertex, FeatureEdge>& v) {
    // TODO: FeatureEdge* to make it faster?
    std::vector<FeatureEdge> ranked(v.num_outedges());
    for (int i = 0; i < v.num_outedges(); i++) {
      ranked[i] = v.outedge(i)->get_data();
    }
    std::sort(ranked.begin(), ranked.end(), rel_comp);

    std::cout << "RANKING for query " << v.get_data().id << ": ";

    double dcg = 0;
    for (size_t i = 0; i < ranked.size(); i++) {
      std::cout << ranked[i].doc << "(" << ranked[i].relevance << "), ";
      //DYN dcg += (pow(2, best[i]->get(best[i]->size() - 2)) - 1) /
      dcg += (pow(2, ranked[i].relevance) - 1) /
             (log(i + 2) / log(2));
    }
    std::cout << std::endl;

    idcg = dcg;

    /* Jeeebus, is there no better way? */
    std::sort(ranked.begin(), ranked.end(), score_comp);
    rank_map.clear();
    std::map<double, int> ranking;
    for (int i = 0; i < v.num_outedges(); i++) {
      ranking[-v.outedge(i)->get_data().score] = i;
    }
    int rank = 0;
    for (std::map<double, int>::const_iterator it = ranking.begin();
         it != ranking.end(); ++it) {
      rank_map[it->second] = rank++;
    }
  }

  /**
   * Returns the delta in the nDCG score if document @p i and @p j change places
   * in the ranking.
   */
  double delta(graphchi_vertex<TypeVertex, FeatureEdge>& v, int i, int j) {
    double ret = 0;
    ret = -ndcg_at_i(v, i, rank_map[i]) - ndcg_at_i(v, j, rank_map[j]) +
           ndcg_at_i(v, i, rank_map[j]) + ndcg_at_i(v, j, rank_map[i]);
    return ret / idcg;
  }

  /** For inverse sorting by relevance. Needed to compute the iDCG. */
  static bool rel_comp(const FeatureEdge& e1, const FeatureEdge& e2) {
    return e1.relevance > e2.relevance;
  }

  /** For inverse sorting by score. Needed to build the rank map. */
  static bool score_comp(const FeatureEdge& e1, const FeatureEdge& e2) {
    return e1.score > e2.score;
  }

private:
  /**
   * Computes the contribution of document @p i in the ndcg score at rank
   * @p rank.
   */
  double ndcg_at_i(graphchi_vertex<TypeVertex, FeatureEdge>& v, int i,
                   int rank) {
    return (pow(2, v.outedge(i)->get_data().relevance) - 1) /
           (log(rank + 2) / log(2));
  }
  /** Stores the ideal DCG for the query. Required to compute nDCG. */
  double idcg;
  /** Document id -> rank map. */
  std::map<int, int> rank_map;
};

#endif
