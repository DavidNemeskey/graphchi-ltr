#include "lambdarank_optimize.h"

#include <algorithm>

bool NdcgOptimizer::rel_comp(const FeatureEdge& e1,
                             const FeatureEdge& e2) {
  return e1.relevance > e2.relevance;
}

bool NdcgOptimizer::score_comp(const FeatureEdge& e1,
                               const FeatureEdge& e2) {
  return e1.score > e2.score;
}

void NdcgOptimizer::compute(graphchi_vertex<TypeVertex, FeatureEdge>& v) {
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

double NdcgOptimizer::delta(graphchi_vertex<TypeVertex, FeatureEdge>& v,
                            int i, int j) {
  double ret = 0;
  ret = -ndcg_at_i(v, i, rank_map[i]) - ndcg_at_i(v, j, rank_map[j]) +
         ndcg_at_i(v, i, rank_map[j]) + ndcg_at_i(v, j, rank_map[i]);
  return ret / idcg;
}

double NdcgOptimizer::ndcg_at_i(graphchi_vertex<TypeVertex, FeatureEdge>& v,
                                int i, int rank) {
  return (pow(2, v.outedge(i)->get_data().relevance) - 1) /
         (log(rank + 2) / log(2));
}

