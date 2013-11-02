#include "ndcg_optimizer.h"

#include <iostream>

const double RealNdcgOptimizer::log_2 = log(2);

void RealNdcgOptimizer::compute_idcg(ArrayXd relevance) {
  std::sort(relevance.data(), relevance.data() + relevance.size(),
            std::greater<double>());
  idcg = 0;
  for (ArrayXd::Index i = 0; i < relevance.size(); i++) {
    idcg += dcg_at_i(relevance, i, i);
  }
}

void RealNdcgOptimizer::rankings(const ArrayXd& outputs) {
  /* First, we sort the outputs, and create a rank -> document index mapping. */
  std::vector<ArrayXd::Index> sorted(outputs.size());
  for (size_t i = 0; i < sorted.size(); i++) {
    sorted[i] = i;
  }
  std::sort(sorted.begin(), sorted.end(),
            [&outputs](const ArrayXd::Index& i, const ArrayXd::Index& j) -> bool
              {return outputs[i] > outputs[j]; });

  /* Now the inverse: the document index -> rank mapping. */
  // TODO: isn't there a simpler way?
  ranking_order.resize(outputs.size());
  for (size_t i = 0; i < sorted.size(); i++) {
    ranking_order[sorted[i]] = i;
  }
}

double RealNdcgOptimizer::delta(
    const ArrayXd& relevance, ArrayXd::Index i, ArrayXd::Index j) {
  double ret = 0;
  ret = - dcg_at_i(relevance, i, ranking_order[i]) 
        - dcg_at_i(relevance, j, ranking_order[j])
        + dcg_at_i(relevance, i, ranking_order[j])
        + dcg_at_i(relevance, j, ranking_order[i]);
  return ret / idcg;
}

double RealNdcgOptimizer::compute_ndcg(const ArrayXd& relevance) {
  double dcg = 0;
  for (size_t i = 0; i < ranking_order.size(); i++) {
    dcg += dcg_at_i(relevance, i, ranking_order[i]);
  }
  return dcg / idcg;
}

inline double RealNdcgOptimizer::dcg_at_i(
    const ArrayXd& relevance, ArrayXd::Index i, ArrayXd::Index rank) {
  return (pow(2, relevance(i)) - 1) / (log(rank + 2) / log_2);
}

//int main(int argc, char* argv[]) {
//  RealNdcgOptimizer r;
//  ArrayXd vr(4);
//  vr << 3, 2, 4, 1;
//  r.compute_idcg(vr);
//  std::cout << r.idcg << std::endl;
//  ArrayXd vo(4);
////  vo << 3, 2, 1, 4;
//  vo << 3, 2, 1, 4;
//  r.rankings(vo);
//  for (size_t i = 0; i < r.ranking_order.size(); i++) {
//    std::cout << r.ranking_order[i] << std::endl;
//  }
//  std::cout << "ndcg " << r.compute_ndcg(vr) << std::endl;
//  std::cout << "delta " << r.delta(vr, 2, 3) << std::endl;
//}
