#include "ml/mart.h"
#include "ml/data_container.h"
#include "ml/learning_rate.h"
#include "ml/regression_tree.h"

MART::MART(LearningRate* learning_rate_)
  : learning_rate(learning_rate_) {
  if (learning_rate == NULL) {
    learning_rate = new ConstantLearningRate(0.9);
  }
  sigma = 1;
}

MART::~MART() {
  delete learning_rate;
  trees.cleanup();
}

double MART::dC_per_ds_i(const double S_ij, const double s_i, const double s_j) {
  return sigma * ((0.5 - 0.5 * S_ij) - 1 / (1 + exp(sigma * (s_i - s_j))));
}

// TODO: This is already lambdamart, and not just because of the lambdas...
//       Factor it out!
void MART::learn(const DataContainer& data, size_t no_trees) {
  /* The outputs for all data points in @c data. */
  ArrayXd F       = ArrayXd::Zero(data.data().rows());
  /** The lambdas, alias the y_i's. */
  ArrayXd lambdas = ArrayXd::Zero(F.size());
  /** The w_i's (d^2C/ds_i^2). */
  ArrayXd w       = ArrayXd::Zero(F.size());

  std::vector<ArrayXi::Index> qid_indices = queries(data);

  /* Initialize the metrics. */
  metric.resize(qid_indices.size() - 1);
  for (size_t i = 0; i < metric.size(); i++) {
    metric[i].initialize(data.relevance().segment(
          qid_indices[i], qid_indices[i + 1] - qid_indices[i]));
  }

  /* And now the algorithm... */
  for (size_t k = 0; k < no_trees; k++) {
    /* Now, we compute the errors (lambdas). Go through all queries... */
    for (size_t qi = 0; qi < qid_indices.size() - 1; qi++) {
      metric[qi].rankings(F.segment(qid_indices[qi],
                                    qid_indices[qi + 1] - qid_indices[qi]));
      const ArrayXd& rel_v = data.relevance().segment(
          qid_indices[qi], qid_indices[qi + 1] - qid_indices[qi]);
      /* ... and then the i - ...*/
      for (ArrayXi::Index i = qid_indices[qi]; i < qid_indices[qi + 1] - 2; i++) {
        int rel_i = static_cast<int>(data.relevance()(i));
        /* ... - j pairs. */
        for (ArrayXi::Index j = i + 1; j < qid_indices[qi + 1] - 1; j++) {
          int rel_j = static_cast<int>(data.relevance()(j));
          if (rel_i != rel_j) {
            double S_ij = rel_i > rel_j ? 1 : -1;
            double delta_metric = fabs(metric[qi].delta(
                  rel_v, i - qid_indices[qi], j - qid_indices[qi]));
            double lambda_ij = dC_per_ds_i(S_ij, F(i), F(j)) * delta_metric;
            /* lambda_ij = -lambda_ji */
            lambdas(i) += lambda_ij;
            lambdas(j) -= lambda_ij;

            double rho_ij = -lambda_ij / (sigma * delta_metric);
            w(i) -= sigma * lambda_ij * (1 - rho_ij);  // simplified
          }
        }  // for j
      }  // for i
    }  // for qi

    RegressionTree* rt = new RegressionTree();
    // TODO: set these parameters? Or maybe use the ones in the papers?
    ReferenceDataContainer tree_data(data.dimensions, data.qids(),
                                     data.data(), lambdas);
    rt->build_tree(tree_data, 1e-6, 50);
    // TODO gamma
    trees.add_model(rt, learning_rate->get());
    /* Break if the end of the learning rate's interval is reached. */
    if (learning_rate->advance() == 0) {
      break;
    }
  }  // for k
}

std::vector<ArrayXi::Index> MART::queries(const DataContainer& data) const {
  const ArrayXi& qids = data.qids();
  std::vector<ArrayXi::Index> ret;
  int last_qid = qids(0) - 1;
  for (ArrayXd::Index i = 0; i < qids.size(); i++) {
    if (qids(i) != last_qid) {
      ret.push_back(i);
      last_qid = qids(i);
    }
  }
  /* Push back the index of the last + 1 element one for good measure. */
  ret.push_back(qids.size());
  return ret;
}

