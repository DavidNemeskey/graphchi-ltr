#include "ml/mart.h"
#include "ml/data_container.h"
#include "ml/learning_rate.h"

MART::MART(LearningRate* learning_rate_)
  : learning_rate(learning_rate_) {
  if (learning_rate == NULL) {
    learning_rate = new ConstantLearningRate(0.9);
  }
}

MART::~MART() {
  delete learning_rate;
}

void MART::learn(const DataContainer& data, size_t no_trees) {
  /* The outputs for all data points in @c data. */
  ArrayXd F       = ArrayXd::Zero(data.data().rows());
  /** The lambdas, alias the y_i's. */
  ArrayXd lambdas = ArrayXd::Zero(F.size());

  for (size_t k = 0; k < no_trees; k++) {
    for (ArrayXd::Index data_i = 0; data_i < lambdas.size(); data_i++) {

    }  // for data_i
    ReferenceDataContainer tree_data(data.dimensions, data.data(), lambdas);
  }  // for k
}
