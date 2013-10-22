#include "ml/mart.h"

MART::MART(DataContainer* data_, LearningRate* learning_rate_)
  : data(data_), learning_rate(learning_rate_) {
  if (learning_rate == NULL) {
    learning_rate = new ConstantLearningRate(0.9);
  }
  F       = ArrayXd::Zero(data->data.rows());
  lambdas = ArrayXd::Zero(data->data.rows());
}

MART::~MART() {
  delete learning_rate;
}

void MART::learn(size_t no_trees) {
  for (size_t i = 0; i < no_trees; i++) {
  }
}
