#include "ml/data_container.h"

using Eigen::Map;

DataContainer::DataContainer(size_t dimensions_)
  : dimensions(dimensions_), data(ArrayXXd(1000, dimensions_)),
    outputs(ArrayXd(1000)), rows_read(0) {}

void DataContainer::read_data_item(const Eigen::ArrayXd& features, const double& output) {
  /* Expand the data matrix. */
  if (data.rows() == rows_read) {
    data.conservativeResize(Eigen::NoChange, 2 * data.rows());
    outputs.conservativeResize(2 * outputs.size());
  }

  // TODO: size() check?
  data.row(rows_read) = features.head(dimensions);
  outputs(rows_read) = output;
  rows_read++;
}

void DataContainer::read_data_item(double* const& features, const double& output) {
  read_data_item(Map<ArrayXd>(features, dimensions), output);
}

void DataContainer::finalize_data() {
  data.conservativeResize(rows_read, Eigen::NoChange);
  outputs.conservativeResize(rows_read);
}

