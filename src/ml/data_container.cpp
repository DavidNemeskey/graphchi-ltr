#include "ml/data_container.h"

using Eigen::Map;

DataContainer::DataContainer(size_t dimensions_) : dimensions(dimensions_) {}

InputDataContainer::InputDataContainer(size_t dimensions_)
  : DataContainer(dimensions_), data_(ArrayXXd(1000, dimensions_)),
    outputs_(ArrayXd(1000)), rows_read(0) {}

void InputDataContainer::read_data_item(const int& qid,
    const Eigen::ArrayXd& features, const double& output) {
  /* Expand the data matrix. */
  if (data_.rows() == rows_read) {
    data_.conservativeResize(Eigen::NoChange, 2 * data_.rows());
    outputs_.conservativeResize(2 * outputs_.size());
  }

  // TODO: size() check?
  data_.row(rows_read) = features.head(dimensions);
  outputs_(rows_read) = output;
  rows_read++;
}

void InputDataContainer::read_data_item(const int& qid,
    double* const& features, const double& output) {
  read_data_item(qid, Map<ArrayXd>(features, dimensions), output);
}

void InputDataContainer::finalize_data() {
  data_.conservativeResize(rows_read, Eigen::NoChange);
  outputs_.conservativeResize(rows_read);
}

ReferenceDataContainer::ReferenceDataContainer(
    size_t dimensions_, const ArrayXi& qids__,
    const ArrayXXd& data__, const ArrayXd& outputs__)
  : DataContainer(dimensions_), qids_(qids__), data_(data__), outputs_(outputs__) {}

ReferenceDataContainer::ReferenceDataContainer(const DataContainer& data)
  : DataContainer(data.dimensions), qids_(data.qids()), data_(data.data()),
    outputs_(data.outputs()) {}

