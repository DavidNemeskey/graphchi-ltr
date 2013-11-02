#pragma once
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
 * A data container to which readers can write their data.
 * @todo A dense and a sparse container?
 */

#include <Eigen/Dense>

using Eigen::ArrayXXd;
using Eigen::ArrayXd;
using Eigen::ArrayXi;

/** Data container interface class. */
struct DataContainer {
  /**
   * Constructor.
   * @param dimensions the number of features in the data.
   */
  DataContainer(size_t dimensions);

  /** The number of features in the data. */
  size_t dimensions;
  /** The query ids of the data points. */
  virtual const ArrayXi& qids() const=0;
  /** The data. */
  virtual const ArrayXXd& data() const=0;
  /** The relevance judgements. */
  virtual const ArrayXd& relevance() const=0;
};

/**
 * DataContainer subclass that builds the data and relevance judgement matrices
 * via callback functions invoked by a reader or a similar class.
 */
class InputDataContainer : public DataContainer {
public:
  /**
   * Constructor.
   * @param dimensions the number of features in the data.
   */
  InputDataContainer(size_t dimensions);

  /**
   * Reads a data point: the features and the relevance value, and stores them
   * in the data matrix.
   */
  void read_data_item(const int& qid, double* const& features,
                      const double& relevance);
  void read_data_item(const int& qid, const Eigen::ArrayXd& features,
                      const double& relevance);

  /** Finalizes the data; to be called after all data points have been read. */
  void finalize_data();

  inline const ArrayXi& qids() const { return qids_; }
  inline const ArrayXXd& data() const { return data_; }
  inline const ArrayXd& relevance() const { return relevance_; }

private:
  /** The query ids of the data points. */
  ArrayXi qids_;
  /** The data, read fully into memory. */
  ArrayXXd data_;
  /** The relevance judgements associated with the data items in @c data. */
  ArrayXd relevance_;

  /** The number of rows read thus far. */
  ArrayXXd::Index rows_read;
};

/** Same as DataContainer, but the data is only referenced. */
class ReferenceDataContainer : public DataContainer {
public:
  ReferenceDataContainer(size_t dimensions, const ArrayXi& qids,
                         const ArrayXXd& data, const ArrayXd& relevance);
  ReferenceDataContainer(const DataContainer& data);

  inline const ArrayXi& qids() const { return qids_; }
  inline const ArrayXXd& data() const { return data_; }
  inline const ArrayXd& relevance() const { return relevance_; }

private:
  const ArrayXi& qids_;
  const ArrayXXd& data_;
  const ArrayXd& relevance_;
};

