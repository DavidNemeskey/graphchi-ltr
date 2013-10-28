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

/** Data container interface class. */
struct DataContainer {
  /**
   * Constructor.
   * @param dimensions the number of features in the data.
   */
  DataContainer(size_t dimensions);

  /** The number of features in the data. */
  size_t dimensions;
  /** The data. */
  virtual const ArrayXXd& data() const=0;
  /** The outputs. */
  virtual const ArrayXd& outputs() const=0;
};

/**
 * DataContainer subclass that builds the data and output via callback functions
 * invoked by a reader or a similar class.
 */
class InputDataContainer : public DataContainer {
public:
  /**
   * Constructor.
   * @param dimensions the number of features in the data.
   */
  InputDataContainer(size_t dimensions);

  /**
   * Reads a data point: the features and the output value, and stores them in
   * the data matrix.
   */
  void read_data_item(double* const& features, const double& output);
  void read_data_item(const Eigen::ArrayXd& features, const double& output);

  /** Finalizes the data; to be called after all data points have been read. */
  void finalize_data();

  inline const ArrayXXd& data() const { return data_; }
  inline const ArrayXd& outputs() const { return outputs_; }

private:
  /** The data, read fully into memory. */
  ArrayXXd data_;
  /** The outputs associated with the data items in @c data. */
  ArrayXd outputs_;

  /** The number of rows read thus far. */
  ArrayXXd::Index rows_read;
};

/** Same as DataContainer, but the data is only referenced. */
class ReferenceDataContainer : public DataContainer {
public:
  ReferenceDataContainer(
      size_t dimensions, const ArrayXXd& data, const ArrayXd& outputs);
  ReferenceDataContainer(const DataContainer& data);

  inline const ArrayXXd& data() const { return data_; }
  inline const ArrayXd& outputs() const { return outputs_; }

private:
  const ArrayXXd& data_;
  const ArrayXd& outputs_;
};

