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
 * In-memory MART.
 */

#include <cstdlib>
#include <Eigen/Dense>

using Eigen::ArrayXXd;
using Eigen::ArrayXd;

class DataContainer;
class LearningRate;

class MART {
public:
  MART(LearningRate* learning_rate=NULL);
  ~MART();

  void learn(const DataContainer& data, size_t no_trees);

private:
  /** The learning rate function. */
  LearningRate* learning_rate;
};

