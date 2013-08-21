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
 * This file contains the root classes of the machine learning model hierarchy.
 */

#include "ml/ml_model.h"
#include "ml/learning_rate.h"

MlModel::MlModel(size_t dimensions, LearningRate* learning_rate)
  : dimensions(dimensions), learning_rate(learning_rate) {
  if (learning_rate == NULL) {
    learning_rate = new ConstantLearningRate(0.9);
  }
}

MlModel::~MlModel() {
  delete learning_rate;
}

