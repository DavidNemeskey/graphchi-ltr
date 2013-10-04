#ifndef DEF_REGRESSION_TREE_H
#define DEF_REGRESSION_TREE_H
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
 * In-memory regression tree model.
 */

#include "ml_model.h"

class RegressionTree : public MlModel {
protected:
  struct Node {
  };

  RegressionTree(RegressionTree& orig);
  
public
  RegressionTree(size_t dimensions, LearningRate* learning_rate=NULL);

  RegressionTree* clone();

  // TODO: does this model need gradients at all? I don't think so.

  double score(double* const& features) const;

  /** Prints the tree. */
  std::string str() const;
};

#endif
