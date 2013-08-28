#ifndef DEF_ML_MODEL_H
#define DEF_ML_MODEL_H
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
#include <cstddef>  // size_t

#include "object.h"

class LearningRate;
class Gradient;

class MlModel : public virtual Object {
protected:
  /** Default constructor; do not use. */
//  MlModel();
  /** Copy constructor. */
  MlModel(MlModel& orig);

public:
  virtual ~MlModel();

  /**
   * Constructor.
   * @param dimensions the number of dimensions of the data.
   * @param learning_rate the learning rate strategy. If @c NULL, a constant
   *        learning rate of 0.9 is used.
   */
  MlModel(size_t dimensions, LearningRate* learning_rate=NULL);
  // TODO: argument type to template? parameter
  /**
   * Returns the score for an item with features @p features. 
   * @note double* const& = const double[]&
   */
  virtual double score(double* const& features) const=0;

  /**
   * Creates the gradient object for the model that aggregates the gradient
   * updates.
   */
  virtual Gradient* get_gradient_object()=0;

  /**
   * Clones the model. Subclasses must implement it so that it calls the copy
   * constructor of the subclass in question.
   */
  virtual MlModel* clone()=0;

protected:
  /** Dimensions of the feature vector. */
  size_t dimensions;
  /** The learning rate function. */
  LearningRate* learning_rate;

  friend class Gradient;
};

/**
 * Ancestor for the gradient objects. Gradients objects collect the steps taken
 * by the learning algorithm in the feature weight space, and then update
 * the model.
 *
 * @see MlModel#get_gradient_object()
 */
class Gradient : virtual public Object {
public:
  Gradient(MlModel& parent);

  /** Resets the weights to 0. */
  virtual void reset()=0;

  /**
   * Updates the gradients.
   *
   * @param[in] features the features of the current item.
   * @param[in] output the output for the current item, i.e. score(features).
   * @param[in] mult a multiplier for the gradients. The LTR algorithm, for
   *                 example, passes d C/d s_i here.
   * @note double* const& = const double[]&
   */
  virtual void update(double* const& features, double output, double mult=1)=0;

  /**
   * Updates the parent and advances the learning rate function.
   *
   * @param[in] num_items the size of the training set. Useful for batch
   *                      learning, where we have to take the average gradient.
   */
  void update_parent(size_t num_items);

protected:
  /** Updates the parent -- subclasses must implement this method. */
  virtual void __update_parent(size_t num_items)=0;

  /** Reference to the parent. */
  MlModel& parent;
};

// TODO: learning_rate to class!
#endif

