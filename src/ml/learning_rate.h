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
 * Learning rate strategies.
 *
 * @warning This file uses features introduced in the C++11 revision of the
 *          standard. Use a compatible compiler with the appropriate compilation
 *          options (e.g. -std=c++0x or -std=c++11 for GCC).
 */

// TODO: Composite: delete

#include <stdexcept>
#include <vector>
#include <string>

/** The ancestor of all learning rate strategies. */
class LearningRate {
public:
  LearningRate(double learning_rate=1);

  /** Returns the current value of the learning rate function. */
  inline double get() const { return learning_rate; }

  /**
   * Advances the function to the next point. Usually called after an
   * iteration. Subclasses must override this method to suit their needs.
   *
   * @return true if we didn't reach the end of the domain; @c false otherwise.
   *              The learning rate IS changed the first time this function
   *              returns @c false; after that, it should remain constant.
   */
  virtual bool advance()=0;

  /** Resets the object to its starting state. */
  virtual void reset()=0;

  /**
   * Clones the model. Subclasses must implement it so that it calls the copy
   * constructor of the subclass in question.
   */
  virtual LearningRate* clone()=0;

protected:
  /** The current value of the learning rate function. */
  double learning_rate;
};

/** A constant learning rate. */
class ConstantLearningRate : public LearningRate {
public:
  /** @throws std::invalid_argument if not <tt>0 < learning_rate</tt>. */
  ConstantLearningRate(double learning_rate) throw (std::invalid_argument);

  inline bool advance() { return true; }
  inline void reset() {};
  ConstantLearningRate* clone();
};

/** Linearly decreasing learning rate in an interval. */
class LinearLearningRate : public LearningRate {
public:
  /**
   * Constructor.
   *
   * @param starting_value the start point of the interval.
   * @param decrease the step size, subtracted from the learning rate at each
   *                 step.
   * @param ending_value the end point of the interval.
   * @throws std::invalid_argument if @p starting_value or @p ending_value is
   *                               less than @c 0.
   */
  LinearLearningRate(double starting_value, double decrease,
                     double ending_value) throw (std::invalid_argument);

  bool advance();
  void reset();
  LinearLearningRate* clone();

private:
  /** The start point of the interval. */
  double start_point;
  /** The learning rate is decreased by this amount every step. */
  double decrease;
  /** The end point of the interval. */
  double end_point;
};

/**
 * Executes several learning rate strategies in succession: the first one is
 * followed until its @c advance() method returns @c false, then the second one,
 * and so on.
 */
class CompositeLearningRate : public LearningRate {
protected:
  CompositeLearningRate(CompositeLearningRate& orig);

public:
  /**
   * Constructor.
   *
   * @param parts the learning rate functions we iterate through.
   * @throws std::invalid_argument if parts is empty or contains only @c NULLs.
   */
  CompositeLearningRate(std::vector<LearningRate*> parts)
    throw (std::invalid_argument);

  bool advance();
  void reset();
  CompositeLearningRate* clone();

private:
  /** The strategies we iterate through. */
  std::vector<LearningRate*> parts;
  /** The index of the currently active strategy. */
  size_t index;
};

/**
 * A "reflection" method: creates a learning rate object by name.
 * @p reflection_name must be in the format
 * <tt>learning_rate_function [:param]* </tt>, where the <tt>param</tt>s are
 * the parameters to the learning rate strategy's constructor. In addition,
 * CompositeLearningRate separates its parts by <tt>;</tt>.
 *
 * @note This function does not handle nest CompositeLearningRate definitions.
 *
 * @throws std::invalid_argument if the constructor of the learning rate
 *                               strategy to be instantiated does.
 */
LearningRate* create_learning_rate_function(const std::string& reflection_name)
  throw (std::invalid_argument);

