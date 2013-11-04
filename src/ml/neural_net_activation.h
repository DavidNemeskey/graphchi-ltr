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
 * Activation functions for neural net models.
 */

class ActivationAct;
class ActivationDeriv;

/**
 * The descendant of the activation function objects. Users call the
 * <tt>act()</tt> and <tt>deriv()</tt> functions to get the functors for the
 * activation function and its derivative, respectively. This is a bit
 * inconvenient, since one has to write e.g. <tt>a.act()(x)</tt>, but that's the
 * best we can do without a working @c mem_fun.
 *
 * Descendants must implement the <tt>activation()</tt> and the
 * <tt>derivative()</tt> methods. 
 */
class Activation {
protected:
  Activation(Activation& orig);

public:
  Activation();
  virtual ~Activation();

  /** The activation function... */
  virtual double activation(double x) const=0;
  /**
   * ... and its derivative.
   *
   * @param act_x the value returned by the <tt>activation()</tt> function.
   */
  virtual double derivative(double act_x) const=0;
  /**
   * Returns the object, whose <tt>operator()</tt> is the activation function.
   */
  inline const ActivationAct& act() const { return *act_; }
  /**
   * Returns the object, whose <tt>operator()</tt> is the derivative of the
   * activation function.
   */
  inline const ActivationDeriv& deriv() const { return *deriv_; }

  /**
   * Clones the activation function. Subclasses must implement it so that it
   * calls the copy constructor of the subclass in question.
   */
  virtual Activation* clone()=0;

private:
  /** The object, whose <tt>operator()</tt> is the activation function. */
  const ActivationAct* act_;
  /**
   * The object, whose <tt>operator()</tt> is the derivative of the activation
   * function.
   */
  const ActivationDeriv* deriv_;
};

struct ActivationAct {
  ActivationAct(const Activation& parent);
  inline double operator()(double x) const { return parent_.activation(x); }

  /** The Activation object whose activation function this object calls. */
  const Activation& parent_;
};

struct ActivationDeriv {
  ActivationDeriv(const Activation& parent);
  inline double operator()(double x) const { return parent_.derivative(x); }

  /** The Activation object whose derivative function this object calls. */
  const Activation& parent_;
};

class Sigma : public Activation {
public:
  /** @param K the parameter for @c sigma. */
  Sigma(double K);

  double activation(double x) const;
  double derivative(double sigma_x) const;

  /** The inverse of the sigma (logistic) function: the logit function. */
  double logit(double x) const;

  Sigma* clone();

private:
  /** Parameter for @c sigma. */
  double K;
};

