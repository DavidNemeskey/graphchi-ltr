#include "ml/neural_net_activation.h"

#include <cmath>

ActivationAct::ActivationAct(const Activation& parent) : parent_(parent) {}
ActivationDeriv::ActivationDeriv(const Activation& parent) : parent_(parent) {}

Activation::Activation(const std::string& name) : act_(*this), deriv_(*this), name(name) {}

Sigma::Sigma(double K) : Activation("Sigma"), K(K) {}

double Sigma::activation(double x) const {
  return 1 / (1 + exp(-K * x));
}

double Sigma::derivative(double sigma_x) const {
//  double sigma_x = *this(x);
  return sigma_x * (1 - sigma_x);
}

double Sigma::logit(double x) const {
  return log(x) - log(1 - x);
}
