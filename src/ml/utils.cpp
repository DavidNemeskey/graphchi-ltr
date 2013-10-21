#include "ml/utils.h"

//const double epsilon = 10e-9;
//const double DNaN = nan("");

bool double_equals(const double& d1, const double& d2) {
  return fabs(d1 - d2) < epsilon;
}
