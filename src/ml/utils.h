/** Contains utility functions. */
#include <cmath>

/** The infinitesimal error of the double ==. */
const double epsilon = 10e-9;

/**
 * NaN for doubles. As the STL only contains a float NAN constant and the
 * function that returns NaN has an argument (=> slow), we need this.
 */
const double DNaN = nan("");

/** Checks if @p d1 and @p d2 are equal with an error margin of @c epsilon. */
bool double_equals(const double& d1, const double& d2);

