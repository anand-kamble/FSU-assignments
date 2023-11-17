#ifndef __CONDITION_CLASS__
#define __CONDITION_CLASS__

#include <cmath>

using namespace std;

/**
 * Generic Condition class for grid evaluation.
 *
 * @tparam C The type of the grid coordinates.
 */
template <typename C = double>
class Condition
{
public:
    /**
     * The value used in the condition check.
     */
    C C_value;

    /**
     * Virtual function for checking a condition within a specified grid region.
     *
     * @param c The value used in the condition check.
     * @param x1 The starting x-coordinate of the grid region.
     * @param x2 The ending x-coordinate of the grid region.
     * @param y1 The starting y-coordinate of the grid region.
     * @param y2 The ending y-coordinate of the grid region.
     * @return True if the condition is satisfied, false otherwise.
     */
    virtual bool check(C &c, C x1, C x2, C y1, C y2) = 0;
};

#endif
