#ifndef __FUNCTION_2_CLASS__
#define __FUNCTION_2_CLASS__

#include <iostream>

#include "./Condition.cpp"

using namespace std;

/**
 * Class representing a specific condition (Function_2) for grid evaluation.
 *
 * @tparam F The type of the grid coordinates.
 */
template <typename F = double>
class Function_2 : public Condition<F>
{
public:
    /**
     * The constant value used in the Function_2 condition.
     */
    const F C_value = 0.8;

    /**
     * Check if a condition specific to Function_2 is satisfied within a specified grid region.
     *
     * @param c The value used in the condition check.
     * @param x1 The starting x-coordinate of the grid region.
     * @param x2 The ending x-coordinate of the grid region.
     * @param y1 The starting y-coordinate of the grid region.
     * @param y2 The ending y-coordinate of the grid region.
     * @return True if the Function_2 condition is satisfied, false otherwise.
     */
    bool check(F &c, F x1, F x2, F y1, F y2) override
    {
        return ((pow(x1, 2) - pow(y1, 2)) <= c) && (c <= (pow(x2, 2) - pow(y2, 2)));
    }
};

#endif
