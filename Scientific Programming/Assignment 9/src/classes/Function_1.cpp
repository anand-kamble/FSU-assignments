#ifndef __FUNCTION_1_CLASS__
#define __FUNCTION_1_CLASS__
#include <iostream>

#include "./Condition.cpp"

using namespace std;

template <typename F = double>
class Function_1 : public Condition<F>
{
public:
    const F C_value = 0.8;
    bool check(F &c, F x1, F x2, F y1, F y2) override
    {
        return (pow(x1, 2) + pow(y1, 2)) <= c && c <= (pow(x2, 2) + pow(y2, 2));
    }
};

#endif