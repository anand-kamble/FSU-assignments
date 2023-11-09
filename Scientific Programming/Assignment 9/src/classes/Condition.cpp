#ifndef __CONDITION_CLASS__
#define __CONDITION_CLASS__

#include <cmath>

using namespace std;

template <typename C = double>
class Condition
{
public:
    C C_value;
    virtual bool check(C &c, C x1, C x2, C y1, C y2) = 0;
};

#endif