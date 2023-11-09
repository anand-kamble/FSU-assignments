#ifndef __BruteForce_CLASS__
#define __BruteForce_CLASS__

template <typename T>
class BruteForce
{
private:
    float **d;
public:
    BruteForce(T x1, T x2, T y1, T y2, int numberOfCells, Condition<T> *condition)
    {
        this->d = new float[numberOfCells][numberOfCells];
        
    }
};

#endif