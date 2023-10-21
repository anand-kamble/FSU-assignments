#include <iostream>

using namespace std;
class Array
{
public:
    int *data;

    Array();

    ~Array();

    Array(int size);

    Array(const Array &);

    Array &operator=(const Array &);

    friend ostream &operator<<(ostream &in, Array &m);

    int *operator[](int);

    void push_back(int,int,int,int);

    void pop_back();

    void remove(int index);

    void insert(int, int, int, int, int);

    int getCapacity() const;

    int size() const;

    void clear();

    void print();
private:
    int numberOfElements;

    int capacity;

    int columns;

    int rows;
};
