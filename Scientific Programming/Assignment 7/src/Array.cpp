#include <iostream>
#include <cmath>
#include "Array.h"

// #define A(x, y, z) (A[(x * n) + y])

using namespace std;

Array::Array()
{
    this->capacity = 0;
    this->numberOfElements = 0;
    this->data = nullptr;
    this->columns = 2;
    this->rows = 2;
    
}

Array::~Array()
{
    delete[] this->data;
}

Array::Array(int size)
{
    this->capacity = pow(2, ceil(log2(size)));
    this->data = new int[this->capacity*this->columns*this->rows];
}

Array::Array(const Array &other) : numberOfElements(other.numberOfElements),
                                   capacity(other.capacity)
{
    data = new int[capacity];
    for (int i = 0; i < numberOfElements; ++i)
    {
        data[i] = other.data[i];
    }
}

Array &Array::operator=(const Array &other)
{
    if (this != &other)
    {
        delete[] data;
        this->numberOfElements = other.numberOfElements;
        this->capacity = other.capacity;
        this->data = new int[capacity];
        for (int i = 0; i < this->numberOfElements; ++i)
        {
            this->data[i] = other.data[i];
        }
    }
    return *this;
}

void Array::setArrayShape(int columns, int rows)
{
    this->columns = columns;
    this->rows = rows;
}

void Array::push_back(int value)
{
    if (this->numberOfElements == this->capacity)
    {
        int newCap = (this->capacity) ? (this->capacity * 2) : this->capacity + 1;
        int *newData = new int[newCap];

        for (int j = 0; j < this->numberOfElements; ++j)
        {
            newData[j] = data[j];
        }

        delete[] data;
        data = newData;
        this->capacity = newCap;
    }

    data[this->numberOfElements] = value;
    ++this->numberOfElements;
    return;
}

void Array::pop_back()
{
    if (this->numberOfElements > 0)
    {
        --this->numberOfElements;
        if (this->numberOfElements < (this->capacity / 2))
        {
            int newCap = this->capacity / 2;
            int *newData = new int[newCap];

            for (int j = 0; j < this->numberOfElements; ++j)
            {
                newData[j] = data[j];
            }

            delete[] data;

            data = newData;
            this->capacity = newCap;
        }
    }
}

void Array::remove(int index)
{
    if (index >= 0 && index < this->numberOfElements)
    {
        for (int j = index; j < this->numberOfElements - 1; ++j)
        {
            data[j] = data[j + 1];
        }

        --this->numberOfElements;

        if (this->numberOfElements < this->capacity / 2)
        {
            int newCap = this->capacity / 2;
            int *newData = new int[newCap];

            for (int j = 0; j < this->numberOfElements; ++j)
            {
                newData[j] = data[j];
            }

            delete[] data;

            data = newData;
            this->capacity = newCap;
        }
    }
}

void Array::insert(int value, int index)
{
    if (index < 0 || index > this->numberOfElements)
    {
        throw std::out_of_range("Insufficient capacity, try using push_back instead.");
    }

    if (this->numberOfElements == this->capacity)
    {
        int newCap = this->capacity * 2;
        int *newData = new int[newCap];

        for (int j = 0; j < this->numberOfElements; ++j)
        {
            newData[j] = data[j];
        }

        delete[] data;

        data = newData;
        this->capacity = newCap;
    }

    for (int j = this->numberOfElements; j > index; --j)
    {
        data[j] = data[j - 1];
    }

    data[index] = value;

    ++this->numberOfElements;
}

int Array::getCapacity() const
{
    return this->capacity;
}

int Array::size() const
{
    return this->numberOfElements;
}

void Array::clear()
{
    delete[] data;

    this->numberOfElements = 0;
    this->capacity = 0;
    data = nullptr;
}
