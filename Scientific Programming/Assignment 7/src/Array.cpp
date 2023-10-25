#include <iostream>
#include <cmath>
#include "Array.h"

using namespace std;

Array::Array()
{
    this->capacity = 0;
    this->numberOfElements = 0;
    this->data = nullptr;
}

Array::~Array()
{
    delete[] this->data;
}

Array::Array(int size)
{
    this->capacity = pow(2, ceil(log2(size)));
    this->data = new int[this->capacity * 4];
}

Array::Array(const Array &other) : numberOfElements(other.numberOfElements),
                                   capacity(other.capacity)
{
    data = new int[capacity * 4];
    for (int i = 0; i < numberOfElements * 4; ++i)
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
        this->data = new int[capacity * 4];
        for (int i = 0; i < this->numberOfElements * 4; ++i)
        {
            this->data[i] = other.data[i];
        }
    }
    return *this;
}

Array &Array::operator*(const Array &other)
{
    if (this->numberOfElements == other.numberOfElements)
    {
        Array &r = *(new Array(this->numberOfElements));
        for (int i = 0; i < this->numberOfElements; i++)
        {
            r.push_back(this->data[i * 4] * other.data[i * 4] + (this->data[i * 4 + 1] * other.data[i * 4 + 2]), this->data[i * 4] * other.data[i * 4 + 1] + (this->data[i * 4 + 1] * other.data[i * 4 + 3]), this->data[i * 4 + 2] * other.data[i * 4] + (this->data[i * 4 + 3] * other.data[i * 4 + 2]), this->data[i * 4 + 2] * other.data[i * 4 + 1] + (this->data[i * 4 + 3] * other.data[i * 4 + 3]));
        }
        return r;
    }
    return *this;
}

int Array::operator%(const Array &other)
{
    if (this->numberOfElements == other.numberOfElements)
    {
        Array &r = *(new Array(this->numberOfElements));
        for (int i = 0; i < this->numberOfElements; i++)
        {
            r.push_back(this->data[i * 4] * other.data[i * 4] + (this->data[i * 4 + 1] * other.data[i * 4 + 2]), this->data[i * 4] * other.data[i * 4 + 1] + (this->data[i * 4 + 1] * other.data[i * 4 + 3]), this->data[i * 4 + 2] * other.data[i * 4] + (this->data[i * 4 + 3] * other.data[i * 4 + 2]), this->data[i * 4 + 2] * other.data[i * 4 + 1] + (this->data[i * 4 + 3] * other.data[i * 4 + 3]));
        }

        int sum = 0;

        for (int i = 0; i < this->numberOfElements * 4; i++)
        {
            sum += this->data[i];
        }

        return sum;
    }
    return 0;
}

int *Array::operator[](int index)
{
    return &this->data[index * 4];
}

void Array::push_back(int v1, int v2, int v3, int v4)
{
    if (this->numberOfElements == this->capacity)
    {
        int newCap = (this->capacity) ? (this->capacity * 2) : this->capacity + 1;
        int *newData = new int[newCap * 4];

        for (int j = 0; j < this->numberOfElements * 4; ++j)
        {
            newData[j] = data[j];
        }

        delete[] data;
        data = newData;
        this->capacity = newCap;
    }

    data[(this->numberOfElements * 4)] = v1;
    data[(this->numberOfElements * 4) + 1] = v2;
    data[(this->numberOfElements * 4) + 2] = v3;
    data[(this->numberOfElements * 4) + 3] = v4;
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
            int *newData = new int[newCap * 4];

            for (int j = 0; j < this->numberOfElements * 4; ++j)
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
            data[(j * 4)] = data[((j + 1) * 4)];
            data[(j * 4) + 1] = data[((j + 1) * 4) + 1];
            data[(j * 4) + 2] = data[((j + 1) * 4) + 2];
            data[(j * 4) + 3] = data[((j + 1) * 4) + 3];
        }

        --this->numberOfElements;

        if (this->numberOfElements < this->capacity / 2)
        {
            int newCap = this->capacity / 2;
            int *newData = new int[newCap * 4];

            for (int j = 0; j < this->numberOfElements; ++j)
            {
                newData[(j * 4)] = data[(j * 4)];
                newData[(j * 4) + 1] = data[(j * 4) + 1];
                newData[(j * 4) + 2] = data[(j * 4) + 2];
                newData[(j * 4) + 3] = data[(j * 4) + 3];
            }

            delete[] data;

            data = newData;
            this->capacity = newCap;
        }
    }
}

void Array::insert(int index, int v1, int v2, int v3, int v4)
{
    if (index < 0 || index > this->numberOfElements)
    {
        throw std::out_of_range("Insufficient capacity, try using push_back instead.");
    }

    if (this->numberOfElements == this->capacity)
    {
        int newCap = this->capacity * 2;
        int *newData = new int[newCap * 4];

        for (int j = 0; j < this->numberOfElements * 4; ++j)
        {
            newData[j] = data[j];
        }

        delete[] data;

        data = newData;
        this->capacity = newCap;
    }

    for (int j = this->numberOfElements; j > index; --j)
    {
        data[(j * 4)] = data[((j - 1) * 4)];
        data[(j * 4) + 1] = data[((j - 1) * 4) + 1];
        data[(j * 4) + 2] = data[((j - 1) * 4) + 2];
        data[(j * 4) + 3] = data[((j - 1) * 4) + 3];
    }

    data[(index * 4)] = v1;
    data[(index * 4) + 1] = v2;
    data[(index * 4) + 2] = v3;
    data[(index * 4) + 3] = v4;

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

void Array::print()
{
    for (int i = 0; i < this->numberOfElements * 4; ++i)
    {
        cout << data[i] << " ";
        if (i > 0 && (i + 1) % 4 == 0)
            cout << endl;
    }
    cout << endl;
}

ostream &operator<<(ostream &in, Array &m)
{
    if (!m.numberOfElements)
    {
        in << "Matrix is empty" << endl;
        return in;
    }
    for (int i = 0; i < m.numberOfElements; i++)
    {
        in << "|" << m.data[(i * 4)] << " ";
        in << m.data[(i * 4) + 1] << "|";
        in << endl;
        in << "|" << m.data[(i * 4) + 2] << " ";
        in << m.data[(i * 4) + 3] << "|" << endl;
        in << endl;
    };
    return in;
};

Array &operator*(float value, Array &m)
{
    if (!m.numberOfElements)
        return m;
    for (int i = 0; i < m.numberOfElements * 4; ++i)
    {
        m.data[i] = m.data[i] * value;
    }
    return m;
}