#include <iostream>
#include <cmath>
#include "Array.h"

using namespace std;

Array::Array()
{
    /**
     * @details Initializes an array with a capacity of 0.
     */
    this->capacity = 0;
    this->numberOfElements = 0;
    this->data = nullptr;
}

Array::~Array()
{
    /**
     * @details Deallocates the memory used by the array.
     */
    delete[] this->data;
}

Array::Array(int size)
{
    /**
     * @details Initializes an array with a specified size and adjusts the capacity
     *          to the nearest power of 2 greater than or equal to the given size.
     * @param size The initial size of the array.
     */
    this->capacity = pow(2, ceil(log2(size)));
    this->data = new int[this->capacity];
}

/**
 * @brief Copy constructor for Array class.
 *        Creates a new array as a copy of the provided array.
 * @param other The array to be copied.
 */
Array::Array(const Array &other) : numberOfElements(other.numberOfElements),
                                   capacity(other.capacity)
{
    data = new int[capacity];
    for (int i = 0; i < numberOfElements; ++i)
    {
        data[i] = other.data[i];
    }
}

/**
 * @brief Assignment operator for Array class.
 *        Assigns the content of the provided array to this array.
 * @param other The array to be assigned.
 * @return A reference to the modified array.
 */
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

/**
 * @brief Adds a new element to the end of the array.
 *        If the array is full, it doubles its capacity before adding the element.
 * @param value The value to be added to the array.
 */
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

/**
 * @brief Removes the last element from the array.
 *        If the number of elements becomes less than half of the capacity,
 *        it reduces the capacity by half to conserve memory.
 */
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

/**
 * @brief Removes the element at the specified index from the array.
 *        If the number of elements becomes less than half of the capacity,
 *        it reduces the capacity by half to conserve memory.
 * @param index The index of the element to be removed.
 */
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

/**
 * @brief Gets the current capacity of the array.
 * @return The current capacity of the array.
 */
int Array::getCapacity() const
{
    return this->capacity;
}

/**
 * @brief Gets the current number of elements in the array.
 * @return The current number of elements in the array.
 */
int Array::size() const
{
    return this->numberOfElements;
}

/**
 * @brief Clears the array by deallocating memory and resetting size and capacity.
 */
void Array::clear()
{
    delete[] data;

    this->numberOfElements = 0;
    this->capacity = 0;
    data = nullptr;
}
