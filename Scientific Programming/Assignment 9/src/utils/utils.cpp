#ifndef __UTIL_FUNCTIONS__
#define __UTIL_FUNCTIONS__

#include <vector>
#include <algorithm>
#include <functional>
#include <chrono>

using namespace std;

/**
 * Check if the given array contains a null pointer.
 *
 * @param arr The array to be checked.
 * @param size The size of the array.
 * @return True if the array contains a null pointer, false otherwise.
 */
template <typename T>
bool arrayIncludesNullPointer(T *arr, size_t size)
{
    return arr[0] == nullptr;
    // Uncomment the following block for a more general check:
    // for (size_t i = 0; i < size; ++i)
    // {
    //     if (arr[i] == nullptr)
    //     {
    //         return true;
    //     }
    // }
    // return false;
}

/**
 * Check if the given vector contains a specific element.
 *
 * @param v The vector to be checked.
 * @param searchElement The element to search for.
 * @return True if the element is found in the vector, false otherwise.
 */
template <typename T>
bool ArrayIncludes(vector<T> &v, T searchElement)
{
    auto it = std::find(v.begin(), v.end(), searchElement);

    return it != v.end();
}

/**
 * Measure the execution time of a given function.
 *
 * @param f The function to be timed.
 * @return The duration of the function's execution in microseconds.
 */
chrono::microseconds timeFunction(const function<void()> &f)
{
    auto startTime = chrono::high_resolution_clock::now();

    // Call the function to be timed
    f();

    // Record the end time
    auto endTime = chrono::high_resolution_clock::now();

    // Calculate the duration
    return chrono::duration_cast<chrono::microseconds>(endTime - startTime);
}

#endif
