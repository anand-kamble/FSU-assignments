#include <iostream>

using namespace std;
/**
 * @brief A class representing a dynamic array with functionalities for mathematical operations.
 *
 * The Array class supports various operations such as element-wise multiplication,
 * addition, and manipulation of elements. It dynamically adjusts its capacity based
 * on the number of elements it holds.
 */
class Array
{
public:
    /**
     * @brief Pointer to the dynamically allocated array storing the elements of the Array.
     */
    int *data;
    /**
     * @brief Default constructor for the Array class.
     */
    Array();
    /**
     * @brief Destructor for the Array class.
     */
    ~Array();
    /**
     * @brief Constructor for the Array class with a specified size.
     * @param size The size of the array to be created.
     */
    Array(int size);
    /**
     * @brief Copy constructor for the Array class.
     * @param other Another instance of the Array class to copy.
     */
    Array(const Array &);
    /**
     * @brief Overloaded assignment operator for the Array class.
     * @param other Another instance of the Array class to assign.
     * @return A reference to the assigned Array.
     */
    Array &operator=(const Array &);

    /**
     * @brief Overloaded stream insertion operator for outputting the array.
     * @param in The output stream.
     * @param m The array to output.
     * @return The output stream.
     */
    friend ostream &operator<<(ostream &in, Array &m);
    /**
     * @brief Overloaded subscript operator for accessing elements of the array.
     * @param index The index of the element to access.
     * @return A pointer to the specified element.
     */
    int *operator[](int);

    /**
     * @brief Overloaded multiplication operator for multiplying an array by a scalar.
     * @param value The scalar value to multiply the array by.
     * @param m The array to be multiplied.
     * @return The resulting array.
     */
    friend Array &operator*(float, Array &);
    /**
     * @brief Overloaded multiplication operator for multiplying two arrays.
     * @param other Another instance of the Array class to multiply with.
     * @return A new Array containing the result of the multiplication.
     */
    Array &operator*(const Array &);
    /**
     * @brief Overloaded modulo operator for calculating the sum of element-wise multiplication.
     * @param other Another instance of the Array class to perform the operation with.
     * @return The sum of element-wise multiplication.
     */
    int operator%(const Array &);
    /**
     * @brief Appends elements to the end of the array.
     * @param v1,v2,v3,v4 The values of the elements to be added.
     */
    void push_back(int, int, int, int);
    /**
     * @brief Removes the last element from the array.
     */
    void pop_back();
    /**
     * @brief Removes an element at the specified index from the array.
     * @param index The index of the element to be removed.
     */
    void remove(int index);
    /**
     * @brief Inserts elements at the specified index in the array.
     * @param index The index at which to insert the elements.
     * @param v1,v2,v3,v4 The values of the elements to be inserted.
     */
    void insert(int, int, int, int, int);
    /**
     * @brief Gets the capacity of the array.
     * @return The capacity of the array.
     */
    int getCapacity() const;
    /**
     * @brief Gets the number of elements in the array.
     * @return The number of elements in the array.
     */
    int size() const;
    /**
     * @brief Clears the array, removing all elements.
     */
    void clear();
    /**
     * @brief Prints the elements of the array.
     */
    void print();

private:
    int numberOfElements; /**< Number of elements in the array. */

    int capacity; /**< Capacity of the array. */

    int columns; /**< Number of columns in the array. */

    int rows; /**< Number of rows in the array. */
};
