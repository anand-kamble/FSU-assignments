/**
 * @class Array
 * @brief A dynamic array implementation with automatic resizing.
 */
class Array
{
public:

    int *data;
    /**
     * @brief Default constructor for Array class.
     *        Initializes an array with a capacity of 1.
     */
    Array();

    /**
     * @brief Destructor for Array class.
     *        Deallocates the memory used by the array.
     */
    ~Array();

    /**
     * @brief Parameterized constructor for Array class.
     *        Initializes an array with a specified size and adjusts the capacity
     *        to the nearest power of 2 greater than or equal to the given size.
     * @param size The initial size of the array.
     */
    Array(int size);

    Array(const Array &);

    Array &operator=(const Array &);

    void push_back(int value);

    void pop_back();

    void remove(int index);

    void insert(int value, int index);

    int getCapacity() const;
    int size() const;

    void clear();

private:
    int numberOfElements;
    /**
     * @brief The capacity of the array.
     */
    int capacity;

    /**
     * @brief The dynamic array to store integers.
     */

};
