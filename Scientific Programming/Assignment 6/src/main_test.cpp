#include <iostream>
#include "Array.h"
#include "Test.cpp"

int main()
{
    // Create an array with default constructor and test its capacity, which should be 0.
    Array myArray;
    (new Test("Constructor creates an Array with capacity 0."))
        ->Expect(myArray.getCapacity())
        .ToBe(0);

    // Push some data to the array
    myArray.push_back(1);
    myArray.push_back(2);
    myArray.push_back(3);

    // Test the capacity of the array, it should now be 4. Since we pushed 3 elements. 
    // and the closest value which is the power of 2 and greater than 3 is 4.
    (new Test("Capacity is increased when pushing elements."))
        ->Expect(myArray.getCapacity())
        .ToBe(4);

    // Test the size of the array, it should now be 3.
    (new Test("Size is increased when pushing elements."))
        ->Expect(myArray.size())
        .ToBe(3);

    // Test the copy constructor.
    Array myArray2 = myArray;
    (new Test("Copy constructor creates an Array with the same elements as the original."))
        ->Expect(myArray2.data[1])
        .ToBe(2);

    // Test the assignment operator.
    Array myArray3;
    myArray3 = myArray;
    (new Test("Assignment operator creates an Array with the same elements as the original."))
        ->Expect(myArray3.data[1])
        .ToBe(2);


    // Add some more elements to the array.
    myArray.push_back(10);
    myArray.push_back(20);
    myArray.push_back(30);

    // Test the capacity of the array, it should now be 8.
    (new Test("The capacity increases in powers of two as elements are pushed."))
        ->Expect(myArray.getCapacity())
        .ToBe(8);

    // Confirm that the elements are added to the end of the array.
    (new Test("New elements are added to the end of the Array."))
        ->Expect(myArray.data[myArray.size() - 1])
        .ToBe(30);

    // Delete the last element.
    myArray.pop_back();

    // Test the size of the array is decreased.
    (new Test("Number of elements is decreased when popping elements."))
        ->Expect(myArray.size())
        .Expect(5);

    // Insert a new element at index 2 of the array.
    myArray.insert(100, 2);

    // Test that the new element is inserted at the correct position.
    (new Test("Elements are inserted in the correct position."))
        ->Expect(myArray.data[2])
        .ToBe(100);

    // Remove the element at index 2 of the array.
    myArray.remove(2);
    
    // Test that the element is removed from the correct position.
    (new Test("Elements are removed from the correct position."))
        ->Expect(myArray.data[2])
        .ToBe(3);

    // Clear the array.
    myArray.clear();
    // Test that the array is cleared.
    (new Test("Array is cleared."))
        ->Expect(myArray.size())
       .ToBe(0);

    return 0;
}