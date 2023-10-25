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
    myArray.push_back(11, 11, 11, 11);
    myArray.push_back(22, 22, 22, 22);
    myArray.push_back(33, 33, 33, 33);

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
        ->ExpectMatrix(myArray2[2])
        .ToBe(myArray2[2]);
    
     // Testing the copy constructor again.
    (new Test("Copy constructor creates an Array with the same elements as the original."))
        ->ExpectMatrix(myArray2[0])
        .ToBe(myArray2[0]);

    // Test the assignment operator.
    Array myArray3;
    myArray3 = myArray;
    (new Test("Assignment operator creates an Array with the same elements as the original."))
        ->ExpectMatrix(myArray3[1])
        .ToBe(myArray[1]);
   // Testing the assignment operator one more time.
    (new Test("Assignment operator creates an Array with the same elements as the original."))
        ->ExpectMatrix(myArray3[2])
        .ToBe(myArray[2]);

    // Add some more elements to the array.
    myArray.push_back(10, 10, 10, 10);
    myArray.push_back(20, 20, 20, 20);
    myArray.push_back(30, 30, 30, 30);

    // Test the capacity of the array, it should now be 8.
    (new Test("The capacity increases in powers of two as elements are pushed."))
        ->Expect(myArray.getCapacity())
        .ToBe(8);

    // Confirm that the elements are added to the end of the array.
    (new Test("New elements are added to the end of the Array."))
        ->ExpectMatrix(myArray[5])
        .ToBe(30, 30, 30, 30);

    // Delete the last element.
    myArray.pop_back();

    // Test the size of the array is decreased.
    (new Test("Number of elements is decreased when popping elements."))
        ->Expect(myArray.size())
        .ToBe(5);

    // Insert a new element at index 2 of the array.
    myArray.insert(2, 99, 99, 99, 99);

    // Test that the new element is inserted at the correct position.
    (new Test("Elements are inserted in the correct position."))
        ->ExpectMatrix(myArray[2])
        .ToBe(99, 99, 99, 99);

    // Remove the element at index 2 of the array.
    myArray.remove(2);

    // Test that the element is removed from the correct position.
    (new Test("Elements are removed from the correct position."))
        ->ExpectMatrix(myArray[2])
        .ToBe(33, 33, 33, 33);

    myArray = 6.142 * myArray;

    (new Test("Array can be multiplied by float."))
        ->ExpectMatrix(myArray[2])
        .ToBe(202, 202, 202, 202);

    // Pushing some elements into Array 2 to make its length equal to Array 1.
    myArray2.push_back(10, 10, 10, 10);
    myArray2.push_back(20, 20, 20, 20);

    // Multiplying two arrays.
    Array r = myArray * myArray2;

    // Testing the result of the multiplication.
    (new Test("Array can be multiplied by another array."))
        ->ExpectMatrix(r[0])
        .ToBe(1474, 1474, 1474, 1474);

    // Testing the result of the multiplication again.
    (new Test("Array can be multiplied by another array."))
        ->ExpectMatrix(r[1])
        .ToBe(5940,5940,5940,5940);

    // Calculating the dot product of two arrays.
    int dot = myArray % myArray2;
    // Testing the result.
    (new Test("“a % b” returns the sum of a[i]*b[i] over its elements."))
        ->Expect(dot)
        .ToBe(2348);

    // Repeating the tests for dot product.
    dot = myArray2 % myArray;
    (new Test("“a % b” returns the sum of a[i]*b[i] over its elements."))
        ->Expect(dot)
        .ToBe(384);

    // Clear the array.
    myArray.clear();
    // Test that the array is cleared.
    (new Test("Array is cleared."))
        ->Expect(myArray.size())
        .ToBe(0);

    return 0;
}
