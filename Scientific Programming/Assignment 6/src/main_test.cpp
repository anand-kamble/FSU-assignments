#include <iostream>
#include "Array.h"
#include "Test.cpp"

int main()
{

    Array myArray;
    (new Test("Constructor creates an Array with capacity 0."))
        ->Expect(myArray.getCapacity())
        .ToBe(0);

    myArray.push_back(1);
    myArray.push_back(2);
    myArray.push_back(3);

    (new Test("Capacity is increased when pushing elements."))
        ->Expect(myArray.getCapacity())
        .ToBe(4);

    (new Test("Size is increased when pushing elements."))
        ->Expect(myArray.size())
        .ToBe(3);

    Array myArray2 = myArray;

    (new Test("Copy constructor creates an Array with the same elements as the original."))
        ->Expect(myArray2.data[1])
        .ToBe(2);

    Array myArray3;
    myArray3 = myArray;
    (new Test("Assignment operator creates an Array with the same elements as the original."))
        ->Expect(myArray3.data[1])
        .ToBe(2);

    myArray.push_back(10);
    myArray.push_back(20);
    myArray.push_back(30);

    (new Test("The capacity increases in powers of two as elements are pushed."))
        ->Expect(myArray.getCapacity())
        .ToBe(8);

    (new Test("New elements are added to the end of the Array."))
        ->Expect(myArray.data[myArray.size() - 1])
        .ToBe(30);

    myArray.pop_back();
    (new Test("Number of elements is decreased when popping elements."))
        ->Expect(myArray.size())
        .Expect(5);

    myArray.insert(100, 2);

    (new Test("Elements are inserted in the correct position."))
        ->Expect(myArray.data[2])
        .ToBe(100);

    myArray.remove(2);
    (new Test("Elements are removed from the correct position."))
        ->Expect(myArray.data[2])
        .ToBe(3);

    myArray.clear();
    (new Test("Array is cleared."))
        ->Expect(myArray.size())
       .ToBe(0);

    return 0;
}