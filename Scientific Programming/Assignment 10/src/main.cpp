#include <iostream>
#include <fstream>
#include "Sparse.cpp"

using namespace std;

int main()
{
    initializeRandomSeed();

    // Test the scale * matrix operator
    SparseMatrix matrix1(3, 3);
    matrix1.set(0, 0, 1.0);
    matrix1.set(1, 1, 2.0);
    matrix1.set(2, 2, 3.0);

    std::cout << "Original Matrix:\n";
    std::cout << matrix1;

    SparseMatrix scaledMatrix = 2.5 * matrix1;

    std::cout << "Scaled Matrix:\n";
    std::cout << scaledMatrix;

    // Test the matrix * matrix operator
    SparseMatrix matrix2(3, 3);
    matrix2.set(0, 0, 0.5);
    matrix2.set(1, 1, 1.5);
    matrix2.set(2, 2, 2.5);

    std::cout << "Matrix 1:\n";
    std::cout << matrix1;

    std::cout << "Matrix 2:\n";
    std::cout << matrix2;

    SparseMatrix productMatrix = matrix1 * matrix2;

    std::cout << "Matrix 1 * Matrix 2:\n";
    std::cout << productMatrix;

    // Test the matrix + matrix operator
    SparseMatrix sumMatrix = matrix1 + matrix2;

    std::cout << "Matrix 1 + Matrix 2:\n";
    std::cout << sumMatrix;

    // Test the << and >> operators
    SparseMatrix matrix3(2, 2);
    matrix3.set(0, 1, 0.8);
    matrix3.set(1, 0, 0.2);

    std::cout << "Original Matrix 3:\n";
    std::cout << matrix3;

    // Save matrix3 to a file
    std::ofstream outFile("matrix3.txt");
    if (!outFile)
    {
        std::cerr << "Error: Could not open file for writing.\n";
        return 1;
    }

    outFile << matrix3;
    outFile.close();

    std::cout << "Matrix 3 written to file 'matrix3.txt'.\n";

    // Read matrix3 back from the file
    std::ifstream inFile("matrix3.txt");
    if (!inFile)
    {
        std::cerr << "Error: Could not open file for reading.\n";
        return 1;
    }

    SparseMatrix newMatrix3(0, 0);
    inFile >> newMatrix3;
    inFile.close();

    std::cout << "Matrix 3 read from file:\n";
    std::cout << newMatrix3;

    return 0;
}