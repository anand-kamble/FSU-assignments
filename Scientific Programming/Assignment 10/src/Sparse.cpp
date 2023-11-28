#ifndef __SPARSE_CPP__
#define __SPARSE_CPP__
#include <vector>
using namespace std;

// Random number generator functions
void initializeRandomSeed()
{
    srand(static_cast<unsigned int>(time(nullptr)));
}

int generateRandomNumber(int M)
{
    return rand() % (M + 1);
}

struct element
{
    int col;
    int row;
    double value;
};

class SparseMatrix
{
private:
    vector<element> data;
    int ncol; // Number of columns
    int nrow; // Number of rows

public:
    SparseMatrix(int i, int j) : ncol(j), nrow(i) {}
    // Copy constructor
    SparseMatrix(const SparseMatrix &other)
        : data(other.data), ncol(other.ncol), nrow(other.nrow) {}

    // Assignment operator
    SparseMatrix &operator=(const SparseMatrix &other)
    {
        if (this != &other)
        { // Avoid self-assignment
            nrow = other.nrow;
            ncol = other.ncol;
            data = other.data;
        }
        return *this;
    }
    // Function to set a random value in the matrix
    void setRandomValue(int i, int j)
    {
        double randomValue = static_cast<double>(generateRandomNumber(100)) / 100.0; // Random value between 0 and 1
        set(i, j, randomValue);
    }

    // Get the i-th row, j-th column element
    double get(int i, int j) const
    {
        if (i < 0 || i >= nrow || j < 0 || j >= ncol)
        {
            cerr << "Error: Index out of bounds.\n";
            return 0.0; // Default value, you can modify it based on your requirements
        }

        int index = i * ncol + j;
        if (index < 0 || index >= static_cast<int>(data.size()))
        {
            return 0.0;
        }

        return data[index].value;
    }

    // Change the i-th row, j-th column element by value
    void set(int i, int j, double v)
    {
        if (i < 0 || i >= nrow || j < 0 || j >= ncol)
        {
            std::cerr << "Error: Index out of bounds.\n";
            return;
        }
        int index = i * ncol + j;
        if ((unsigned)index < data.size())
        {
            data[index].value = v;
        }
        else
        {
            element E = {i, j, v};
            data.push_back(E);
        }
    }
    // Shortcut to get i-th row, j-th column (same as get(int i, int j))
    double operator()(int i, int j) const
    {
        return get(i, j);
    }

    SparseMatrix operator+(const SparseMatrix &other) const
    {
        if (nrow != other.nrow || ncol != other.ncol)
        {
            cerr << "Error: Matrices must have the same dimensions for addition.\n";
            return *this; // Returning the original matrix for simplicity
        }

        SparseMatrix result(nrow, ncol);

        auto it1 = data.begin();
        auto it2 = other.data.begin();

        while (it1 != data.end() || it2 != other.data.end())
        {
            if (it1 != data.end() && (it2 == other.data.end() || it1->row < it2->row || (it1->row == it2->row && it1->col < it2->col)))
            {
                // Add element from the current matrix
                result.set(it1->row, it1->col, it1->value);
                ++it1;
            }
            else if (it2 != other.data.end() && (it1 == data.end() || it2->row < it1->row || (it2->row == it1->row && it2->col < it1->col)))
            {
                // Add element from the other matrix
                result.set(it2->row, it2->col, it2->value);
                ++it2;
            }
            else
            {
                // Add elements from both matrices (they are in the same location)
                result.set(it1->row, it1->col, it1->value + it2->value);
                ++it1;
                ++it2;
            }
        }

        return result;
    }

    SparseMatrix operator*(const SparseMatrix &other) const
    {
        if (ncol != other.nrow)
        {
            cerr << "Error: Number of columns in the first matrix must be equal to the number of rows in the second matrix for multiplication.\n";
            return SparseMatrix(0, 0); // Returning an empty matrix for simplicity
        }

        SparseMatrix result(nrow, other.ncol);

        for (int i = 0; i < nrow; ++i)
        {
            for (int j = 0; j < other.ncol; ++j)
            {
                double sum = 0.0;
                for (int k = 0; k < ncol; ++k)
                {
                    sum += get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }

        return result;
    }

    // Scalar multiplication operator for SparseMatrix * scalar
    template <typename U>
    friend SparseMatrix operator*(const SparseMatrix &mat, U scalar)
    {
        SparseMatrix result(mat.nrow, mat.ncol);

        for (const auto &elem : mat.data)
        {
            result.set(elem.row, elem.col, elem.value * static_cast<U>(scalar));
        }

        return result;
    }

    // Scalar multiplication operator for scalar * SparseMatrix
    template <typename U>
    friend SparseMatrix operator*(U scalar, const SparseMatrix &mat)
    {
        return mat * scalar;
    }

    // Overload << operator for output
    friend std::ostream &operator<<(std::ostream &os, const SparseMatrix &matrix)
    {
        os << matrix.nrow << " " << matrix.ncol << "\n";
        for (const auto &elem : matrix.data)
        {
            os << elem.row << " " << elem.col << " " << elem.value << "\n";
        }
        return os;
    }

    // Overload >> operator for input
    friend std::istream &operator>>(std::istream &is, SparseMatrix &matrix)
    {
        int rows, cols;
        is >> rows >> cols;

        matrix = SparseMatrix(rows, cols); // Create a new matrix with the specified dimensions

        element e;
        while (is >> e.row >> e.col >> e.value)
        {
            matrix.data.push_back(e);
        }

        return is;
    }
};
#endif
