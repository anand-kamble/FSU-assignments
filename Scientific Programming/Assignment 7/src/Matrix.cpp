#include <iostream>

using namespace std;

template <typename ElementType>
class Matrix
{
private:
    ElementType *matrix;

public:
    int rows;
    int cols;
    Matrix()
    {
        rows = 0;
        cols = 0;
        matrix = nullptr;
    }
    Matrix(int r, int c)
    {
        rows = r;
        cols = c;
        matrix = new ElementType[rows * cols];
    }

    void populateMatrix(int i, int j, ElementType value)
    {
        i <= this->cols &&j <= this->rows ? this->matrix[this->cols * i + j] = value : throw std::out_of_range("Index out of range");
    }

    friend ostream &operator<<(ostream &in, Matrix<ElementType> &m)
    {
        if (!m.rows || !m.cols)
        {
            in << "Matrix is empty" << endl;
            return in;
        }
        for (unsigned int i = 0; i < m.cols; i++)
        {
            for (unsigned int j = 0; j < m.rows; j++)
            {
                in << m(i, j) << " ";
            }
            in << endl;
        };
        return in;
    };

    ~Matrix()
    {
        delete[] matrix;
    };

    ElementType &operator()(int i, int j)
    {
        return this->matrix[i * this->cols + j];
    }
};
