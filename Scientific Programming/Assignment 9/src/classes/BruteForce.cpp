#ifndef __BruteForce_CLASS__
#define __BruteForce_CLASS__

#include <vector>
#include <iostream>

#include "./Condition.cpp"

/**
 * Class representing a brute force approach to grid evaluation based on a condition.
 *
 * @tparam T The type of the grid coordinates.
 */
template <typename T = double>
class BruteForce
{
private:
    T **d;
    T C_value = 0.4;
    int numberOfCells;
    vector<vector<char>> *grid;

public:
    /**
     * Constructor for the BruteForce class.
     *
     * @param x1 The starting x-coordinate of the grid.
     * @param x2 The ending x-coordinate of the grid.
     * @param y1 The starting y-coordinate of the grid.
     * @param y2 The ending y-coordinate of the grid.
     * @param numberOfCells The number of cells in each dimension of the grid.
     * @param condition The condition object used to determine if a cell satisfies a condition.
     */
    BruteForce(T x1, T x2, T y1, T y2, const int numberOfCells, Condition<T> *condition) : numberOfCells(numberOfCells)
    {
        this->grid = new vector<vector<char>>(this->numberOfCells, vector<char>(this->numberOfCells, ' '));

        T deltaX = (x2 - x1) / this->numberOfCells;
        T deltaY = (y2 - y1) / this->numberOfCells;

        for (int i = 0; i < this->numberOfCells; i++)
        {
            for (int j = 0; j < this->numberOfCells; j++)
            {
                if (condition->check(this->C_value, x1 + i * deltaX, x1 + (i + 1) * deltaX, y1 + j * deltaY, y1 + (j + 1) * deltaY))
                {
                    (*this->grid)[i][j] = 'X';
                }
            }
        }
    }

    /**
     * Print the grid representation generated by the BruteForce approach.
     */
    void print()
    {
        for (int i = 0; i < this->numberOfCells; ++i)
        {
            for (int j = 0; j < this->numberOfCells; ++j)
            {
                cout << (*this->grid)[i][j] << ' ';
            }
            cout << endl;
        }
    }
};

#endif
