#ifndef __Quadtree_CLASS__
#define __Quadtree_CLASS__

#include <iostream>
#include <vector>

#include "./QuadNode.cpp"
#include "../utils/utils.cpp"
#include "./Condition.cpp"

using namespace std;

template <typename T = double>
class QuadTree
{
private:
    // Pointer to the condition object used in the QuadTree
    Condition<T> *condition;

    // Pointer to the root QuadNode of the QuadTree
    QuadNode<T> *root;

    // Number of cells in the grid represented by the QuadTree
    int numberOfCells;

    // Constant value used in the QuadNode constructor
    T C_value = 0.4;

    // Vectors to store the grid indices i and j
    vector<int> i;
    vector<int> j;

    // Recursively divide nodes in the QuadTree
    void divideNodes(QuadNode<T> *toDivide)
    {
        if (arrayIncludesNullPointer<QuadNode<T> *>(toDivide->children, 4))
        {
            toDivide->divide();
        }
        else
        {
            for (int i = 0; i < 4; i++)
            {
                this->divideNodes(toDivide->children[i]);
            }
        }
    }

    // Helper function to print the QuadTree grid
    void print(QuadNode<T> *node)
    {
        vector<vector<char>> grid(this->numberOfCells, vector<char>(this->numberOfCells, ' '));

        for (size_t i = 0; i < this->i.size(); ++i)
        {
            if (this->i[i] >= 0 && this->i[i] < this->numberOfCells && this->j[i] >= 0 && this->j[i] < this->numberOfCells)
            {
                grid[this->i[i]][this->j[i]] = 'X';
            }
        }

        for (int i = 0; i < this->numberOfCells; ++i)
        {
            for (int j = 0; j < this->numberOfCells; ++j)
            {
                cout << grid[i][j] << ' ';
            }
            cout << endl;
        }
    }

public:
    /**
     * @brief Constructor for QuadTree class.
     *
     * @param x1 The minimum x-coordinate of the QuadTree bounding box.
     * @param x2 The maximum x-coordinate of the QuadTree bounding box.
     * @param y1 The minimum y-coordinate of the QuadTree bounding box.
     * @param y2 The maximum y-coordinate of the QuadTree bounding box.
     * @param numberOfCells Number of cells in the grid represented by the QuadTree.
     * @param condition Pointer to the condition object used in the QuadTree.
     */
    QuadTree(T x1, T x2, T y1, T y2, int numberOfCells, Condition<T> *condition)
    {
        this->condition = condition;
        this->numberOfCells = numberOfCells;
        this->root = new QuadNode<T>(this->C_value, this->condition, this->i, this->j);
        this->root->setPoints(x1, x2, y1, y2);

        // Ensure the QuadTree has enough leaf nodes to cover the grid
        while (this->root->getLeafNodes() < numberOfCells * numberOfCells * 4)
        {
            this->divideNodes(this->root);
        }
    }

    /**
     * @brief Scan the QuadTree and mark nodes based on the condition.
     *
     * @return true if the scan is successful, false otherwise.
     */
    bool scan()
    {
        return this->root->scan();
    }

    /**
     * @brief Set a new condition for the QuadTree.
     *
     * @param c New condition object to be set.
     */
    void setCondition(Condition<T> c)
    {
        this->condition = c;
    }

    /**
     * @brief Print the grid representation of the QuadTree.
     */
    void printTree()
    {
        this->print(this->root);
    }
};

#endif
