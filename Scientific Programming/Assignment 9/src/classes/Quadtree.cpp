#ifndef __Quadtree_CLASS__
#define __Quadtree_CLASS__

#include <iostream>

#include "./QuadNode.cpp"
#include "../utils/utils.cpp"
#include "./Condition.cpp"

using namespace std;

template <typename T = double>
class QuadTree
{
private:
    int nodeCount = 1;

    Condition<T> *condition;

    int numberOfCells;

    T C_value = 0.4;

    QuadNode<T> *root;

    void divideNodes(QuadNode<T> *toDivide)
    {
        if (arrayIncludesNullPointer<QuadNode<T> *>(toDivide->children, 4))
        {
            nodeCount += 4;
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
    int count = 0;
    void print(QuadNode<T> *node)
    {

        if (arrayIncludesNullPointer<QuadNode<T> *>(node->children, 4))
        {
            this->count += 1;
            cout << (node->marked ? "X" : "_");
            if (this->count % this->numberOfCells == 0)
                cout << endl;
        }
        else
        {
            for (int i = 0; i < 4; i++)
            {
                this->print(node->children[i]);
            }
        }
    }

public:
    QuadTree(T x1, T x2, T y1, T y2, int numberOfCells, Condition<T> *condition)
    {
        this->condition = condition;
        this->numberOfCells = numberOfCells;
        this->root = new QuadNode<T>(this->C_value, this->condition);
        this->root->setPoints(x1, x2, y1, y2);

        while (this->root->getLeafNodes() < numberOfCells * numberOfCells * 4)
        {
            this->divide();
        }
    }

    bool scan()
    {
        return this->root->scan();
    }

    void divide()
    {
        this->divideNodes(this->root);
    }

    int getNodeCount()
    {
        return this->nodeCount;
    }

    void setCondition(Condition<T> c)
    {
        this->condition = c;
    }

    void printTree()
    {
        this->print(this->root);
    }
};

#endif