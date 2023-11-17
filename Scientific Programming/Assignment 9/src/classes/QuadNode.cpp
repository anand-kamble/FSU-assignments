#ifndef __QuadNode_CLASS__
#define __QuadNode_CLASS__

#include <iostream>
#include <vector>
#include <cmath>

#include "./Condition.cpp"

using namespace std;

template <typename N = double>
class QuadNode
{
public:
    /**
     * @brief Constructor for QuadNode class.
     *
     * @param c Reference to the external variable.
     * @param condition Pointer to a condition object.
     * @param i Reference to the external vector i.
     * @param j Reference to the external vector j.
     */
    QuadNode(N &c, Condition<N> *condition, vector<int> &i, vector<int> &j) : c(c), condition(condition), i(i), j(j){};

    // Reference to the external variable
    N &c;

    // Pointer to a condition object
    Condition<N> *condition;

    // Child nodes of the QuadNode
    QuadNode<N> *children[4] = {nullptr, nullptr, nullptr, nullptr};

    // Array representing X1, X2, Y1, Y2 coordinates of the node
    N *points = new N[4];

    // Reference to external vectors i and j
    vector<int> &i;
    vector<int> &j;

    /**
     * @brief Get a specific child node by number.
     *
     * @param nodeNumber The number of the child node (0 to 3).
     * @return Pointer to the specified child node.
     */
    QuadNode<N> *getNode(int nodeNumber)
    {
        return this->children[nodeNumber];
    }

    /**
     * @brief Divide the QuadNode into four child nodes.
     */
    void divide()
    {
        for (int i = 0; i < 4; i++)
        {
            this->children[i] = new QuadNode<N>(this->c, this->condition, this->i, this->j);
        }

        N midX = (this->points[0] + this->points[1]) / 2.0;
        N midY = (this->points[2] + this->points[3]) / 2.0;

        this->children[0]->setPoints(this->points[0], midX, this->points[2], midY);
        this->children[1]->setPoints(midX, this->points[1], this->points[2], midY);
        this->children[2]->setPoints(this->points[0], midX, midY, this->points[3]);
        this->children[3]->setPoints(midX, this->points[1], midY, this->points[3]);
    }

    /**
     * @brief Set the X1, X2, Y1, Y2 coordinates of the node.
     *
     * @param x1 The minimum x-coordinate.
     * @param x2 The maximum x-coordinate.
     * @param y1 The minimum y-coordinate.
     * @param y2 The maximum y-coordinate.
     */
    void setPoints(N x1, N x2, N y1, N y2)
    {
        this->points[0] = x1;
        this->points[1] = x2;
        this->points[2] = y1;
        this->points[3] = y2;
    }

    /**
     * @brief Scan the QuadNode and mark nodes based on the condition.
     *
     * @return true if the scan is successful, false otherwise.
     */
    bool scan()
    {
        if (!arrayIncludesNullPointer<QuadNode<N> *>(this->children, 4))
        {
            for (int i = 0; i < 4; i++)
            {
                this->children[i]->scan();
            }
        }
        else
        {
            bool condition_result = condition->check(this->c, this->points[0], this->points[1], this->points[2], this->points[3]);
            if (condition_result)
            {
                N i = (this->points[0] - (-1)) / (this->points[1] - this->points[0]);
                N j = (this->points[2] - (-1)) / (this->points[3] - this->points[2]);
                this->i.push_back(round(i));
                this->j.push_back(round(j));
            }
            return condition_result;
        }
        return false;
    }

    /**
     * @brief Get the number of leaf nodes in the QuadNode.
     *
     * @return The number of leaf nodes.
     */
    int getLeafNodes()
    {
        int leafNodes = 0;
        if (arrayIncludesNullPointer<QuadNode<N> *>(this->children, 4))
        {
            leafNodes += 4;
        }
        else
        {
            for (int i = 0; i < 4; i++)
            {
                leafNodes += this->children[i]->getLeafNodes();
            }
        }
        return leafNodes;
    }
};

#endif
