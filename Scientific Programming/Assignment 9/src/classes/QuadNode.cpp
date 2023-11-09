#ifndef __QuadNode_CLASS__
#define __QuadNode_CLASS__

#include <iostream>

#include "./Condition.cpp"

using namespace std;


template <typename N = double>
class QuadNode
{

public:
    N &c;
    Condition<N> *condition;
    QuadNode(N &c, Condition<N> *condition) : c(c), condition(condition){};

    QuadNode<N> *children[4] = {nullptr, nullptr, nullptr, nullptr};
    // X1, X2, Y1, Y2
    N *points = new N[4];
    bool marked = false;

    QuadNode<N> *getNode(int nodeNumber)
    {
        return this->children[nodeNumber];
    }

    void divide()
    {
        for (int i = 0; i < 4; i++)
        {
            this->children[i] = new QuadNode<N>(this->c, this->condition);
        }

        N midX = (this->points[0] + this->points[1]) / 2.0;
        N midY = (this->points[2] + this->points[3]) / 2.0;

        this->children[0]->setPoints(this->points[0], midX, this->points[2], midY);
        this->children[1]->setPoints(midX, this->points[1], this->points[2], midY);
        this->children[2]->setPoints(this->points[0], midX, midY, this->points[3]);
        this->children[3]->setPoints(midX, this->points[1], midY, this->points[3]);
    }

    void setPoints(N x1, N x2, N y1, N y2)
    {
        this->points[0] = x1;
        this->points[1] = x2;
        this->points[2] = y1;
        this->points[3] = y2;
    }

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
            this->marked = condition_result;
            return condition_result;
        }
        return false;
    }

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