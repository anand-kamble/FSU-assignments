#include <iostream>

#include "./classes/Quadtree.cpp"
#include "./classes/Function_1.cpp"

using namespace std;

void UseQuadtree(int N)
{
    cout << "Making tree with " << N << "*" << N << " cells." << endl;

    Function_1<double> *func1 = new Function_1();
    auto tree = new QuadTree(-1., 1., -1., 1., N, func1);

    tree->scan();

    tree->printTree();
}

int main()
{
    UseQuadtree(32);
    return 0;
}