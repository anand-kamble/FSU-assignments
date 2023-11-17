#include <iostream>

#include "./classes/Quadtree.cpp"
#include "./classes/BruteForce.cpp"
#include "./classes/Function_1.cpp"
#include "./classes/Function_2.cpp"
#include "./classes/Condition.cpp"
#include "./utils/utils.cpp"

using namespace std;

/**
 * Use a Quadtree to evaluate a condition on a given domain and print the time taken.
 *
 * @tparam T The type of the grid coordinates.
 * @param D A vector representing the domain of the grid.
 * @param N The number of cells in each dimension of the grid.
 * @param C The condition object used to determine if a cell satisfies a condition.
 */
template <typename T = double>
void UseQuadtree(vector<double> D, int N, Condition<T> *C)
{
    // Create a Quadtree object
    auto tree = new QuadTree(D[0], D[1], D[2], D[3], N, C);

    // Measure the time taken by the Quadtree to scan the grid
    auto time_Q = timeFunction([&]()
                               { tree->scan(); });

    // Print the time taken
    cout << "Time taken by Quad tree  : " << time_Q.count() << " ms"
         << " For N : " << N << endl;

    // Optionally print the Quadtree grid
    // tree->printTree();
}

/**
 * Use a Brute Force approach to evaluate a condition on a given domain and print the time taken.
 *
 * @tparam T The type of the grid coordinates.
 * @param D A vector representing the domain of the grid.
 * @param N The number of cells in each dimension of the grid.
 * @param C The condition object used to determine if a cell satisfies a condition.
 */
template <typename T = double>
void UseBruteForce(vector<double> D, int N, Condition<T> *C)
{
    BruteForce<double> *bruteForce;

    // Measure the time taken by the Brute Force approach to create the grid
    auto time_B = timeFunction([&]()
                               { bruteForce = new BruteForce(D[0], D[1], D[2], D[3], N, C); });

    // Print the time taken
    cout << "Time taken by Brute force: " << time_B.count() << " ms"
         << " For N : " << N << endl;

    // Optionally print the Brute Force grid
    // bruteForce->print();
}

int main()
{
    // Define the domain of the grid
    vector<double> Domain = {-1., 1., -1., 1.};

    // Define a set of grid sizes
    vector<int> N = {32, 64, 128, 256, 512, 1000};
    // vector<int> N = {32};

    // Create instances of Function_1 and Function_2
    Function_1<double> *func1 = new Function_1();
    Function_2<double> *func2 = new Function_2();

    // Iterate through grid sizes and evaluate conditions using both approaches
    for (auto &n : N)
    {
        // Evaluate Function_1 with Brute Force and Quadtree approaches
        UseBruteForce(Domain, n, func1);
        UseQuadtree(Domain, n, func1);

        // Evaluate Function_2 with Brute Force and Quadtree approaches
        UseBruteForce(Domain, n, func2);
        UseQuadtree(Domain, n, func2);
    }

    return 0;
}
