#ifndef NEWTON_H
#define NEWTON_H

#include <iostream>
#include <cmath>
#include "solver.cpp"

using namespace std;

/**
 * @brief Class implementing Newton's method for finding roots of a function.
 *
 * This class inherits from the Solver base class and provides an implementation
 * of Newton's method for finding roots of a target function.
 *
 * @tparam T The data type for the variables and the result.
 */
template <typename T>
class Newton : public Solver<T>
{
public:
    /**
     * @brief Pointer to the derivative function of the target function.
     */
    T (*derivative)
    (T);

    /**
     * @brief Computed root of the equation.
     *
     * The root is the solution found by Newton's method.
     */
    T root;

    /**
     * @brief Constructor for the Newton class.
     *
     * @param targetFunction The target function for which the root needs to be found.
     * @param derivative The derivative of the target function.
     */
    Newton(T (*targetFunction)(T), T (*derivative)(T))
    {
        this->targetFunction = targetFunction;
        this->derivative = derivative;
    };

    /**
     * @brief Compute the root of the target function using Newton's method.
     *
     * Overrides the virtual method in the base class to provide the Newton's method implementation.
     *
     * @param initialGuess The initial guess for the root.
     * @param epsilon The desired level of accuracy.
     * @param maxIterations The maximum number of iterations allowed.
     * @return The computed root of the equation.
     */
    T ComputeRoot(T initialGuess, T epsilon, int maxIterations) override
    {
        T x = initialGuess;

        for (int i = 0; i < maxIterations; ++i)
        {
            T f = this->targetFunction(x);
            T df = this->derivative(x);

            x = x - f / df;

            if (abs(f) < epsilon)
            {
                cout << "[Newton] Root : " << x << ", Iterations : " << i + 1 << endl;
                this->root = x;
                return x;
            }
        }

        cerr << "Newton's method did not converge within the specified number of iterations." << endl;

        return initialGuess;
    }

    /**
     * @brief Verify the computed solution against the expected solution.
     *
     * Overrides the virtual method in the base class to provide verification of the computed solution.
     *
     * @param expected_solution The expected solution for verification.
     * @return The error between the expected and computed solutions.
     */
    T Verify(T expected_solution) override
    {
        auto error = abs(expected_solution - this->root);
        cout << "[Newton] Error: " << error << endl;
        return error;
    }
};

#endif
