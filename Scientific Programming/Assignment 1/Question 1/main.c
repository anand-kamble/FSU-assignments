/*
Course    : SCIENTIFIC PROGRAMMING
Author    : Anand Kamble
Date      : 12th September 2023
Assignment: Integration

Question 1

*/

#include <stdio.h>
#include <math.h>

// Creating a structure which will hold our results.
struct ResultsStructure
{
    double Simpsons[5];
    double Trapezoidal[5];
    double SimpsonsError[5];
    double TrapezoidalError[5];
    double SimpsonsOrderOfConvergence[5];
    double TrapezoidalOrderOfConvergence[5];
};

// Function to calculate sin(x).
/* 
    Arguments:
        x : value of which sin is to be calculated.
 */
double f(double x)
{
    return sin(x);
}

// Simpson's rule for numerical integration.
/* 
    Arguments:
        func : Function which we are applying simpsons rule to.
        lowerlimit : Lowerlimit of the integral.
        upperLimit : Upper limit of the integral.
 */
double SimpsonIntegrate(double (*func)(double), double lowerLimit, double upperLimit, int n)
{
    if (n % 2 != 0)
    {
        printf("Number of subintervals (n) must be even for Simpson's rule.\n");
        return 0.0;
    }

    double h = (upperLimit - lowerLimit) / n;
    double integral = func(lowerLimit) + func(upperLimit);

    for (int i = 1; i < n; i++)
    {
        double x = lowerLimit + i * h;
        if (i % 2 == 0)
        {
            integral += 2 * func(x);
        }
        else
        {
            integral += 4 * func(x);
        }
    }

    return integral * h / 3.0;
}

// Trapezoidal rule for numerical integration.
/* 
    Arguments:
        func : Function which we are applying trapezoidal rule to.
        lowerlimit : Lowerlimit of the integral.
        upperLimit : Upper limit of the integral.
 */
double TrapezoidalIntegrate(double (*func)(double), double lowerLimit, double upperLimit, int n)
{
    double h = (upperLimit - lowerLimit) / (n);
    double integral = 0.5 * (func(lowerLimit) + func(upperLimit));
    for (int i = 1; i < n; i++)
    {
        double x = lowerLimit + i * h;
        integral += func(x);
    }

    return integral * h;
}

// Function that calculates the error for Trapezoidal rule.
/* 
    Arguments:
        lowerlimit : Lowerlimit of the integral.
        upperLimit : Upper limit of the integral.
        n : number of intervals
 */
double TrapezoidalError(double lowerLimit, double upperLimit, int n)
{
    double a = upperLimit - lowerLimit;
    return fabs((a * a * a) / (12 * n * n));
}

// Function that calculates the error for Simpsons rule.
/* 
    Arguments:
        lowerlimit : Lowerlimit of the integral.
        upperLimit : Upper limit of the integral.
        n : number of intervals
 */
double SimpsonsError(double lowerLimit, double upperLimit, int n)
{
    double a = upperLimit - lowerLimit;
    return fabs((a * a * a * a * a) / (180 * n * n * n * n));
}

// Function that calculates the order of convergence.
/* 
    Arguments:
        e1 : absolute value of error.
        e2 : consecutive value of error.
 */
double OrderOfConvergence(double e1, double e2)
{
    return fabs(log(e1 / e2) / log(2));
}

void main()
{
    // Defining Pi since it is not included in the math library.
    const double PI = 3.14159265358979323846;

    /* Defining the limits of the integral. */
    double lowerLimit = 0.0;
    double upperLimit = PI;

    /* Defining the intervals. */
    int intervals[5] = {20, 40, 80, 160, 320};

    struct ResultsStructure Results;

    /* Calculating integrals and errors for each value of N */
    for (size_t i = 0; i < sizeof(intervals) / sizeof(intervals[0]); i++)
    {
        
        Results.Simpsons[i] = SimpsonIntegrate(f, lowerLimit, upperLimit, intervals[i]);
        Results.Trapezoidal[i] = TrapezoidalIntegrate(f, lowerLimit, upperLimit, intervals[i]);
        Results.SimpsonsError[i] = fabs(2 - Results.Simpsons[i]);
        Results.TrapezoidalError[i] = fabs(2 - Results.Trapezoidal[i]);
    }

    /* Calculating order of convergence. */
    for (size_t i = 0; i < ((sizeof(intervals) / sizeof(intervals[0]))); i++)
    {
        if (i != 0)
        {
            Results.SimpsonsOrderOfConvergence[i] = OrderOfConvergence(Results.SimpsonsError[i-1], Results.SimpsonsError[i]);
            Results.TrapezoidalOrderOfConvergence[i] = OrderOfConvergence(Results.TrapezoidalError[i-1], Results.TrapezoidalError[i]);
        }
    }

    /* Printing the results. */
    printf("\nNumber of   | Trapezoidal       | Error      | Convergence     | Simpson Rule      | Error      | Convergence\n");
    printf("Intervals   | Rule Result       |            | Order           | Result            |            | Order\n");
    printf("-------------------------------------------------------------------------------------------------------------\n");
    for (size_t i = 0; i < sizeof(intervals) / sizeof(intervals[0]); i++)
    {
        if (i == 0)
        {
            printf("%3d         | %lf          | %lf   |                 | %lf          | %lf   |    \n", intervals[i], Results.Trapezoidal[i], Results.TrapezoidalError[i], Results.Simpsons[i], Results.SimpsonsError[i]);
        }
        else
        {

            printf("%3d         | %lf          | %lf   | %lf        | %lf          | %2e   | %lf\n", intervals[i], Results.Trapezoidal[i], Results.TrapezoidalError[i], Results.TrapezoidalOrderOfConvergence[i], Results.Simpsons[i], Results.SimpsonsError[i], Results.SimpsonsOrderOfConvergence[i]);
        }
    }
}
