/*
Course    : SCIENTIFIC PROGRAMMING
Author    : Anand Kamble
Date      : 12th September 2023
Assignment: Integration

Question 2

*/

#include <stdio.h>
#include <math.h>

// Function to calculate sec(x).
/* 
    Arguments:
        theta : The value of angle theta.
 */
double sec(double theta)
{
    return 1 / cos(theta);
}

// Function to calculate average distance between two points.
/* 
    Arguments:
        theta : The value of angle theta
 */
double f(double theta)
{
    return ((sec(theta) * sec(theta) * sec(theta)) / 12) - ((sec(theta) * sec(theta) * sec(theta) * tan(theta) / 20));
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

    return integral * h * 8;
}

void main()
{

    const double PI = 3.14159265358979323846;

    const double EXACT_VALUE = (2 + sqrt(2) + (5 * log(sqrt(2) + 1))) / 15;

    /* Defining the intervals. */
    int intervals[5] = {20, 40, 80, 160, 320};

    double errors[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double results[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double orderOfConvergence[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

    /* Calculating the average distance between two random points by trapezoidal rule  */
    for (size_t i = 0; i < sizeof(intervals) / sizeof(intervals[0]); i++)
    {
        double avg = TrapezoidalIntegrate(f, 0, PI / 4, intervals[i]);
        double error = fabs(EXACT_VALUE - avg);
        results[i] = avg;
        errors[i] = error;
    }

    /* Calculating the order of convrgence */
    for (size_t i = 0; i < sizeof(intervals) / sizeof(intervals[0]); i++)
    {
        double convergence = (log(errors[i] / errors[((sizeof(intervals) / sizeof(intervals[0])) - 1) == i ? (i - 1) : (i + 1)]) / log(2));
        orderOfConvergence[i] = fabs(convergence);
    }

    printf("Number of  | Trapezoidal           | Error            | Order of \n");
    printf("Intervals  | Rule Result           |                  | Convergence\n");
    printf("--------------------------------------------------------------------- \n");

    for (size_t i = 0; i < sizeof(intervals) / sizeof(intervals[0]); i++)
    {
        printf("%3d        | %lf               | %lf        | %lf\n", intervals[i], results[i], errors[i], orderOfConvergence[i]);
    }
}