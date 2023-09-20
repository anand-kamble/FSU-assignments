#include <stdio.h>

/**
 * @brief Function which prints elements of an array in reverse order.
 *
 * @param arr Array which has to be printed in reverse order.
 * @param size Size of the array.
 */
void reverse(float arr[], int size)
{
    if (size - 1 >= 0)
    {
        printf("%f\n", arr[size - 1]);
        reverse(arr, size - 1);
    }
    else
    {
        return;
    }
}

/**
 * @brief Function which initializes the array and call the 'reverse' function.
 *
 * @return int
 */
int Question4()
{
    float a[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    reverse(a, sizeof(a) / sizeof(a[0]));

    return 0;
}
