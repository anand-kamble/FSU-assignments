#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/**
 * @brief Function which calculates the sequence where any element is the sum of previous three elements.
 *
 * @param arr Array which holds the elements.
 * @param size Size of the Array.
 * @param startIndex Index from where the sequence will be calculated.
 */
void sum(long long arr[], int size, int startIndex)
{
    if (startIndex < size)
    {
        arr[startIndex] = arr[startIndex - 1] + arr[startIndex - 2] + arr[startIndex - 3];
        sum(arr, size, startIndex + 1);
    }else{
        return;
    }
}

int Question5()
{
    long long *arr = (long long *)calloc(100, sizeof(long long));
    if (arr[0] == 0 || arr[1] == 0 || arr[2] == 0)
    {
        arr[0] = 1;
        arr[1] = 2;
        arr[2] = 3;
    }

    sum(arr, 100, 3);
    printf("72nd term in the sequence is %lld.\n", arr[71]);

    free(arr);
    return 0;
}