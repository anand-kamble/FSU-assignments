#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

/**
 * @brief Function which will allocate one block of memory for 2D array and print the difference between adresses.
 *
 * @return int
 */
int Question7()
{
    const int col = 10;
    const int row = 10;
    /* Allocating one block of memory for storing the 2D array.*/
    int **A = (int **)calloc((col * row), sizeof(int));

    for (size_t i = 0; i < 10; i++)
    {
        A[i] = (int *)&A[i * 10];
    }

    printf("ADDRESS OF A[0] : %p\n", &A[0]);
    printf("ADDRESS OF A[1] : %p\n", &A[1]);

    ptrdiff_t diff1 = &A[1] - &A[0];
    printf("DIFF BETWEEN A[0] & A[1] : %td\n", diff1);

    printf("ADDRESS OF A[0][3] : %p\n", &A[0][3]);
    printf("ADDRESS OF A[1][3] : %p\n", &A[1][3]);

    ptrdiff_t diff2 = &A[1][3] - &A[0][3];
    printf("DIFF BETWEEN A[0][3] & A[1][3] : %td\n", diff2);

    return 0;
}